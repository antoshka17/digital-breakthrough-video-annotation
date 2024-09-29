import os
import torch
from torch import nn
import numpy as np
import cv2
import albumentations as A
from models_vit import VTSum_BLIP_TT
from models_vit import load_blip_pretrained_checkpoint
# from models_vit import load_checkpoint
# from models_vit import TimmModel
# from vit import VisionTransformer
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from transformers import ViTImageProcessor, ViTForImageClassification
# import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
# from IPython.display import Video
from googletrans import Translator


class Interester:
	def __init__(self, interest_model_path: str) -> None:
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.interest_model = VTSum_BLIP_TT()
		self.interest_model, msg = load_blip_pretrained_checkpoint(self.interest_model, interest_model_path)
		self.interest_model = self.interest_model.to(self.device)
		self.interest_model.eval()
		
		vit_model_name = "google/vit-base-patch16-224"
		self.processor = ViTImageProcessor.from_pretrained(vit_model_name)
		self.vit_model = ViTForImageClassification.from_pretrained(vit_model_name)
		self.vit_model = self.vit_model.to(self.device)
		self.vit_model.head = nn.Identity()
		self.vit_model.classifier = nn.Identity()
		
		caption_model_name = "nlpconnect/vit-gpt2-image-captioning"
		self.caption_model = VisionEncoderDecoderModel.from_pretrained(caption_model_name)
		self.feature_extractor = ViTImageProcessor.from_pretrained(caption_model_name)
		self.tokenizer = AutoTokenizer.from_pretrained(caption_model_name)
		
		self.caption_model.to(self.device)
		
	@staticmethod
	def read_video(video_path, n_frames=15000):
		cap = cv2.VideoCapture(video_path)
		al = []
		i = 0
		while cap.isOpened() and i < n_frames:
			ret, frame = cap.read()
			if frame is None:
				break
			arr = np.array(frame)
			al.append(arr)
			i += 1
			
		return np.array(al)
	
	@staticmethod
	def pre_video(video_embeddings, max_frames=8192):
		video_mask = torch.ones(video_embeddings.size(0), dtype=torch.long)
		
		if video_embeddings.size(0) > max_frames:
			video_embeddings = video_embeddings[:max_frames]
			video_mask = video_mask[:max_frames]
		
		return video_embeddings, video_mask
	
	def predict_step(self, model, feature_extractor, tokenizer, images):
		max_length = 32
		num_beams = 4
		gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
		
		pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
		pixel_values = pixel_values.to(self.device)
		
		output_ids = model.generate(pixel_values, **gen_kwargs)
		
		preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
		preds = [pred.strip() for pred in preds]
		return preds
	
	def make_embeddings(self, model, np_video, preprocess='classic', processor=None):
		model.eval()
		embeddings = []
		transforms_val = A.Compose([
			A.Resize(224, 224),
			A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
		])
		for nframe in tqdm(range(np_video.shape[0])):
			frame = np_video[nframe, ...]
			if preprocess == 'classic':
				frame = transforms_val(image=frame)['image']
				frame = torch.FloatTensor(frame)
				frame = frame.view(1, *frame.shape)
				frame = frame.permute(0, 3, 1, 2)
				frame = frame.to(self.device)
				with torch.no_grad():
					embed = model(frame)
			else:
				frame = processor(images=frame, return_tensors='pt')
				frame = frame.to(self.device)
				with torch.no_grad():
					embed = model(**frame).logits
			embeddings.append(embed)
		
		embeddings = torch.cat(embeddings, dim=0)
		embeddings = embeddings.view(embeddings.shape[0], 1, embeddings.shape[1])
		
		return embeddings
	
	@staticmethod
	def translator_translate(translator, word: str) -> str:
		result = translator.translate(word, dest='en')
		return result.text
	
	@staticmethod
	def med(x, window_size):
		return np.array([np.median(x[i:i + window_size]) for i in range(0, len(x) - window_size, 5)])
	
	def get_interests_summary(self, video_path: str, translate: bool = False,
	                          display: bool = False, num_interesting_frames: int = 3):
		np_video = self.read_video(video_path)
		
		self.vit_model.eval()
		
		video_embedding = self.make_embeddings(self.vit_model, np_video, preprocess='unclassic', processor=self.processor)
		video_embedding, video_mask = self.pre_video(video_embedding)
		video_embedding = video_embedding.permute(1, 0, 2)
		
		video_mask = video_mask.view(1, *video_mask.shape).to(self.device)

		res = self.interest_model.generate(video_embedding, video_mask, sample=True, num_beams=1)
		x = res[1][0, :, 0].detach().cpu().numpy()
		x = self.med(x, 20)
		
		if display:
			from IPython import display
			display.display(display.Video(video_path, embed=True))
		
		window_size = len(x) // num_interesting_frames
		captions, interesting_frames_indexes = [], []
		for ninteresting_frame in range(num_interesting_frames):
			interesting_frame = np_video[int((x[ninteresting_frame * window_size:(ninteresting_frame + 1) * window_size].argmax() + ninteresting_frame * window_size) * len(np_video) / len(x)), ...]
			caption = self.predict_step(self.caption_model, self.feature_extractor, self.tokenizer, interesting_frame)
			
			captions.append(caption[0])
			interesting_frames_indexes.append(int((x[ninteresting_frame * window_size:(ninteresting_frame + 1) * window_size].argmax() + ninteresting_frame * window_size) * len(
				np_video) / len(x)))
		
		intt = np.array([int(interesting_frames_indexes[i] * (len(x) / len(np_video)))
		                 for i in range(num_interesting_frames)])
		
		plt.plot(list(range(len(x))), x)
		plt.scatter(np.array(list(range(len(x))))[intt], x[intt], color='red', s=40, marker='o')
		plt.title(video_path)
		plt.savefig('summary.png')
		
		if translate:
			translator = Translator()
			caption_translated = [' '.join([self.translator_translate(translator, word)
			                                for word in caption[0].split()])]
			return res[1][0, :, 0].detach().cpu().numpy(), caption_translated
		
		return x, captions, intt
	

def main():
	project_path = '/Users/vaneshik/hack/CP_CODE'
	interest_model_path = os.path.join(project_path, 'models/VTSum/vtsum_tt.pth')
	interester = Interester(interest_model_path)
	print(interester.get_interests_summary(os.path.join(project_path, 'video_with_audio/chetvertak.mp4')))
	
	
if __name__ == '__main__':
	main()