from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
# nltk.download('punkt')

access_token = "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"

class Tagger:
	def __init__(self, model_path: str, model_name: str="fabiochiu/t5-base-tag-generation") -> None:
		self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
		self.model = AutoModelForSeq2SeqLM.from_pretrained(
			model_path,
			use_safetensors=True,
			# local_files_only=True,
			token=access_token
		)

	def __call__(self, text: str) -> list[str]:
		prompt =  "there are audio description of video and visual description of video: \n ```" + text + "\n```"
		inputs = self.tokenizer([text], max_length=512, truncation=True, return_tensors="pt")
		output = self.model.generate(**inputs, num_beams=8, do_sample=True, min_length=10,
		                        max_length=64)
		decoded_output = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
		tags = list(set(decoded_output.strip().split(", ")))
		return tags
		

if __name__ == '__main__':
	tagger = Tagger(
		"fabiochiu/t5-base-tag-generation",
		"/Users/vaneshik/hack/CP_CODE/models/T5_TAGGING_folder"
	)
	message = "Who are you waiting for? My friend? Listen, we know you're here to see an 11 year old girl.  She said she was 12!"
	tags = tagger(message)
	print(*tags)