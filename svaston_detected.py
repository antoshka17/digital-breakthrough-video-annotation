import numpy as np
import cv2
import os
from ultralytics import YOLO
from pathlib import Path
from PIL import Image


class ForbiddenDetector:
	def __init__(self, path_to_model: str) -> None:
		self.yolo = YOLO(path_to_model)
		self.dick = {
			0: 'anne_frank', 1: 'antifa', 2: 'atomwaffen', 3: 'bh_emblem', 4: 'black_sun',
			5: 'british_union_of_fascist', 6: 'broken_sun_cross', 7: 'celtic_cross', 8: 'combat_18',
			9: 'doppelsiegrune', 10: 'elhaz', 11: 'fasces', 12: 'golden_dawn', 13: 'hammerskins',
			14: 'happy_merchant', 15: 'hitler', 16: 'hitler-salute', 17: 'ib_lambda', 18: 'judenstern',
			19: 'kolovrat', 20: 'middle', 21: 'national_rebirth_poland', 22: 'othala',
			23: 'pepe', 24: 'sa_emblem', 25: 'siegrune', 26: 'ss_skull', 27: 'star_of_david',
			28: 'swastika', 29: 'tiwaz', 30: 'triskelion', 31: 'valknut', 32: 'volksfront',
			33: 'vril', 34: 'wolfsangel'
		}
		self.images_path = 'temp_forbidden_detecting'
	
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
	
	def save_images_from_video(self, frames):
		if not Path(self.images_path).exists():
			os.mkdir(self.images_path)

		freq = 6
		for i in range(0, len(frames), freq):
			frame = frames[i, ...]
			cv2.imwrite(os.path.join(self.images_path, str(i) + '.png'), frame)
	
	def make_submission(self) -> str:
		fns = os.listdir(self.images_path)
		symbols_detected = []
		for fn in fns:
			image = Image.open(os.path.join(self.images_path, fn))
			result = self.yolo(image, conf=0.4, show=False, verbose=False)
			if len(result[0].boxes.cls.detach().cpu().numpy()):
				symbols_detected.append(self.dick[result[0].boxes.cls.detach().cpu().numpy()[0]])
		os.system(f'rm -r {self.images_path}')
		
		symbols_detected = [sym for sym in set(symbols_detected)
		                    if sym == 'swastika' or sym == 'antifa']
		if len(symbols_detected) == 0:
			return 'No forbidden symbols detected'
		else:
			return 'Following forbidden symbols were detected: ' + str(','.join(symbols_detected))
		
	def predict_bad_symbols(self, video_paths: str) -> list[tuple[str, str]]:
		verdicts = []
		for video_path in os.listdir(video_paths):
			fullpath = os.path.join(video_paths, video_path)
			video = self.read_video(fullpath)
			self.save_images_from_video(video)
			verdict = self.make_submission()
			verdicts.append((fullpath, verdict))
			
		return verdicts
	
	
def main():
	project_path = '/Users/vaneshik/hack/CP_CODE'
	model_path = os.path.join(project_path, 'models/YOLOv3_FORBIDDEN/best.pt')
	forbidden_detector = ForbiddenDetector(model_path)
	
	video_paths = os.path.join(project_path, 'test')
	x = forbidden_detector.predict_bad_symbols(video_paths)
	print(*x, sep="\n")

if __name__ == '__main__':
	main()