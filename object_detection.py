import cv2
import numpy as np
import os
from collections import Counter


class Detector:
	def __init__(self, weights_path: str, config_path: str, classes_path: str) -> None:
		self.net = cv2.dnn.readNet(weights_path, config_path)
		self.layer_names = self.net.getLayerNames()
		self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
		
		with open(classes_path) as f:
			self.classes = [line.strip() for line in f.readlines()]
			
	def __call__(self, video_path: str) -> list[tuple[str, int]]:
		cap = cv2.VideoCapture(video_path)
		detected_objects = Counter()
		frame_count = 0
		while True:
			ret, frame = cap.read()
			if not ret:
				break
			if frame is not None and frame.size > 0:
				frame_count += 1
				if frame_count % 10 == 0:
					blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
					self.net.setInput(blob)
					outputs = self.net.forward(self.output_layers)
					for output in outputs:
						for detection in output:
							scores = detection[5:]
							class_id = np.argmax(scores)
							confidence = scores[class_id]
							if confidence > 0.4:
								detected_objects[self.classes[class_id]] += 1
								
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		
		return detected_objects.most_common()
	
	
def main():
	project_path = '/Users/vaneshik/hack/CP_CODE'
	weights_path = os.path.join(project_path, 'models/YOLOv3/yolov3.weights')
	config_path = os.path.join(project_path, 'models/YOLOv3/yolov3.cfg')
	classes_path = os.path.join(project_path, 'models/YOLOv3/coco.names')
	detector = Detector(weights_path, config_path, classes_path)
	
	video_path = os.path.join(project_path, 'video_with_audio/arsen.mp4')
	print(detector(video_path))
	
	
if __name__ == '__main__':
	main()