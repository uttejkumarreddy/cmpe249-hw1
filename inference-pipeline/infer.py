from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image

import sys, getopt, os, datetime
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class InferencePipeline:
	def __init__(self, model, image):
		self.model = model
		self.image = image

	def prechecks(self):
		self.model_path = None
		if 'yolo-pretrained' in self.model:
			self.model_path = 'models/pretrained-yolov5s.pt'
			self.model_class = 'yolo'
		elif 'yolo-custom' in self.model:
			self.model_path = 'models/custom-trained-yolov5.pt'
			self.model_class = 'yolo'
		elif 'rcnn' in self.model:
			self.model_class = 'rcnn'
		else:
			print('Invalid model name. Should be one of yolo-pretrained, yolo-custom or rcnn')
			sys.exit(2)

		# Check if file exists at image_path and that it is a image
		if not os.path.isfile(self.image) or not self.image.endswith(('.jpg', '.jpeg', '.png')):
			print('Invalid image path.')
			sys.exit(2)

	def load_model(self):
		if self.model_path:
			self.model_to_predict = torch.hub.load('D:\\present\\cmpe249-hw1\\training-yolov5\\yolov5', 'custom', path=self.model_path, source='local')
		else:
			self.model_to_predict = fasterrcnn_resnet50_fpn(pretrained=True)
			self.model_to_predict.eval()

	def transform_image(self):
		if self.model_class == 'yolo':
			self.image_to_predict = self.image
		elif self.model_class == 'rcnn':
			imageRGB = Image.open(self.image).convert('RGB')
			self.image_to_predict = F.to_tensor(imageRGB).unsqueeze(0)

	def predict(self):
		self.results = self.model_to_predict(self.image_to_predict)

	def run(self):
		# Perform prechecks to see if the inputs are valid
		self.prechecks()

		# Load model
		self.load_model()

		# Transform image
		self.transform_image()

		# Perform inference
		self.predict()

		# Show results
		self.visualize()

	def visualize(self):
		if self.model_class == 'yolo':
			self.results.show()
			self.results.save()
			
		# TODO: Visualize results for RCNN
		# image = Image.open(self.image)
		# plt.imshow(self.image)

		# ax = plt.gca()
		# x, y, width, height = self.results['bbox']
		# rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
		# ax.add_patch(rect)
		# plt.text(x, y, f"{coco_val.cats[result['category_id']]['name']}: {result['score']:.2f}", color='white', bbox=dict(facecolor='r', edgecolor='r', boxstyle='round'))

		# plt.show()

def main(argv):
	# try:
		model = argv[0]
		image = argv[1]

		start_time = datetime.datetime.now()

		# Run Inference Pipeline
		infPipeline = InferencePipeline(model, image)
		infPipeline.run()

		end_time = datetime.datetime.now()

		print('Output Latency: {}'.format(end_time - start_time))
	# except:
	# 	print('Expected two arguments: model (yolo-pretrained/yolo-custom) and image_path. Got {}'.format(len(argv)))
	# 	sys.exit(2)

if __name__ == "__main__":
	main(sys.argv[1:])