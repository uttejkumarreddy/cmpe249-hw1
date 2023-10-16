import sys, getopt, os, datetime
import torch

def main(argv):
	try:
		model = argv[0]
		image_path = argv[1]
	except:
		print('Expected two arguments: model (yolo-pretrained/yolo-custom) and image_path. Got {}'.format(len(argv)))
		sys.exit(2)

	path = ''
	if 'yolo-pretrained' in model:
		path = 'models/pretrained-yolov5s.pt'
	elif 'yolo-custom' in model:
		path = 'models/custom-trained-yolov5.pt'
	else:
		print('Invalid model name. Input either yolo-pretrained or yolo-custom')
		sys.exit(2)

	# Check if file exists at image_path and that it is a image
	if not os.path.isfile(image_path) or not image_path.endswith(('.jpg', '.jpeg', '.png')):
		print('Invalid image path. Input a valid image path')
		sys.exit(2)

	start_time = datetime.datetime.now()

	# Load model
	model = torch.hub.load('D:\\present\\cmpe249-hw1\\training-yolov5\\yolov5', 'custom', path=path, source='local')

	# Inference
	results = model(image_path)

	# Show the boundary boxes on the image and save it
	results.show()
	results.save()

	end_time = datetime.datetime.now()
	print('Output Latency: {}'.format(end_time - start_time))

if __name__ == "__main__":
	main(sys.argv[1:])