# TODO: check with a large image and see how the boxes translate
# of if they need scaling to the original size VS newtork input size

import cv2

import torch
import numpy as np

from PIL import Image
from torchvision import transforms
from models.yolo_nano import YOLONano

np.set_printoptions(suppress=True)

IMAGE_PATH = "/home/plantroot/temp/opened_10_frame_00009.jpg"
CONFIDENCE_THRESHOLD = 0.95
# image = cv2.imread(IMAGE_PATH)
# gt = [47, 244, 82, 140]


######################
###   LOAD MODEL   ###
######################

model = YOLONano(num_classes=2, image_size=416)

checkpoint = torch.load("/home/plantroot/Code/yolo-nano/checkpoints/epoch_600.pth")
state_dict = checkpoint["state_dict"]

model.load_state_dict(state_dict)
model.eval()

#####################
##   INFERENCE    ###
#####################

image = Image.open(IMAGE_PATH)
transform = transforms.Compose(
    [
        # Resize image to match the input size of the model
        transforms.Resize((416, 416)),
        # Convert image to tensor
        transforms.ToTensor(),
        # TODO: Investigate if the model has normalization during training and see if we need it here
        # Normalize based on ImageNet dataset
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

input_tensor = transform(image)
# Add batch dimension
input_tensor = input_tensor.unsqueeze(0)

# Disable gradient calculation for inference
with torch.no_grad():
    OUTPUT = model(input_tensor)

# Remove batch dimension
output_squeezed = OUTPUT.squeeze(0)

######################
### PROCESS OUTPUT ###
######################

# Filter predictions with a high enough confidence score (e.g., > 0.5)
predictions = output_squeezed[output_squeezed[:, 4] > CONFIDENCE_THRESHOLD]

# Load your image (replace 'image_path' with the actual image file path)
image = cv2.imread(IMAGE_PATH)

# Load original image for correct scaling
# original_height, original_width = image.shape[:2]

# Draw each bounding box
print(len(predictions))

for prediction in predictions:
    prediction = np.array(prediction)
    print(prediction)
    # COCO bbox format for predictions
    xmin, ymin, width, height = prediction[:4]
    xmin, ymin, width, height = int(xmin), int(ymin), int(width), int(height)

    xmax = xmin + width
    ymax = ymin + height

    top_left_corner = (xmin, ymin)
    bottom_right_corner = (xmax, ymax)

    cv2.rectangle(image, top_left_corner, bottom_right_corner, (255, 0, 0), 2)

    # Display the image
    cv2.imshow("Image with Bounding Box", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(image.shape)
