import torch
import torch.nn as nn
import numpy as np
import cv2

from torchvision import models, transforms
from PIL import Image
from models.yolo_nano import YOLONano

IMAGE_PATH = "/home/plantroot/temp/partially_1_frame_00000.jpg"
CONFIDENCE_THRESHOLD = 0.2

######################
###   LOAD MODEL   ###
######################

model = YOLONano(num_classes=2, image_size=416)

checkpoint = torch.load("/home/plantroot/Code/yolo-nano/checkpoints/epoch_600.pth")
state_dict = checkpoint["state_dict"]

model.load_state_dict(state_dict)
model.eval()

######################
###   INFERENCE    ###
######################

image = Image.open(IMAGE_PATH)
transform = transforms.Compose(
    [
        # Resize image to match the input size of the model
        transforms.Resize((416, 416)),
        # Convert image to tensor
        transforms.ToTensor(),
        # Normalize based on ImageNet dataset
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

# Draw each bounding box
for prediction in predictions:
    # Extract values from the tensor
    x_center, y_center, width, height = prediction[0:4]

    # Calculate top-left and bottom-right corners
    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)

    # Draw the bounding box on the image
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Display the image
    cv2.imshow("Image with Bounding Box", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
