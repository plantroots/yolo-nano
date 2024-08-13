import torch
import torch.nn as nn
import numpy as np
import cv2

from torchvision import models, transforms
from PIL import Image
from models.yolo_nano import YOLONano

IMAGE_PATH = "/home/plantroot/temp/partially_1_frame_00000.jpg"
CONFIDENCE_THRESHOLD = 0.5

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

# Convert the image to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Draw each bounding box
for prediction in predictions:
    cx, cy, w, h = prediction[0:4]

    # Convert center (cx, cy) to top-left corner (x, y)
    x = int((cx - w / 2) * image.shape[1])
    y = int((cy - h / 2) * image.shape[0])
    width = int(w * image.shape[1])
    height = int(h * image.shape[0])

    # Draw the rectangle (bounding box)
    cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), 2)

    # Optionally: Add the confidence score on top of the box
    score = prediction[4]
    cv2.putText(
        image,
        f"{score:.2f}",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        2,
    )

    # Convert back to BGR to save or display in OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Display the image with bounding boxes
    cv2.imshow("Object Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the image if needed
    cv2.imwrite("output_image.jpg", image)
