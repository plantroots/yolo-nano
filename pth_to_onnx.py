import torch
import torch.onnx
import onnx

from models.yolo_nano import YOLONano


model = YOLONano(2, 416)
checkpoint = torch.load("/home/plantroot/Code/yolo-nano/checkpoints/epoch_600.pth")

# If the model's state_dict is under the 'state_dict' key
model.load_state_dict(checkpoint["state_dict"])
# Set the model to evaluation mode
model.eval()


# This should match the input size of your model
dummy_input = torch.randn(1, 3, 416, 416)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    verbose=True,
    input_names=["input"],
    output_names=["output"],
)
