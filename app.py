import gradio as gr
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from src.models.model import ShapeClassifier  # Import your model class
from torchvision import transforms
import os
from src.data.transform import data_transform


def classify_drawing(drawing_image):
    # return null if no drawing is provided
    if drawing_image is None:
        return None

    # Load the trained model
    num_classes = 3  # Set the number of classes
    # Initialize your model class
    model = ShapeClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()  # Set the model to evaluation mode

    # Convert the drawing to a grayscale image
    drawing = np.array(drawing_image)

    drawing_tensor = data_transform(Image.fromarray(drawing))

    # save all the drawing to a folder draw with index
    # Image.fromarray(drawing).save(f'draw/{len(os.listdir("draw"))}.png')

    # Perform inference
    with torch.no_grad():
        output = model(drawing_tensor)

    shape_classes = ["Circle", "Square", "Triangle"]
    predicted_class = torch.argmax(output, dim=1).item()
    predicted_label = shape_classes[predicted_class]

    return predicted_label


iface = gr.Interface(
    fn=classify_drawing,
    inputs=gr.Image(type="pil"),  # Use Sketchpad as input
    outputs="text",
    live=True,
    capture_session=True,
)
iface.launch(server_port=7860)
