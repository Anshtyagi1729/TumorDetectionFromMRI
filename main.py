import gradio as gr
from PIL import Image
import torch
from model import load_model
from utils import preprocess_image, class_names

# Load the model
model = load_model()

# Function to process the uploaded image and return prediction
def predict(image: Image.Image):
    try:
        # Preprocess the image
        tensor = preprocess_image(image)
        
        # Run the model
        with torch.no_grad():
            output = model(tensor.unsqueeze(0))  # Add batch dimension
            prediction = torch.argmax(output, dim=1).item()

        return class_names[prediction]
    except Exception as e:
        return {"error": str(e), "message": "Invalid image or preprocessing failed"}

# Create Gradio interface
iface = gr.Interface(
    fn=predict, 
    inputs=gr.Image(type="pil"), 
    outputs="text",
    title="Upload an MRI image to predict the class of the medical condition. The model uses CNN to classify the condition ."
)

iface.launch(share=True)
