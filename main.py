import gradio as gr
from PIL import Image
import torch
from model import load_model
from utils import preprocess_image, class_names

model = load_model()

def predict(image: Image.Image):
    try:
       
        tensor = preprocess_image(image)
        
        
        with torch.no_grad():
            output = model(tensor.unsqueeze(0)) 
            prediction = torch.argmax(output, dim=1).item()

        return class_names[prediction]
    except Exception as e:
        return {"error": str(e), "message": "Invalid image or preprocessing failed"}

iface = gr.Interface(
    fn=predict, 
    inputs=gr.Image(type="pil"), 
    outputs="text",
    title="Upload an MRI image to predict the class of the medical condition. The model uses CNN to classify the condition ."
)

iface.launch(share=True)
