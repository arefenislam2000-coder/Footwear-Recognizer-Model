import gradio as gr
import torch
from fastai.vision.all import *
from pathlib import Path
import numpy as np

# Load the FastAI model
def load_model():
    try:
        # Load the exported model (recommended)
        learn = load_learner('footwear_model_v4.pkl')
        return learn
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Initialize model
model = load_model()

def predict_footwear(image):
    """
    Predict footwear category from uploaded image
    """
    if model is None:
        return {"Error": "Model not loaded properly"}
    
    try:
        # Convert to PIL Image if needed
        if hasattr(image, 'convert'):
            image = image.convert('RGB')
        
        # Make prediction using FastAI
        pred_class, pred_idx, pred_probs = model.predict(image)
        
        # Get all class labels from the model
        labels = model.dls.vocab
        
        # Create results dictionary with probabilities
        results = {}
        for i, label in enumerate(labels):
            probability = float(pred_probs[i])
            results[label] = probability
        
        return results
        
    except Exception as e:
        return {"Error": f"Prediction failed: {str(e)}"}

# Create Gradio interface
def create_interface():
    interface = gr.Interface(
        fn=predict_footwear,
        inputs=[
            gr.Image(
                type="pil", 
                label="Upload Footwear Image",
                height=400
            )
        ],
        outputs=[
            gr.Label(
                num_top_classes=5,
                label="Prediction Results"
            )
        ],
        title="🦶 Footwear Recognizer",
        description="""
        **Upload an image of footwear to classify the type!**
        
        This model was trained using FastAI with ResNet34 architecture to recognize different types of footwear.
        Simply upload a clear image of shoes, boots, sandals, or other footwear for instant classification.
        """,
        article="""
        ### Model Information:
        - **Architecture**: ResNet34 (transfer learning)
        - **Framework**: FastAI
        - **Input Size**: 224x224 pixels
        - **Data Augmentation**: Rotation, zoom, lighting adjustments
        
        ### Tips for best results:
        - Use clear, well-lit images
        - Ensure the footwear is the main subject
        - Avoid heavily cropped or blurry images
        """,
        examples=None,
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            font-family: 'Arial', sans-serif;
        }
        .title {
            text-align: center;
            color: #2d5aa0;
        }
        """
    )
    
    return interface

# Launch the app
if __name__ == "__main__":
    # Check if model loaded successfully
    if model is None:
        print("❌ Failed to load model. Please check if 'footwear_model_v4.pkl' exists.")
    else:
        print("✅ Model loaded successfully!")
        print(f"📊 Model can classify: {model.dls.vocab}")
    
    # Create and launch interface
    iface = create_interface()
    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )