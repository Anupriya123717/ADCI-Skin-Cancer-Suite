import gradio as gr
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def build_adci_suite():

    base = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base.output)
    preds = Dense(7, activation='softmax')(x)
    m = Model(inputs=base.input, outputs=preds)
    

    m.load_weights('ADCI_Weights.weights.h5')
    return m

research_model = build_adci_suite()

def analyze(img):
    img_res = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
    batch = np.expand_dims(img_res, axis=0)
    preds = research_model.predict(batch)
    
    labels = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis', 
              'Dermatofibroma', 'Melanoma', 'Melanocytic nevi', 'Vascular lesions']
    
    idx = np.argmax(preds)
    conf = float(preds[0][idx] * 100)
    return f"ADCI Diagnostic: {labels[idx]}\nConfidence: {conf:.2f}%"

demo = gr.Interface(
    fn=analyze, 
    inputs=gr.Image(), 
    outputs="text",
    title="ADCI Skin Cancer Diagnostic Suite",
    description="Developed by Anupriya Sharma"
)

if __name__ == "__main__":
    demo.launch()
