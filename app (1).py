import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("/content/ISL_MobileNetV2_97_5.h5")

# Class labels (A–X)
class_names = [
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X'
]

IMG_SIZE = 128  # must match training image size

def predict_sign(image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)
    index = np.argmax(predictions)

    return class_names[index]

app = gr.Interface(
    fn=predict_sign,
    inputs=gr.Image(type="pil", label="Upload Hand Sign Image"),
    outputs=gr.Text(label="Predicted Sign"),
    title="Indian Sign Language Detection",
    description="Upload an image of a hand sign (A–X) to get prediction."
)

if __name__ == "__main__":
    app.launch(share=True)
