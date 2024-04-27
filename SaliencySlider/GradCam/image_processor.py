from matplotlib import cm
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import numpy as np
import io, base64
import requests
from PIL import Image, ImageFilter
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils import normalize

global_model = VGG19(weights='imagenet')

def process_image(image_url, intensity):

    response = requests.get(image_url)  # fetches image from URL
    img = Image.open(io.BytesIO(response.content))  # opens image from a bytes buffer

    img = img.resize((224, 224))
    img_array = img_to_array(img)
    print(img_array.shape)

    # Check if image has an alpha channel (RGBA) and convert it to RGB
    if img.mode == 'RGBA':
        img = img.convert('RGB')    

    img_array = np.expand_dims(img_array, axis=0)  # Make 'batch' of 1
    img_array = preprocess_input(img_array)

    # Predictions
    predictions = global_model.predict(img_array) 
    top_pred = np.argmax(predictions[0]) # can change to retrieve more predictions like 1-5
    
    # GradCam stuff -- might be beneficial to take it out of this function somehow
    gradcam = Gradcam(global_model, model_modifier=None, clone=False)
    # cam = gradcam(top_pred, img_array, penultimate_layer=-1)  # Use appropriate layer # -- error here 
    # Error processing image: Score object must be callable! [823]
    # Internal Server Error: /GradCam/update_image/

    # heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)  # Get heatmap

    # Process the image based on intensity or other parameters -- to be removed
    img = img.filter(ImageFilter.GaussianBlur(radius=intensity))

    # Convert the processed image to a data URL
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")  # Save image to the buffer in JPEG format
    image_data = buffer.getvalue()
    image_data_url = "data:image/jpeg;base64," + base64.b64encode(image_data).decode('utf-8')
    return image_data_url



# def process_image(image_url, intensity):
#     # Load model
#     model = VGG19(weights='imagenet')
#     model.summary()  # For debugging, to see model layers

#     # Fetch image from URL and preprocess
#     response = requests.get(image_url)
#     img = Image.open(io.BytesIO(response.content))
#     img = img.resize((224, 224))  # Resize for the model input
#     img_array = img_to_array(img)  # Convert to array
#     img_array = np.expand_dims(img_array, axis=0)  # Make 'batch' of 1
#     img_array = preprocess_input(img_array)  # Preprocess the input

#     # Predictions
#     predictions = model.predict(img_array)
#     top_pred = np.argmax(predictions[0])

#     # GradCAM
#     gradcam = Gradcam(model, model_modifier=None, clone=False)
#     # Generate heatmap with GradCAM
#     cam = gradcam(top_pred, img_array, penultimate_layer=-1)  # Use appropriate layer
#     heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)  # Get heatmap

#     # Merge heatmap with original image
#     heatmap = Image.fromarray(heatmap)
#     merged_img = Image.blend(img.convert("RGBA"), heatmap.convert("RGBA"), alpha=0.5)
    
#     # Convert to data URL
#     buffer = io.BytesIO()
#     merged_img.save(buffer, format="JPEG")
#     image_data = buffer.getvalue()
#     image_data_url = "data:image/jpeg;base64," + base64.b64encode(image_data).decode('utf-8')

#     return image_data_url