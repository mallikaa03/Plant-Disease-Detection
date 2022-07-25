from matplotlib.pyplot import get
import numpy as np
from fastapi import FastAPI, UploadFile, File, Request
from requests import request
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
app= FastAPI()

templates = Jinja2Templates(directory="templates")

MODEL= tf.keras.models.load_model("../models/1")
CLASS_NAMES= ["Early Blight", "Late Blight", "Healthy"]

@app.get("/helloo")
async def helloo():
    return "hello"
def read_file_as_image(data)-> np.ndarray:
    image= np.array(Image.open(BytesIO(data)))
    return image

@app.route('/', methods=['GET', 'POST'])
def index(request: Request): 
    return templates.TemplateResponse('index.html', context={'request':request})

@app.post('/result')
async def predict(file: UploadFile= File(...),
):
    
    image= read_file_as_image(await file.read())
    img_batch= np.expand_dims(image, 0)
    predictions= MODEL.predict(img_batch)
    predicted_class= CLASS_NAMES[np.argmax(predictions[0])]
    confidence= np.max(predictions[0])
    result= {'class': predicted_class,
        'confidence': float(confidence)
    }
    return result



if __name__== "__main__":
    uvicorn.run(app, host='localhost', port=8080)