from ultralytics import YOLO
import numpy as np
import cv2
import random 
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from keras.callbacks import Callback, ModelCheckpoint
import keras.backend as K
from PIL import Image

from os import listdir
from os.path import isfile, join


def get_cropped_ped_yolo(model, frame, return_shape=(224, 224)):
    class_list = [0, 1]
    results = model.predict(source=frame, classes=class_list)
    
    ped = []
    bounds=[]
    
    # Process results
    for result in results:
        boxes = result.boxes  # Bounding box outputs
        
        for box in boxes:
            bound = box.xyxy[0]
            x_min = int(bound[0].item())
            y_min = int(bound[1].item())
            x_max = int(bound[2].item())
            y_max = int(bound[3].item())
            
            # Crop the image
            cropped_image = frame[y_min:y_max, x_min:x_max].copy()
            
            # Convert BGR to RGB
            cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            
            # Resize the image
            cropped_image_rgb = cv2.resize(cropped_image_rgb, return_shape)
            
            # Append the cropped RGB image to the list
            ped.append(cropped_image_rgb)
            bounds.append([x_min,y_min,x_max,y_max]);
    
    return ped,bounds

    



# Define the path to the SavedModel directory
model_path = r"C:\Projects\ML\Distracted-Pedestrian-Monitoring-System\saved_models\pedestrian_tracker_v1"



model = YOLO("yolov8n.pt")
#results = model(frame)


frame = cv2.imread("example_images/distracted_ped_01.jpg") 


# Get cropped human images from the function
ped ,bounds = get_cropped_ped_yolo(model, frame)


cv2.waitKey(0)

count=0
for i in range(len(ped)):
    detect_color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
    cv2.rectangle(frame, (bounds[i][0], bounds[i][1]), (bounds[i][2], bounds[i][3]), detect_color, 2)


cv2.imshow("Image",frame)
cv2.waitKey(0)


