import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams


img_path = os.path.join("/home/nithin/Downloads/cat.jpg")
img = image.load_img(img_path, target_size= (224, 224))
plt.imshow(img)
plt.show()

model = tf.keras.applications.resnet50.ResNet50()

def Predict(img_path):
    img = image.load_img(img_path)
    model= tf.keras.applications.resnet50.ResNet50()
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis = 0)
    img_preprocessed = preprocess_input(img_batch)
    prediction = model.predict(img_preprocessed)
    print(decode_predictions(prediction, top= 3)[0])

#Predict(img_path)

def download_sample_image(filename):
    import requests
    url = f"https://raw.githubusercontent.com/PracticalDL/Practical-Deep-Learning-Book/master/sample-images/{filename}"
    open(filename, 'wb').write(requests.get(url).content)


