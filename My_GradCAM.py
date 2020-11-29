import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import image
#from keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions

#from keras.applications.resnet50 import preprocess_input, decode_predictions
import cv2


DIM = 299

model = InceptionV3(weights='imagenet')
model.summary()
photo = './cat.jpg'


def GradCam(original_image, intensity=0.5, resolution=250):
    img = image.load_img(original_image, target_size=(DIM, DIM))

    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    X = preprocess_input(X)

    predictions = model.predict(X)
    print(decode_predictions(predictions)[0][0][1])

    with tf.GradientTape() as tape:
        last_conv_layer = model.get_layer('conv2d_93')
        iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(X)
        class_out = model_out[:, np.argmax(model_out[0])]
        grads = tape.gradient(class_out, last_conv_layer)
        pooled_grads = keras.backend.mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = heatmap.reshape(8, 8)

    img = cv2.imread(original_image)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    img = heatmap * intensity + img

    cv2.imshow('original image', cv2.resize(cv2.imread(original_image), (resolution, resolution)))
    cv2.imshow('image with heatmap', cv2.resize(img, (resolution, resolution)))


GradCam(photo)
