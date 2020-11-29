import os
import random

import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

DATA_DIR = "/home/nithin/scikit_learn_data/DL_Data"
CATEGORIES = ["Dog", "Cat"]

training_data = []


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATA_DIR, category)  # path to cats or dogs directory
        class_num = CATEGORIES.index(category)

        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (50, 50))

            try:
                training_data.append([new_array, class_num])

            except Exception as e:
                pass


create_training_data()

random.shufflle(training_data)
