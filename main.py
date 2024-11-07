import os
import matplotlib
import numpy as np
from PIL import Image
from keras.utils.image_utils import img_to_array
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

data = []
labels = []

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

classes = 42
current_directory = os.getcwd()
folder_path = os.path.join(current_directory, 'Train')
print(folder_path)

# for num in range(0, classes):
#     path = os.path.join('Train', str(num))
#     imagePaths = os.listdir(path)
#     for img in imagePaths:
#       image = Image.open(path + '/'+ img)
#       image = image.resize((30,30))
#       image = img_to_array(image)
#       data.append(image)
#       labels.append(num)