import numpy as np
from PIL import Image
import os
import random

data = []
# This part will have to be adjusted accordingly based on the file location of the dataset
# NOTE: This is not my dataset; dataset can be downloaded at http://vision.ucsd.edu/content/yale-face-database
for file in os.listdir("CroppedYale"):

    for dir in os.listdir("CroppedYale\\" + file):
        if not dir[-4:] == ".pgm" or "Ambient" in dir: # ensures we are reading proper gray-scaled photos
            pass
        else:
            image = Image.open("CroppedYale\\" + file + "\\" + dir)
            image = image.crop((0, 12, 168, 180))
            image = image.resize((128, 128))
            # change image size to a cube so it's easier to work with
            pixel_data = np.asarray(image)
            pixel_data = pixel_data / 255 - 0.5
            pixel_data = pixel_data.reshape(128, 128, 1)
            data.append(pixel_data)


random.shuffle(data)
# shuffle data to ensure model sees a variety of data
data = np.array(data)
print(data.shape)
