import numpy as np
from PIL import Image
import tensorflow as tf
from PIL import ImageFilter

model = tf.keras.models.load_model('saved_model/face_gen_3')
model.summary()
image = Image.open("CroppedYale/yaleB01/yaleB01_P00A+000E+45.pgm") # use an image through it's file path
image = image.crop((0, 12, 168, 180))
image = image.resize((128, 128))
# image.show()
altered_image = image.filter(ImageFilter.CONTOUR) # optional modification


# altered_image.show()

def run(input_image):
    pixel_data = np.asarray(input_image)

    pixel_data = pixel_data / 255 - 0.5

    pixel_data = pixel_data.reshape(-1, 128, 128, 1)
    pred = model.predict(pixel_data)

    pred = pred[0, :, :, 0]
    pred = (pred + 0.5) * 255

    img = Image.fromarray(pred)
    # img.show()
    return img


# altered_image = Image.fromarray(np.load('user_drawn_image.npy'))
# use this if you want to load an image from FaceDraw.py
altered_image.show()
source = altered_image
for _ in range(1):
    source = run(source)
    # because the input and output dimensions are the same, we could keep feeding our output back into the model
    source.show()
