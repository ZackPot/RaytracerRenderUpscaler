import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

def img_extract(target_letter):
    print('LQ')
    root_dir = './images/'
    filenames = []
    images = []

    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            if filename.lower().startswith(target_letter.lower()):
                filenames.append(os.path.join(root, filename))

    for image in filenames:
        img = np.array(Image.open(image))
        img = img.astype('float32') / 255.0
        images.append(img.reshape(-1))

    return images

X = img_extract('l')
y = img_extract('h')

try:
    model = tf.keras.models.Sequential([
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(y[0]), activation='sigmoid'),
    ])

    model.compile(optimizer='adam', loss='mme', metrics=['accuracy'])

    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
    model.fit(X, y, epochs=10)
    model.save('upscaler.h5')
except Exception as e:
    print(e)
    print('Most likely cause is that image resolutions are different.')