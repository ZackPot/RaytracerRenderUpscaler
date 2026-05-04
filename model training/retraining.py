import tensorflow as tf
from datasets import load_dataset
from PIL import ImageOps
import numpy as np

X = tf.keras.utils.image_dataset_from_directory(
    'images/lq', label_mode=None, image_size=(100, 100), batch_size=2, shuffle=False)

y = tf.keras.utils.image_dataset_from_directory(
    'images/hq', label_mode=None, image_size=(600, 600), batch_size=2, shuffle=False)

X = np.concatenate(list(X.as_numpy_iterator()), axis=0)
y = np.concatenate(list(y.as_numpy_iterator()), axis=0)

X = np.array(X)
y = np.array(y)

ds = load_dataset("eugenesiow/Div2k", "bicubic_x2")
model = tf.keras.models.load_model('upscaler.h5')

train = ds["train"]
lq = np.zeros((len(train), 100, 100, 3))
hq = np.zeros((len(train), 600, 600, 3))

for imgs in enumerate(train):
    lr_img = imgs[1]["lr"].convert("RGB")
    hr_img = imgs[1]["hr"].convert("RGB")

    lr_square = ImageOps.fit(lr_img, (100, 100))
    hr_square = ImageOps.fit(hr_img, (600, 600))

    lq[imgs[0]] = np.array(lr_square)
    hq[imgs[0]] = np.array(hr_square)

X = np.concatenate((X, lq), axis=0)
y = np.concatenate((y, hq), axis=0)

X = X.astype('float32') / 255.0
y = y.astype('float32') / 255.0

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X, y = X[indices], y[indices]

for layer in model.layers[:4]:
    layer.trainable = False

model.compile(tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse', metrics=['mae'])
model.fit(X, y, epochs=3)