import tensorflow as tf
from tensorflow.keras import layers
 
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

X = tf.keras.utils.image_dataset_from_directory(
    'images/lq', label_mode=None, image_size=(100, 100), batch_size=2, shuffle=False)

y = tf.keras.utils.image_dataset_from_directory(
    'images/hq', label_mode=None, image_size=(600, 600), batch_size=2, shuffle=False)

# noinspection PyShadowingNames
ds = tf.data.Dataset.zip((X, y)).map(lambda x, y: (x/255.0, y/255.0))

for x_batch, y_batch in ds.take(1):
    print(f"Low-Res Shape: {x_batch[0].shape}")
    print(f"High-Res Shape: {y_batch[0].shape}")

try:
    model = tf.keras.models.Sequential([
        layers.Input(shape=(100, 100, 3)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Dropout(0.2),
        layers.UpSampling2D(size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D(size=(3, 3)),
        layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    model.summary()
    model.fit(ds, epochs=10)
    model.save('upscaler.h5')
except Exception as e:
    print(e)
    print('Most likely cause is that image resolutions are different.')