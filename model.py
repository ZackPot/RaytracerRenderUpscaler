import tensorflow as tf
from tensorflow.keras import layers

X = tf.keras.utils.image_dataset_from_directory(
    'images/lq', label_mode=None, image_size=(100, 100), batch_size=4, shuffle=False)

y = tf.keras.utils.image_dataset_from_directory(
    'images/hq', label_mode=None, image_size=(2000, 2000), batch_size=4, shuffle=False)

ds = tf.data.Dataset.zip((X, y)).map(lambda x, y: (x/255.0, y/255.0))

for x_batch, y_batch in ds.take(1):
    print(f"Low-Res Shape: {x_batch[0].shape}")
    print(f"High-Res Shape: {y_batch[0].shape}")

try:
    model = tf.keras.models.Sequential([
        layers.Input(shape=(100, 100, 3)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D(size=(5, 5)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D(size=(4, 4)),
        layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
    model.fit(ds, epochs=10)
    model.save('upscaler.h5')
except Exception as e:
    print(e)
    print('Most likely cause is that image resolutions are different.')