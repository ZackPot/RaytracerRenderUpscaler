import os
from PIL import Image
import numpy as np

# Low Quality Image Extraction
target_letter = 'l'
root_dir = './images/'
lq_filenames = []
lq_images = []

for root, dirs, files in os.walk(root_dir):
    for filename in files:
        if filename.lower().startswith(target_letter.lower()):
            lq_filenames.append(os.path.join(root, filename))

for image in lq_filenames:
    img = Image.open(image)
    lq_images.append(np.array(img).reshape(-1))

# High Quality Image Extraction
target_letter = 'h'
root_dir = './images/'
hq_filenames = []
hq_images = []

for root, dirs, files in os.walk(root_dir):
    for filename in files:
        if filename.lower().startswith(target_letter.lower()):
            hq_filenames.append(os.path.join(root, filename))

for image in lq_filenames:
    img = Image.open(image)
    hq_images.append(np.array(img).reshape(-1))

print(hq_images[0])