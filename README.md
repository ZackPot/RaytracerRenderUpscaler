# AI Render Upscaler

This project takes 3D renders and upscales them to make them higher quality. It uses tensorflow and is aimed at peope who have slow computers and want to improve their render quality. The code is all in python and uses numpy, pillow and tensorflow to take 3D renders - images - made by a Raytracer and train a neural network to upscale them.


## Usage

To train the model, first run renderer.py then sort out the low quality and high quality images into hq/data and lq/data.
After that, run model.py on the Suzanne dataset, then run the retraining.py file to train the model weights for a wide variety of subjects.
