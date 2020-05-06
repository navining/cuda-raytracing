import numpy as np
from PIL import Image
import cv2

ref = np.array(cv2.imread('build/result_25k.png'))
img = np.array(cv2.imread('build/image-cuda.ppm'))
mse = np.mean((img - ref)**2)

print("mean squared error: ", mse)