import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2hsv
from skimage.exposure import histogram, cumulative_distribution
from skimage.filters import threshold_otsu

img = imread("Images/cup.png")
hsv = rgb2hsv(img)

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.colorbar(plt.imshow(hsv), fraction=0.046, pad=0.04)
plt.show()
