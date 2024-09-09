import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2hsv, rgb2gray
from skimage.exposure import histogram, cumulative_distribution
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola

img = imread("Images/cup.png")
gray = rgb2gray(img)

# #segmentation by thresolding and manual input
#
# for i in range(10):
#     binary_img = (gray> i*0.1)*1
#     plt.subplot(5,2,i+1)
#     plt.imshow(binary_img, cmap='gray')
#     plt.title("Thresold: >" +str(round(i*0.1, 1)))
#
# plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=1.5, wspace=0.4)
# plt.show()

#segmentation by thresholding by skimage.filter module

threshold = threshold_otsu(gray)
binary_img = (gray> threshold)*1
plt.subplot(2,2,1)
plt.title("Threshold: >" +str(round(threshold,1)))
plt.imshow(binary_img, cmap='gray')

threshold = threshold_niblack(gray)
binary_img = (gray> threshold)*1
plt.subplot(2,2,2)
plt.title("Niblack Thresholding")
plt.imshow(binary_img, cmap='gray')

threshold = threshold_sauvola(gray)
plt.subplot(2,2,3)
plt.title("Sauvola Thresholding")
plt.imshow(binary_img, cmap='gray')

plt.show()

