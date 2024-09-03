import cv2
import numpy as np

img = cv2.imread('Images/input_image-1.jpg')
# if img is None:
#     print('Could not find the image')
#
# kernel1 = np.array([[-1, -1, -1],
#                    [-1, 9, -1],
#                    [-1, -1, -1]])
# identity = cv2.filter2D(img, -1, kernel1)
# cv2.imshow('Original', img)
# cv2.imshow('Identity', identity)
# cv2.waitKey(0)
# # cv2.imwrite('Identity', identity)
# cv2.destroyAllWindows()
#
# kernel2 = np.ones((5, 5), np.float32) / 25
# img2=cv2.filter2D(img, -1, kernel2)
#
# cv2.imshow('Original', img)
# cv2.imshow('Identity', img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#Blur
# img_blur = cv2.blur(img, (5, 5))
# cv2.imshow('Original', img)
# cv2.imshow('Blur', img_blur)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#median blur
# img_blur = cv2.medianBlur(img, ksize=5)
# cv2.imshow('Original', img)
# cv2.imshow('Median Blurred', img_blur)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #sharpening by convolution kernel
# kernel3= np.array([[0, -1, 0],
#                    [-1, 5, -1],
#                    [0, -1, 0]])
#
# sharpen_img = cv2.filter2D(img, -1, kernel3)
# cv2.imshow('Original', img)
# cv2.imshow('Sharpened',  sharpen_img)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#bilateral filtering
bilateral_filter = cv2.bilateralFilter(img, 9, 75, 75)
cv2.imshow('Original', img)
cv2.imshow('Sharpened',  bilateral_filter)

cv2.waitKey(0)
cv2.destroyAllWindows()