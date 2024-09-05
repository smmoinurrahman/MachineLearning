import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread('Images/input_image-1.jpg')
img= cv2.resize(img,(400,300))
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# #translation
# rows, cols = img.shape
# M = np.float32([[1, 0, 100], [0, 1, 50]])
# dst = cv2.warpAffine(img, M, (cols, rows))
# cv2.imshow('Image',dst)
# cv2.waitKey(0)

# #Reflection
# rows, cols = img.shape
# M1 = np.float32([[-1, 0, cols],
#                  [0, 1, 0],
#                  [0,0,1]])
# reflected_image = cv2.warpPerspective(img, M1, (cols, rows))
#
# cv2.imshow('reflected image', reflected_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #Rotation
# img_rotation =cv2.warpAffine(img, cv2.getRotationMatrix2D((cols/2, rows/2), 30,0.6), (cols, rows))
# cv2.imshow('Rotated Image', img_rotation)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#Scaling

# image_resize = cv2.resize(img,(300,200),interpolation=cv2.INTER_AREA)
# cv2.imshow('image',image_resize)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# image_enlarge = cv2.resize(image_resize,(700,500),interpolation=cv2.INTER_CUBIC)
#
# cv2.imshow('image_enlarge',image_enlarge)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #Cropping
# cropped_image = img[50:200,0:300]
# cv2.imshow('Cropped Image',cropped_image)
# cv2.waitKey(0)

#Shearing X_axis
rows, cols = img.shape
M2= np.float32([[1,0.5,0],[0,1,0],[0,0,1]])
sheared_image = cv2.warpPerspective(img,M2, (int(cols*1.5), int(rows*1.5)))
cv2.imshow('Sheared Image',sheared_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Shearing X_axis
rows, cols = img.shape
M2= np.float32([[1,0,0],[0.5,1,0],[0,0,1]])
sheared_image = cv2.warpPerspective(img,M2, (int(cols*1.5), int(rows*1.5)))
cv2.imshow('Sheared Image',sheared_image)
cv2.waitKey(0)
cv2.destroyAllWindows()