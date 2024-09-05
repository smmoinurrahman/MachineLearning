import cv2
import numpy as np

from matplotlib import pyplot as plt

image = cv2.imread("Images/Mask.jpg")
image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#equslization

eqlz= cv2.equalizeHist(image)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_image = clahe.apply(image)
com = np.hstack((image,eqlz, clahe_image))
cv2.imshow("Image Comparison", com)


cv2.waitKey(0)
#histogram
hist1 = cv2.calcHist([clahe_image],[0],None,[256],[0,256])
hist2 = cv2.calcHist([eqlz],[0],None,[256],[0,256])
hist3 = cv2.calcHist([image],[0],None,[256],[0,256])
plt.figure()
plt.title("Histogram Comparison")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist1)
plt.plot(hist2)
plt.plot(hist3)
plt.legend(["Clahe Histogram","Equalized Histogram","Histogram"])
plt.xlim([0,256])
plt.show()
# #normalized histogram
# hist /= hist.sum()
# plt.figure()
# plt.title("Grayscale Histogram(Normalized)")
# plt.xlabel("Bins")
# plt.ylabel("# of Pixels")
# plt.plot(hist)
# plt.xlim([0,256])
# plt.show()



