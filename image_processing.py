import os

import cv2
path = 'Ronaldo'

image_files = [f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
images =[]
for img in image_files:
    image_path = os.path.join(path, img)
    img = cv2.imread(image_path)
    images.append(img)

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades  + 'haarcascade_frontalface_default.xml')

for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces= face_detector.detectMultiScale(gray, 1.1, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow('images', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()