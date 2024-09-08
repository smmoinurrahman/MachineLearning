import os

import cv2
import numpy as np


path = 'Image_Dataset'

recognizer = cv2.face.LBPHFaceRecognizer.create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades  + 'haarcascade_frontalface_default.xml')


def prepare_training_data(path):
    image_files = [f for f in os.listdir(path)]
    images = []
    ids = []
    for img in image_files:
        image_path = os.path.join(path, img)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(image_path)[-1].split(".")[1])
        images.append(img)
        ids.append(id)

    return images,ids



faces, ids = prepare_training_data(path)

recognizer.train(faces, np.array(ids))
recognizer.save('trainer/face_recognizer.yml')
