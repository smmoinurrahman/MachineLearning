import cv2
import matplotlib.pyplot as plt

# importing classifier
face_classifier= cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

# # face in image
# img= cv2.imread('Images/ronaldo.png')
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# face = face_classifier.detectMultiScale(gray_img, 1.1, 5, minSize=(100,100))
#
# # drawing bounding box
# for (x,y,w,h) in face:
#     cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4)
#
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# plt.figure(figsize=(20,10))
# plt.imshow(img_rgb)
# plt.show()

# realtime face detection

video_capture = cv2.VideoCapture(0)

def detect_bounding_box(vid):
    gray = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces= face_classifier.detectMultiScale(gray, 1.1, 5, minSize=(40,40))
    for (x,y,w,h) in faces:
        cv2.rectangle(vid,(x,y),(x+w,y+h),(0,255,0),4)

    return faces

while True:
    res , frame = video_capture.read()
    if res is False:
        break
    faces = detect_bounding_box(frame)

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

