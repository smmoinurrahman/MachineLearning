import cv2

recognizer = cv2.face.LBPHFaceRecognizer.create()
recognizer.read('trainer/face_recognizer.yml')

detector = cv2.CascadeClassifier(cv2.data.haarcascades  + 'haarcascade_frontalface_default.xml')

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0

Names = ['None', 'Moin', 'Sarwar']

cam=cv2.VideoCapture(0)

#min window size to recognize face



while True:
    ret, frame = cam.read()
    if ret == False:
        break
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence < 100:
            id = Names[id]
            confidence = " {0}%".format(round(100 - confidence))
        else:
            id = "Unknown"
            confidence = "{0}%".format(round(100 - confidence))

        cv2.putText(frame, id, (x+5, y-5), font, 1, (255, 255, 255), 2)

        cv2.putText(frame, str(confidence), (x+5, y+h-5), font, 1, (255, 255, 0), 2)
    cv2.imshow('Face Recognizer', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cam.release()
cv2.destroyAllWindows()
