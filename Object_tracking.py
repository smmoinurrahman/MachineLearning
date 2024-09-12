import cv2
from tracker import *

#creat tracker obeject
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture('Images/highway.mp4',)

obeject_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=False)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480), interpolation = cv2.INTER_AREA)

    #Extract region of interest
    roi = frame[0:300, 250:500]

    # object detection
    mask = obeject_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detection = []
    for cnt in contours:
        #calculate area and remove unwanted elements
        area = cv2.contourArea(cnt)
        if area > 100:
            #cv2.drawContours(roi,[cnt],-1,(0,255,0),2)
            x,y,w,h = cv2.boundingRect(cnt)


            detection.append([x,y,w,h])

    #object tracking
    box_ids = tracker.update(detection)
    for box in box_ids:
        x,y,w,h, id = box
        cv2.putText(roi, str(id), (x,y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('roi',roi)
    cv2.imshow('frame',frame)
    cv2.imshow('mask', mask)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

