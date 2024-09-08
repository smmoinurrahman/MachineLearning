import cv2
import matplotlib.pyplot as plt
config_file = 'Obeject_det_file/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model_file = 'Obeject_det_file/frozen_inference_graph.pb'
model = cv2.dnn.DetectionModel(frozen_model_file, config_file)

classlabels = []
file_name = 'Obeject_det_file/labels.txt'
with open(file_name, 'rt') as fpt:
    classlabels = fpt.read().rstrip('\n').split('\n')

model.setInputSize(600, 600)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)


# img = cv2.imread('Images/input_image-1.jpg')

cam = cv2.VideoCapture(0)
while True:
    ret, img = cam.read()
    img = cv2.resize(img, (600, 600))
# plt.imshow(img)
# plt.show()

    ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.50)

    print(ClassIndex)

    font = cv2.FONT_HERSHEY_PLAIN
    fontScale = 3
    if (len(ClassIndex) != 0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if (ClassInd <=80):
                cv2.rectangle(img, boxes, (0, 255, 0), 2)
                cv2.putText(img, classlabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale, (0, 255, 0), 2)

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()

cam.release()
cv2.destroyAllWindows()