from ultralytics import YOLO
import cv2
import cvzone
import math
import time

# cap = cv2.VideoCapture(0)  # For Webcam
cap = cv2.VideoCapture("../Videos/ppe-3-1.mp4")  # For Video
# cap.set(3, 640)
# cap.set(4, 720)


model = YOLO("ppe.pt")

classNames = ['Person', 'goggles', 'helmet', 'no-goggles',
              'no-helmet', 'no-vest', 'vest']

myColor = (0, 0, 255)
while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            # cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            print(currentClass)
            if conf > 0.5:
                if currentClass == 'no-helmet' or currentClass == 'no-vest' or currentClass == "no-goggles":
                    myColor = (0, 0, 255)
                elif currentClass == 'helmet' or currentClass == 'vest' or currentClass == "goggles":
                    myColor = (0, 255, 0)
                else:
                    myColor = (255, 0, 0)

                cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                   colorT=(255, 255, 255), colorR=myColor, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)