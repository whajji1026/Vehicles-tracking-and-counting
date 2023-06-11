from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("../videos/carnb.mp4")


model = YOLO("../yolo-weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike",
              "aeroplane", "bus", "train", "truck",
              "boat", "traffic light", "fire hydrant",
              "stop sign", "parking meter", "bench",
              "bird", "cat", "dog", "horse", "sheep",
              "cow", "elephant", "bear", "zebra",
              "giraffe", "backpack", "umbrella",
              "handbag" ,"tie", "suitcase", "frisbee",
              "skis", "snowboard", "sports ball",
              "kite", "baseball bat", "baseball glove",
              "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass",
              "cup", "Fork", "knife", "spoon", "bowl",
              "banana", "apple", "sandwich", "orange",
              "broccoli", "carrot", "hot dog", "pizza",
              "donut", "cake", "chair", "sofa",
              "pottedplant", "bed", "diningtable",
              "toilet", "tvmonitor", "laptop", "mouse",
              "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink",
              "refrigerator", "book", "clock", "vase",
              "scissors", "teddy bear", "hair drier",
              "toothbrush"]

mask = cv2.imread("mask.png")

#tracker
tracker = Sort(max_age=1, min_hits=3,iou_threshold=0.3)
totalcount=[]
limits=[0,370,650,370]

while True:
    success, img = cap.read()
    imgregion = cv2.bitwise_and(img,mask)
    results = model(imgregion, stream=True)

    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:

            #bounding box
            #for cv2
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            #for opencv
            w, h = x2-x1,y2-y1
            #confidence
            conf = math.ceil((box.conf[0]*100))/100

            #class name
            cls = int(box.cls[0])
            currentclass = classNames[cls]

            if currentclass == "car" or currentclass == "truck"\
                    or currentclass == "bus" or currentclass == "motorbike" and conf > 0.2 :
                #cvzone.putTextRect(img,f'{currentclass} {conf}',(max(0,x1),max(35,y1)) ,
                 #             scale=1,thickness=1,offset=1)
                #cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2)
                currentarray=np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections, currentarray))






    resultstracker = tracker.update(detections)
    cv2.line(img,(limits[0],limits[1]),
             (limits[2], limits[3]),(0,0,255),5)

    for result in resultstracker:
        x1, y1, x2, y2,id =result
        print(result)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9,
        rt = 2, colorR=(255,0,255))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)),
                           scale=1, thickness=1, offset=1)
        cx,cy= x1+w//2,y1+h//2
        cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)

        if limits[0]<cx<limits[2] and limits[1]-15<cy<limits[1]+15:
            if totalcount.count(id)==0:
                totalcount.append(id)
                cv2.line(img, (limits[0], limits[1]),
                         (limits[2], limits[3]), (0, 255, 0), 5)


    cvzone.putTextRect(img, f'Count: {len(totalcount)}', (50,50))





    cv2.imshow("image",img)
    #cv2.imshow("imageregion",imgregion)

    cv2.waitKey(1)
