#Import All the Required Libraries
import json
import cv2
from matplotlib.pyplot import gray
from ultralytics import YOLO
import numpy as np
import math
import re
import os
import sqlite3
from datetime import datetime
from paddleocr import PaddleOCR
import easyocr
read_char= easyocr.Reader(['bn'])

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#Create a Video Capture Object
cap = cv2.VideoCapture("data/bandicam.mp4")
#Initialize the YOLOv10 Model
model = YOLO("weights/New folder/best.pt")
#Initialize the frame count
count = 0
#Class Names
className = ["10"]
#Initialize the Paddle OCR
# ocr = PaddleOCR(use_angle_cls = True, use_gpu = False)



# def paddle_ocr(frame, x1, y1, x2, y2):
#     frame = frame[y1:y2, x1: x2]
#     result = ocr.ocr(frame, det=False, rec = True, cls = False)
#     text = ""
#     for r in result:
#         #print("OCR", r)
#         scores = r[0][1]
#         if np.isnan(scores):
#             scores = 0
#         else:
#             scores = int(scores * 100)
#         if scores > 60:
#             text = r[0][0]
#     pattern = re.compile('[\W]')
#     text = pattern.sub('', text)
#     text = text.replace("SUB", "")
#     text = text.replace("O", "0")
#     text = text.replace("粤", "")
#     return str(text)



startTime = datetime.now()
license_plates = set()


while True:
    ret, frame = cap.read()
    if ret:
        currentTime = datetime.now()
        count += 1
        print(f"Frame Number: {count}")
        results = model.predict(frame, conf = 0.45)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                classNameInt = int(box.cls[0])
                clsName = className[classNameInt]
                conf = math.ceil(box.conf[0]*100)/100
                label = f'{clsName}:{conf}'
                # label = paddle_ocr(frame, x1, y1, x2, y2)
                  #bangla character reading with 'bn'
                finalimage=frame[y1:y2-35,x1:x2-1]
                cv2.imshow("image",finalimage)
                # finalimage=gray[x1:x2+1, y1:y2+1]
                detected= read_char.readtext(finalimage)

                print(detected)
                if label:
                    license_plates.add(label)
                textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                c2 = x1 + textSize[0], y1 - textSize[1] - 3
                cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255,255,255], thickness=1, lineType=cv2.LINE_AA)
        if (currentTime - startTime).seconds >= 20:
            endTime = currentTime
            startTime = currentTime
            license_plates.clear()
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break
    else:
        break


    
cap.release()
cv2.destroyAllWindows()
