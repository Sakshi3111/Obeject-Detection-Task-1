#!/usr/bin/env python
# coding: utf-8

# #  THE SPARKS FOUNDATION 
#    ##  OBJECT DETECTION- (OPENCV)
#    ##  NAME- SAKSHI VERMA
#    ## TASK- 1

# In[12]:


# import libraries
import cv2
import matplotlib.pyplot as plt


# In[13]:


get_ipython().run_line_magic('pwd', '')


# In[14]:


# importing ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt and frozen_inference_graph.pb

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

frozen_model = 'frozen_inference_graph.pb'

file_name = 'coco.names'

# test pretrained model

classLabels = [0]

with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')
    classLabels.append(fpt.read())
    
test = cv2.dnn_DetectionModel(frozen_model, config_file)
test.setInputSize(550, 320)
test.setInputScale(1.5 / 127.5)  
test.setInputMean((127.5, 127.5, 127.5)) 
test.setInputSwapRB(True)


# In[15]:


# importing "traffic video.mp4"
cap = cv2.VideoCapture("walk.mp4")  


if not cap.isOpened():
    raise print("video not found")

font_scale = 1
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()

    ClassIndex, confidence, bbox = test.detect(frame, confThreshold=0.55)

    
    if len(ClassIndex) != 0:
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if ClassInd <= 80:
                cv2.rectangle(frame, boxes, (0, 255, 0), 2)
                cv2.putText(frame, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font,
                            fontScale=font_scale, color=(0, 0, 255), thickness=2)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(2) & 0XFF == ord('k'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




