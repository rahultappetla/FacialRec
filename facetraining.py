
import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('C:/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
rec = cv2.createLBPHFaceRecognizer();
rec.load("C:/Users/Rahul Tappetla/Documents/Academics Tamuk/Digital Image Processing/files/face_rec/trainingdata.yml")
id=0
font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
    
        if id==1:
            id="rahul"
        if id==2:
            id="akash"
        if id==3:
            id="sneha"
        if id==4:
            id="shashank"
        if id==5:
            id="vinay"    
        
        cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,255)
    cv2.imshow('img',img)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()