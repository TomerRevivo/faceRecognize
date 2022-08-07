import cv2
import numpy as np
import pickle

# xml for front face recognition cascade classifier
fc = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("training/face-trainer.yml") # read the training file

labels = {"person_name": 1}
with open("labels/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

# get the video
cap = cv2.VideoCapture(0)

while(True):
    # catch frame by frame
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # turn color from Blue Green Red to gray
    faces = fc.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4) # detect face algorithm

    # for loop on every face detection
    for(x,y,w,h) in faces:
        # draw a rectangle around face
        color_face = (0, 0, 255)  # color of rectangle borders
        stroke = 4  # border width
        end_cord_x = x + w  # end point x - x+width
        end_cord_y = y + h  # end point y - y+height
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color_face, stroke)  # create rectangle

        roi_gray = gray[y:y + h, x:x + w]  # area to transfer the image to gray - open cv need this
        roi_color = frame[y:y + h, x:x + w] # cut to show only face

        id_, conf = recognizer.predict(roi_gray)

        color_face = (255, 255, 255)
        stroke = 2
        font = cv2.FONT_HERSHEY_SIMPLEX

        if  60<= conf <= 80:
            id2_ , conf2 = recognizer.predict(roi_gray)
            name = labels[id_]
            color_face = (0,255,0)
            cv2.putText(frame, name, (x, y), font, 1, color_face, stroke, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color_face, stroke)  # create rectangle
            print(str(conf) + " " + name)
        else:
            cv2.putText(frame, "UNRECOGNIZED PERSON", (x, y), font, 0.7, color_face, stroke, cv2.LINE_AA)

        img_item = "capture_image.png"
        cv2.imwrite(img_item, roi_gray) # save image as "capture_image.png"



    cv2.imshow('fram',frame)

    # close program by pressing 'q'
    if cv2.waitKey(20) & 0Xff == ord('q'):
        break

# release camera
cap.release()

# close all windows
cv2.destroyAllWindows()

