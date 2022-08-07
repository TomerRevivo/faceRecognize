import cv2
import os
import numpy as np
from PIL import Image  # Python Imaging Library
import pickle  # used for serializing and de-serializing a Python object structure

from PIL.Image import Resampling

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # get path of current directory
img_dir_data = os.path.join(CURRENT_DIR, "picsData")  # get path to folder of all images folders

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')  # xml file for front face recognition cascade classifier


recognizer = cv2.face.LBPHFaceRecognizer_create()  # create the recognizer


current_id = 0
label_ids = {}
names_labels_face = []
training_pics_face = []

# run on picsData folder and sub folders
for root, dirs, files in os.walk(img_dir_data):
    for file in files:
        if file.endswith("jpg") or file.endswith("png") or file.endswith("jpeg"):
            path = os.path.join(root, file)  # get file path
            label = os.path.basename(root).replace("_", " ").lower()  # get the folder name

            # give id to folder if not exist in list
            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id_ = label_ids[label]

            # open image, resize
            pil_image = Image.open(path).convert("L")
            size = (550, 550)
            final_image = pil_image.resize(size)

            image_array = np.array(final_image, "uint8")

            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.2, minNeighbors=4)  # detect face algorithm

            for (x, y, w, h) in faces:
                roi = image_array[y:y + h, x:x + w]
                training_pics_face.append(roi)  # add to training
                names_labels_face.append(id_)  # add to labels

# write data to pickle file
with open("labels/face-labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(training_pics_face, np.array(names_labels_face))  # start training
recognizer.save("training/face-trainer.yml")  # save result file
print("finish training.. :"+str(names_labels_face))

