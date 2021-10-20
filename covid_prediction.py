

from PIL import Image
import cv2
# import wget
# from tkinter import filedialog
import pathlib

from tensorflow.keras.models import load_model
# from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import csv

# img_path = ""


def predict():

    PATH_TO_TEST_IMAGES_DIR = pathlib.Path('./Testset')
    TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpeg")))
    print(TEST_IMAGE_PATHS)
    count = 0
    
    print("[INFO] loading network...")
    model =load_model('./covid_model.h5')

    labels = ['Covid','Normal'] #These labels will be used for showing output
    
    for img in TEST_IMAGE_PATHS:
        print(f"[INFO] reading image {img}")
        frame = np.array(Image.open(img))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = cv2.imread(img)

        roi_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.resize(frame,(224,224)) # Resize
        roi = roi_gray.astype('float')/255.0 # Scaling - 0-255 to 0-1
        roi = img_to_array(roi)
        roi = np.expand_dims(roi,axis=0) # Single array conversion

        print("[INFO] classifying image...")

        preds = model.predict(roi)[0] # [0] element is image. returns array of size of class with its respective probability.
        #print(preds)
        #print(preds.argmax())
        label=labels[preds.argmax()] # returns the index class with max probability/


        # if(label=='Bulbasur'):
        #     image = cv2.rectangle(frame, start_point, end_point, (0,0,255), thickness)
        #     cv2.putText(image,label,(30,60),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3)
        # else:
        #     image = cv2.rectangle(frame, start_point, end_point, (0,255,0), thickness)
        #     cv2.putText(image,label,(30,60),cv2.FONT_HERSHEY_SIMPLEX,1.6,(0,0,0),3)
            
        # cv2.imshow('COVID Detector',frame)
        count+=1
        print(f"[INFO] Image {count} : {label}")
        classification = open('classification.csv', mode='a', newline='')
        classify_writer = csv.writer(classification, delimiter=',')
        classify_writer.writerow([count, img, label])
        # cv2.imwrite("./Output/detected13.jpg",frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #if(label=='Covid'):
         #   tkinter.messagebox.showinfo("COVID Predicted!","Take Care. Be Alert.\t \nStay Safe.")
        #else:
         #   tkinter.messagebox.showinfo("NORMAL Report","Don't Worry! \nYour Report is Normal")

if __name__ == '__main__':
    predict()
