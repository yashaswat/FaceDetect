import setuptools.dist
import threading
import os

import numpy as np
from PIL import Image
import cv2 as cv
from deepface import DeepFace

import setup


# function checks if webcam frame matches reference image
# updates face_match flag
def check_face(frame, model, threshold):
    
    global face_match
    global reference_img
    
    try:
        if DeepFace.verify(frame, reference_img.copy(), model_name=model, 
                           detector_backend='opencv', align=True, threshold=threshold)['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        face_match = False



# opens webcam and applies face recognition on live feed
def webcam_recognition(model, threshold):
    
    camera = cv.VideoCapture(0) # starting webcam feed
    frame_no = 0 # counter
    
    if not camera.isOpened():
        
        print('couldnt open cam')
        exit()
        
    else:
        
        while True:
            
            ret, frame = camera.read() # read frame from webcam
            
            if ret:
                
                # run check_face function every 30 frames
                if frame_no % 30 == 0:
                    try:
                        threading.Thread(target=check_face, args=(frame.copy(), model, threshold)).start()
                    except ValueError:
                        pass
                
                frame_no += 1
                
                # display match status on webcam feed
                if face_match:
                    cv.putText(frame, 'MATCH!', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv.LINE_AA)
                else:
                    cv.putText(frame, 'NO MATCH!', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv.LINE_AA)

                cv.imshow('face video', frame)
            
            # key press handling
            key = cv.waitKey(1) & 0xFF        
            
            if key == ord('c'): # capture and save frame
                            
                cv.imwrite(f'Snapshots/frame{frame_no}.jpg', frame)
                frame_no += 1
                continue
            
            if key == ord('q'): # quit
                break
        
    camera.release()
    cv.destroyAllWindows()
    

# env variable to download model weights in current folder
os.environ['DEEPFACE_HOME'] = 'C:/Users/YASHASWAT/Desktop/FaceDetect/' 
DeepFace.build_model("Facenet") # download model weights

# loading reference image for verification
reference_img = cv.imread('Verified Faces/yash_face2.jpg')
face_match = False # face match flag

model = 'Facenet' # model to use for face recognition
threshold = 0.4 # threshold for embedding distance

webcam_recognition(model, threshold)