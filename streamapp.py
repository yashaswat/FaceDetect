import os

import streamlit as st
from PIL import Image
import cv2 as cv
from deepface import DeepFace
import numpy as np

if 'view' not in st.session_state:
    st.session_state.view = 'home'


@st.cache_resource
def get_model(model_name):
    
    os.environ['DEEPFACE_HOME'] = 'C:/Users/YASHASWAT/Desktop/FaceDetect/'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    DeepFace.build_model(model_name)
get_model('Facenet')


def change_state(state):
    
    print(st.session_state.view)
    
    if state == 'home':
        st.session_state.view = 'home'
    
    if state == 'attendance':
        st.session_state.view = 'attendance'
        
    if state == 'add_emp':
        st.session_state.view = 'add_emp'


def face_recog(img):
    
    global reference    
    bytes_data = img.getvalue()
    cv2_img = cv.imdecode(np.frombuffer(bytes_data, np.uint8), cv.IMREAD_COLOR)
    
    try:
        
        recog = DeepFace.verify(cv2_img, reference, model_name='Facenet', 
                           detector_backend='opencv', align=True, threshold=0.45, normalization='Facenet')
        
        if recog['verified']:
            st.header('True')
            st.subheader(f'Distance: {recog['distance']}')
        else:
            st.header('False')
            st.subheader(f'Distance: {recog['distance']}')
    except ValueError:
        st.header('False')
        st.subheader(f'Distance: {recog['distance']}')

col1, col2, col3 = st.columns(3)
col2.title('FaceDetect', anchor=False)

reference = 'Verified Faces/yash_pport.jpg'

if st.session_state.view == 'home':
    
    attend = st.button('Take Attendance', use_container_width=True, on_click=change_state, args=('attendance',))
    add_emp = st.button('Add Employee', use_container_width=True, on_click=change_state, args=('add_emp',))

if st.session_state.view == 'attendance':
    
    st.header('Employee Attendance', anchor=False)
    
    cam_photo = st.camera_input('Capture your image: ')
    st.info('Please keep only your shoulder up view in the camera frame.')

    if cam_photo is not None:
        face_recog(cam_photo)
        
if st.session_state.view == 'add_emp':
    
    st.header('Add new employee', anchor=False)
    
    with st.container(border=True):
        
        emp_fname = st.text_input('Enter first name: ')
        emp_lname = st.text_input('Enter last name: ')
        
        img_type = st.segmented_control('Add photo: ', options = ['Upload', 'Capture'])
                
        if img_type == 'Upload':
            emp_photo = st.file_uploader('Upload your passport size photo: ', type=['jpg', 'jpeg'])
        if img_type == 'Capture':
            emp_photo = st.camera_input('Capture your image: ')
            st.info('Please keep only your shoulder up view in the camera frame.')
        
        st.write('\n')
        submit = st.button('Submit', type='primary', use_container_width=True)
    