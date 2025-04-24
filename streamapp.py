import os
import base64

import streamlit as st
from PIL import Image
import cv2 as cv
from deepface import DeepFace
import numpy as np
import pandas as pd

from datetime import datetime

if 'view' not in st.session_state:
    st.session_state.view = 'home'
if 'emp_fname' not in st.session_state:
    st.session_state.emp_fname = ''
if 'emp_lname' not in st.session_state:
    st.session_state.emp_lname = ''


# @st.cache_resource
def get_model(model_name):
    
    os.environ['DEEPFACE_HOME'] = 'C:/Users/YASHASWAT/Desktop/FaceDetect/'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    DeepFace.build_model(model_name)
get_model('Facenet')


def change_state(state):
        
    if state == 'home':
        st.session_state.view = 'home'
    
    if state == 'attendance':
        st.session_state.view = 'attendance'
        
    if state == 'add_emp':
        st.session_state.view = 'add_emp'
        
    if state == 'check':
        st.session_state.view = 'check'


def mark_attendance(first_name, last_name, similarity):
    
    print('Marking...')
    
    file_name = 'attendance.csv'
    entry = {
        'first name': first_name,
        'last name': last_name,
        'similarity score': similarity,
        'date': datetime.now().date(),
        'timestamp': datetime.now().time(),
        'image path': f'http://localhost:8501/app/static/{first_name}_{last_name}.jpg'
    }
    
    if os.path.exists(file_name):
        
        df = pd.read_csv('attendance.csv')
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    else:
        df = pd.DataFrame([entry])
    
    df.to_csv(file_name, index=False)


def face_recog(img):
    
    bytes_data = img.getvalue()
    cv2_img = cv.imdecode(np.frombuffer(bytes_data, np.uint8), cv.IMREAD_COLOR)
    
    try:
        
        recog = DeepFace.find(cv2_img, db_path=DB_PATH, model_name='Facenet', threshold=0.25, 
                       detector_backend='opencv', distance_metric='cosine', align=True, normalization='Facenet')
        print(recog[0]['identity'])
        
        result_box = st.container(border=True)
        
        if recog[0].shape[0]:
            result_box.subheader(':green[Identity Verified] ✅', anchor=False)
            identity = recog[0]['identity'][0].lstrip(f'{DB_PATH}/').rstrip('.jpg').split('_')
            
            first_name = identity[0]
            last_name = identity[1]
            similarity = round(100*(1- recog[0]['distance'][0]), 2)
            
            result_box.write(f'**Name**: {first_name} {last_name}')
            result_box.write(f'**Similarity**: {similarity}%')
            
            mark_attendance(first_name, last_name, similarity)
            st.success('Attendance marked!', icon='✅')
            
        else:
            result_box.subheader(':red[Person Not Found] :x:', anchor=False)
            
    except ValueError:
        pass
    #     st.subheader(':red[Person Not Found] :heavy_multiplication_x:', anchor=False)


def save_employee(fname, lname, image):
    with open(f'{DB_PATH}/{fname}_{lname}.jpg', 'wb') as file:
        file.write(image.getvalue())
    print('EMP IMAGE SAVED.')


def clear_form():
    
    st.session_state.emp_fname = ''
    st.session_state.emp_lname = ''


col1, col2 = st.columns([0.3, 0.7], vertical_alignment='center')
col1.image('excellent_logo.png')
col2.title('FaceDetect Attendance System', anchor=False)
st.subheader('Mark your attendance, the easy way...', anchor=False)
st.divider()

DB_PATH = 'static/'

if st.session_state.view == 'home':
    
    st.write('\n')
    attend = st.button('Take Attendance', use_container_width=True, type='primary',
                       on_click=change_state, args=('attendance',))
    st.write('\n')
    add_emp = st.button('Add Employee', use_container_width=True, type='primary',
                        on_click=change_state, args=('add_emp',))
    st.write('\n')
    view_att = st.button('View Attendance', use_container_width=True, type='primary',
                        on_click=change_state, args=('check',))  
    
if st.session_state.view == 'attendance':
    
    st.header('Employee Attendance', anchor=False)
    
    cam_photo = st.camera_input('Capture your image: ')
    instruction = st.info('Please keep only your shoulder up view in the camera frame.')

    if cam_photo is not None:
        face_recog(cam_photo)
    
    instruction.empty()
    st.write('\n')
    st.button('Back to Home', type='secondary',
              on_click=change_state, args=('home',))
        
if st.session_state.view == 'add_emp':
    
    info = st.empty()
    emp_photo = None
    st.header('Add new employee', anchor=False)
    
    with st.container(border=True):
        
        emp_fname = st.text_input('Enter first name: ', key='emp_fname')
        emp_lname = st.text_input('Enter last name: ', key='emp_lname')
        
        img_type = st.segmented_control('Add photo: ', options = ['Upload', 'Capture'])
                
        if img_type == 'Upload':
            emp_photo = st.file_uploader('Upload your passport size photo: ', type=['jpg', 'jpeg'])
        if img_type == 'Capture':
            emp_photo = st.camera_input('Capture your image: ')
            info = st.info('Please keep only your shoulder up view in the camera frame.')
            

        st.write('\n')
        button1, button2 = st.columns([0.7, 0.3], gap='medium')
        submit = button1.button('Submit', type='primary', use_container_width=True,
                           on_click=save_employee, args=(emp_fname, emp_lname, emp_photo))
        if submit:
            info.empty()
            st.success('Employee Identity added to Database!')
        reset_form = button2.button('Reset Form', use_container_width=True,
                                    on_click=clear_form)

    st.write('\n')
    st.button('Back to Home', on_click=change_state, args=('home',))

if st.session_state.view == 'check':
    
    st.header('Employee Attendance Record', anchor=False)
    
    st.write('\n')
    edit_mode = st.toggle('Edit Mode')
    
    attendance = pd.read_csv('attendance.csv')
    display_box = st.container(border=True)
    mode_badge = display_box.empty()
    table_box = display_box.empty()
    
    st.write('\n')
    st.button('Back to Home', on_click=change_state, args=('home',))
    
    if edit_mode:
        mode_badge.badge('Edit Mode', color='orange', icon=':material/edit_note:')
        table_box.data_editor(attendance, num_rows='dynamic', use_container_width=True)
    else:
        mode_badge.badge('View Mode', color='violet', icon=':material/visibility:')
        table_box.dataframe(attendance, use_container_width=True,
                              column_config={
                                  'image path': st.column_config.ImageColumn('Employee Photo', pinned=True)
                              })
    
    