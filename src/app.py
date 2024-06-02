import cv2
import os
import sys
from matplotlib import pyplot as plt
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

import utils


st.title("Sudoku Solver using CNN")

# artifacts\scratch_char74k.h5
model = tf.keras.models.load_model(os.path.join('artifacts/','scratch_char74k_3.h5'))

# model = tf.load

session_state = dict()
if 'imageCaptured' not in st.session_state.keys():
    session_state['imageCaptured'] = None
st.write('Upload Sudoku image')
session_state['imageCaptured'] = st.file_uploader('Sudoku image',type = ['.jpeg','.png','.jpg'],accept_multiple_files=False)
col1,col2,col3 = st.columns(3)

# with col1:
    # captureSudoku = st.camera_input("Show sudoku image",key = "sudokuImage",help = "Place the Sudoku image correctly for better results")

    # if captureSudoku:
        # session_state['imageCaptured'] = captureSudoku

with col1:
    st.subheader('Unsolved Sudoku')
    if session_state['imageCaptured']:
        st.image(session_state['imageCaptured'])
with col2:
    pass

with col3:
    st.subheader('Solved sudoku')
    if session_state['imageCaptured']:
        img = Image.open(session_state['imageCaptured'])
        img = np.array(img)
        result = utils.preprocess_image(img)
        # st.image(result)
        img_res,contours = utils.find_contours(result)
        warped_image = utils.warping_image(img,contours)
        warp_cleaned_image = utils.preprocess_image(warped_image)
        horizontal_lines = utils.get_horizontal_lines(warp_cleaned_image)
        vertical_lines = utils.get_vertical_lines(warp_cleaned_image)
        grid_warped = utils.draw_grid_lines(warp_cleaned_image,horizontal_lines,vertical_lines)
        # plt.imshow(vertical_lines)
        squares = utils.get_cells(grid_warped)
        cleaned_squares,sudoku_numbers = utils.predict_digits(squares,model)
        completed_sudoku = utils.sudoku_solver(sudoku_numbers)
        print('********sudoku_numbers************')
        print(sudoku_numbers)
        print(completed_sudoku)
        if completed_sudoku is not None:
            final_result = utils.draw_digits_on_wraped(warped_image,completed_sudoku,cleaned_squares)
            st.image(final_result)
        else:
            st.write('The sudoku predictions are incorrect')


    # sudoku_image = session_state['imageCaptured']







