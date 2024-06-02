import os
import sys 
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import dill

import cv2

from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok = True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)

def preprocess_image(img):
    try:
        # img = cv2.imread(img)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(9,9),0)
        thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,1)
        inverted = cv2.bitwise_not(thresh)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        morph = cv2.morphologyEx(inverted,cv2.MORPH_OPEN,kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
        result = cv2.dilate(morph,kernel,iterations = 2)
        return result
    except Exception as e:
        raise CustomException(e,sys)
    
def find_contours(result):
    try:
        mask = np.zeros((result.shape),np.uint8)
        contour,hier = cv2.findContours(result,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        best_cnt = None
        for cnt in contour:
            area = cv2.contourArea(cnt)
            if area > 1000:
                if area > max_area:
                    max_area = area
                    best_cnt = cnt
        cv2.drawContours(mask,[best_cnt],0,255,-1)
        cv2.drawContours(mask,[best_cnt],0,100,10)

        res = cv2.bitwise_and(result,mask)
        return (res,best_cnt)
    except Exception as e:
        raise CustomException(e,sys)

def warping_image(img,best_cnt):
    try:
        perimeter = cv2.arcLength(best_cnt,True)
        approx = cv2.approxPolyDP(best_cnt,0.02*perimeter,True)
        if(len(approx) == 4):
            (top_left, top_right,bottom_left,bottom_right) = (approx[0][0],approx[-1][0],approx[1][0],approx[2][0])
        width = max(abs(top_left[0]-top_right[0]),
            abs(bottom_right[0]-bottom_left[0]),abs(top_left[1]-bottom_right[1]),abs(top_right[1]-bottom_right[1]))
        perspective_matrix = cv2.getPerspectiveTransform(np.float32([top_left,top_right,bottom_left,bottom_right]),
                                                 np.float32([(0,0),(width,0),(0,width),(width,width)]))
        wresult = cv2.warpPerspective(img,perspective_matrix,(width,width))
        return wresult
    except Exception as e:
        raise CustomException(e,sys)

def get_horizontal_lines(wresult):
    rows = wresult.shape[0]
    size = rows//10
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(size,1))
    wresult = cv2.erode(wresult,kernel,iterations = 1)
    wresult = cv2.dilate(wresult,kernel,iterations = 1)
    plt.imshow(wresult)
    plt.show()
    plt.savefig("result.png")
    return wresult

def get_vertical_lines(wresult):
    cols = wresult.shape[1]
    size = cols//10
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,size))
    wresult = cv2.erode(wresult,kernel,iterations = 1)
    wresult = cv2.dilate(wresult,kernel,iterations = 1)
    return wresult

def draw_grid_lines(wresult,wresult_hor,wresult_ver):
    try:
        grid = cv2.add(wresult_hor,wresult_ver)
        grid = cv2.erode(grid,cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)),iterations = 2)
        grid = cv2.dilate(grid,cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)),iterations = 2)
        print('***********************yes********************')
        print(grid.shape)
        print(len(grid))
        # print(grid)
        # plt.imshow(grid)
        # grid = image_to_array(grid)
        pts = cv2.HoughLines(grid,.3,np.pi/90,200)
        grid_1 = grid.copy()
        pts = np.squeeze(pts)
        for p in pts:
            x0 = p[0]*np.cos(p[1])
            y0 = p[0]*np.sin(p[1])
            x1 = int(x0+1000*(-np.sin(p[1])))
            y1 = int(y0+1000*(np.cos(p[1])))
            x2 = int(x0-1000*(-np.sin(p[1])))
            y2 = int(y0-1000*(np.cos(p[1])))
            cv2.line(grid_1,(x1, y1), (x2, y2), (255, 255, 255), 1)
        return cv2.bitwise_or(wresult,grid_1)
    except Exception as e:
        raise CustomException(e,sys)

def get_cells(wresult):
    width = wresult.shape[1]//9
    squares = []
    for i in range(9):
        for j in range(9):
            p1 = (i*width,j*width)
            p2 = ((i+1)*(width),(j+1)*(width))
            squares.append(wresult[p1[0]+6:p2[0],p1[1]+6:p2[1]])
    return squares

def clean(img):
    if(np.isclose(img,0).sum()/(img.shape[0]*img.shape[1]) >= 0.95):
        return np.zeros_like(img),False

    height,width = img.shape
    mid = width//2
    # print(np.isclose(img[:,int(mid-width*0.4):int(mid+width*0.4)],0).sum()/(2*width*0.4*height))

    if(np.isclose(img[:,int(mid-width*0.4):int(mid+width*0.4)],0).sum()/(2*width*0.4*height) >= 0.90):
        return np.zeros_like(img),False

    #centering image
    contours, _ = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours,key = cv2.contourArea,reverse = True)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    # plt.imshow(cv2.rectangle(img.copy(), (x, y), (x + w, y + h), 128, 2))
    start_x = (width-w)//2
    start_y = (width-h)//2
    new_img = np.zeros_like(img)
    new_img[start_y:start_y+h,start_x:start_x+w] = img[y:y+h,x:x+w]
    return new_img,True

def predict_digits(squares,model):
    cleaned_squares = []
    sudoku_numbers = ''
    for square in squares:
        
        new_img,is_img = clean(square)
        if(is_img):
            cleaned_squares.append(new_img)
            i = cv2.resize(new_img,(150,150))
            i = cv2.adaptiveThreshold(i,255,0,0,129,0)
            i = np.array(255) - i
            i = i/255
            probs = model.predict(tf.expand_dims(i,0))
            predicted_categoris = tf.argmax(probs,axis = 1)
            sudoku_numbers += str(int(predicted_categoris[0]))

        else:
            cleaned_squares.append(0)
            sudoku_numbers += '0'

    return (cleaned_squares,sudoku_numbers)

def sudoku_solver(sudoku_numbers):
    if len(sudoku_numbers) != 81:
        return None
    arr = list(sudoku_numbers)
    arr = [int(i) for i in arr]
    arr = np.array(arr).reshape(9,9)
    if solve(arr,0,0):
        return arr
    else:
        return None
    
def solve(arr,row,col):
    if row == 8 and col == 9:
        return True
    if col == 9:
        row += 1
        col = 0
    if arr[row][col] != 0:
        return solve(arr,row,col+1)
    for i in range(1,10,1):
        if is_safe(arr,row,col,i):
            arr[row][col] = i
            if solve(arr,row,col+1):
                return True
            arr[row][col] = 0
    return False

def is_safe(arr,row,col,num):
    for i in range(9):
        if arr[row][i] == num:
            return False
        if arr[i][col] == num:
            return False
    startRow = row - row % 3
    startCol = col - col % 3
    for i in range(3):
        for j in range(3):
            if arr[i+startRow][j+startCol] ==num:
                return False
    return True

def draw_digits_on_wraped(wresult,solved_puzzle,cleaned_squares):
  width = wresult.shape[0]//9
  img_w_text = np.zeros_like(wresult)
  index = 0
  for j in range(9):
    # print('index-j',index)
    for i in range(9):
      # print('index-i',index)
      if type(cleaned_squares[index]) == int:
        p1 = (i * width, j * width)  # Top left corner of a bounding box
        p2 = ((i + 1) * width, (j + 1) * width)  # Bottom right corner of bounding box

        center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        text_size, _ = cv2.getTextSize(str(solved_puzzle[j][i]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        text_origin = (center[0] - text_size[0] // 2, center[1] + text_size[1] // 2)

        cv2.putText(wresult, str(solved_puzzle[j][i]),text_origin, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        # print(solved_puzzle[i][j])
      index += 1
        # index += 1

  return wresult
