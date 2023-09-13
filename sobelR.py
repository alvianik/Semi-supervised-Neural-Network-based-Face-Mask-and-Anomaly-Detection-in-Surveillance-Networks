# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 01:11:44 2021

@author: Dell
"""

import cv2
import glob
from PIL import Image
import os


input_path = glob.glob("E:/Anik Alvi/unsupervised-face-mask-detection/mtcnn-face-detection/code/mtcnn/resizedR/*.jpg") #It is the path to images from where it will read
output_path = "E:/Anik Alvi/unsupervised-face-mask-detection/mtcnn-face-detection/code/mtcnn/sobelR/" ##It is the path to images to where it will write

for file in input_path:
    
    # Choose an image to detect faces in
    img = cv2.imread(file)
     
    # Applying Sobel Edge to face image
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S 

    grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)


    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)


    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    im = Image.fromarray(grad)


    src_fname, ext = os.path.splitext(file)  # split filename and extension
    # construct output filename, basename to remove input directory
    save_fname = os.path.join(output_path, os.path.basename(src_fname)+'.jpg')
    im.save(save_fname)