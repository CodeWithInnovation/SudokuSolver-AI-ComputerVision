import processing 
import cv2
import argparse
import solve_sudoku as solver
import copy 
import numpy as np
#from skimage.segmentation import clear_border
#import imutils
from keras.models import load_model
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to sudoku image")
ap.add_argument("-m", "--model", required=True,
	help="ath to trained digit classifier")
args = vars(ap.parse_args())

img=cv2.imread(args['image'])
result_image,warped=processing.extract_sudoku(img)
model=load_model(args['model'])
sudoku_grid,cellLocs=processing.extract_digit(warped,model)

puzzle=copy.deepcopy(sudoku_grid)
if solver.solve(puzzle):
    print("------------ Solving sudoku puzzle ------------")
    res_img=copy.deepcopy(result_image)
    for i in range(len(puzzle)):
          for j in range(len(puzzle)):	  
            box=cellLocs[i][j]
            digit=puzzle[i][j]
            # unpack the cell coordinates
            startX, startY, endX, endY = box
            # compute the coordinates of where the digit will be drawn
            # on the output puzzle image
            textX = int((endX - startX) * 0.33)
            textY = int((endY - startY) * -0.2)
            textX += startX
            textY += endY
            # draw the result digit on the Sudoku puzzle image
            if(sudoku_grid[i][j]==0):
                cv2.putText(res_img, str(digit), (textX, textY),cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 4)
    cv2.imwrite("solution.png",res_img)
    current_directory = os.getcwd()
    print(f"Solution saved in : {current_directory}\solution.png")         
else:
    print("No solution exists.")
