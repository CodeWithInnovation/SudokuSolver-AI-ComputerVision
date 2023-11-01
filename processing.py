import numpy as np
import cv2
import imutils
from skimage.segmentation import clear_border

# Extract the sudoku from the image
def extract_sudoku(img,draw_cnt=False):
  # Pre Processing the image
  img = imutils.resize(img, width=600)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # apply a Gaussian blur with a kernel size (height, width)
  img_proc = cv2.GaussianBlur(gray.copy(), (7, 7), 3)
  # binary adaptive thresholding operations allow us to peg grayscale pixels toward each end of the [0, 255] pixel range
  img_proc = cv2.adaptiveThreshold(img_proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
  # to make gridlines have non-zero pixel values, we will invert the colours
  img_proc = cv2.bitwise_not(img_proc)

  """
  Find contours in the thresholded image and sort them by size in descending order 
  and then find the 4 extreme corners of the largest contour in the image """ 

  contours = cv2.findContours(img_proc.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = imutils.grab_contours(contours)
  contours = sorted(contours, key=cv2.contourArea, reverse=True)

  # initialize a contour that corresponds to the puzzle outline
  puzzleCnt = None
  # find the 4 extreme corners of the largest contour in the image.
  for c in contours:
      # approximate the contour
      peri = cv2.arcLength(c, True)
      approx = cv2.approxPolyDP(c, 0.02 * peri, True)
      # if our approximated contour has four points, then we can
      # assume we have found the outline of the puzzle
      if len(approx) == 4:
        puzzleCnt = approx
        break
  # if the puzzle contour is empty raise an error
  if puzzleCnt is None:
      raise Exception(("Could not find Sudoku puzzle outline. "
				"Try debugging your thresholding and contour steps."))
  if draw_cnt:
      cv2.drawContours(img, [puzzleCnt], 0, (255, 0, 0), 2)

  cnt=puzzleCnt.reshape(4, 2)
	
  # Find the oordinate of the contour
  top_left = np.min(cnt, axis=0)
  top_right = np.array([np.max(cnt[:, 0]), np.min(cnt[:, 1])])
  bottom_left = np.array([np.min(cnt[:, 0]), np.max(cnt[:, 1])])
  bottom_right = np.max(cnt, axis=0)

  src_pts = np.array([top_left, top_right, bottom_left, bottom_right], dtype='float32')
  side = max([  distance_between(bottom_right, top_right),
							distance_between(top_left, bottom_left),
							distance_between(bottom_right, bottom_left),
							distance_between(top_left, top_right) ])
  dst_pts = np.array([[0, 0], [side, 0], [0,side], [side, side]], dtype='float32')
	
  # Calculate the perspective transform matrix
  matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
  result_image=cv2.warpPerspective(img, matrix, (int(side), int(side)))
  warped=cv2.warpPerspective(gray, matrix, (int(side), int(side)))
    
  return result_image, warped

def distance_between(p1, p2):
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))

def extract_digit(warped,model):
  step_x = warped.shape[1] // 9
  step_y = warped.shape[0] // 9
  sudoku_grid=[]
  cellLocs=[]
  for y in range(0, 9):
    # initialize the current list of cell locations
    row_loc=[]
    # initialize the current row value 
    row = []
    for x in range(0, 9):
      # compute the starting and ending (x, y)-coordinates of the
      # current cell
      xmin = x * step_x
      ymin = y * step_y
      xmax = (x + 1) * step_x
      ymax = (y + 1) * step_y
      
      # add the (x, y)-coordinates to our cell locations list
      row_loc.append((xmin, ymin, xmax, ymax))
      
      cell = warped[ymin:ymax, xmin:xmax]

      im = cell.copy()
      try:
        thresh = cv2.threshold(cell, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = clear_border(thresh)
      except:
        return None, None
      cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
      cnts = imutils.grab_contours(cnts)
      if len(cnts) == 0:
        row.append(0)
      else:
        c = max(cnts, key=cv2.contourArea)
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        # compute the percentage of masked pixels relative to the total
        # area of the image
        (h,w)= thresh.shape
        percentFilled = cv2.countNonZero(mask) / float(w * h)
        # if less than 3% of the mask is filled then we are looking at
        # noise and can safely ignore the contour
        if percentFilled < 0.03:
          row.append(0)
        else:
          # apply the mask to the thresholded cell
          digit = cv2.bitwise_and(thresh, thresh, mask=mask)
          #cv2.imwrite("images/mat" + str(y) + str(x) + ".png",digit)
          digit=cv2.resize(digit, (28, 28))
          digit = digit[:, :, np.newaxis]
          digit = np.expand_dims(digit, 0)
          digit = digit.astype("float") / 255.0
          prediction = model.predict(digit)
          row.append(prediction.argmax(axis=1)[0])
    sudoku_grid.append(row)
    cellLocs.append(row_loc) 
  return sudoku_grid,cellLocs
