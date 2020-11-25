from collections import deque
from imutils.video import VideoStream
import time
import numpy as np
import cv2 as cv
import imutils

maxDequeLen = 64
pts = deque(maxlen=maxDequeLen)

# turning on main web-cam
# cap = cv.VideoCapture(0)

# turning on iVCam app
cap = cv.VideoCapture(1)

time.sleep(2.0)

if not cap.isOpened():
	print('Cannot open camera')
	exit()

while True:
	ret, frame = cap.read()

	if not ret:
		print("Can't receive frame (stream end?). Exiting ...")
		break

	# resize the frame
	frame = imutils.resize(frame, width=500)
	# blur it
	frame_gau_blurred = cv.GaussianBlur(frame, (13, 13), 0)
	# convert from BGR to HSV color space
	hsv = cv.cvtColor(frame_gau_blurred, cv.COLOR_BGR2HSV)

	# range of color
	lower_color = np.array([145, 50, 50])
	upper_color = np.array([170, 255, 255])

	# getting the range of pink color in frame
	color_mask = cv.inRange(hsv, lower_color, upper_color)
	color_mask = cv.erode(color_mask, None, iterations=3)
	color_mask = cv.dilate(color_mask, None, iterations=3)

	res_color = cv.bitwise_and(frame_gau_blurred, frame_gau_blurred, mask=color_mask)

	color_s_gray = cv.cvtColor(res_color, cv.COLOR_BGR2GRAY)

	# canny_edge = cv.Canny(color_s_gray, 50, 240)

	circles = cv.HoughCircles(color_s_gray, cv.HOUGH_GRADIENT_ALT, dp=1.5,
							minDist=20, param1=100, param2=0.8,
							minRadius=10, maxRadius=-1)

	center = None
	if circles is not None:
		circles = np.uint16(np.around(circles))
		max_circle = max(circles[0,:], key=lambda x:x[2])

		for i in [max_circle]:
			# drawing on detected circle and its center
			center = (i[0], i[1])
			cv.circle(frame, center, i[2], (0, 255, 0), 2)
			cv.circle(frame, center, 2, (255, 0, 0), 3)
	pts.appendleft(center)

	# loop over the set of tracked points
	for i in range(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
		if pts[i - 1] is None or pts[i] is None:
			continue
		# otherwise, compute the thickness of the line and
		# draw the connecting lines
		thickness = int(np.sqrt(maxDequeLen / float(i + 1)) * 2.5)
		cv.line(frame, pts[i - 1], pts[i], (255, 0, 0), thickness)
	# show the frame to our screen
	cv.imshow('CIRCLES', frame)
	# cv.imshow('COLOR MASK', color_mask)
	cv.imshow('GRAY', color_s_gray)
	# cv.imshow('CANNY', canny_edge)
	key = cv.waitKey(1) & 0xFF
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

cap.release()
# close all windows
cv.destroyAllWindows()
