from collections import deque
from imutils.video import VideoStream
import time
import numpy as np
import cv2 as cv
import imutils

WINDOW_NAME = '[Ball tracking] Elfimov 6222'

greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

maxDequeLen = 64
pts = deque(maxlen=maxDequeLen)

cap = cv.VideoCapture(0)

# allow the camera to warm up
time.sleep(2.0)

if not cap.isOpened():
	print('Cannot open camera')
	exit()

while True:
	ret, frame = cap.read()

	if not ret:
		print("Can't receive frame (stream end?). Exiting ...")
		break

	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=600)
	blurred = cv.GaussianBlur(frame, (11, 11), 0)
	hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

	# construct a mask for the color XXX, then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv.inRange(hsv, greenLower, greenUpper)
	mask = cv.erode(mask, None, iterations=2)
	mask = cv.dilate(mask, None, iterations=2)

	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
							cv.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None
	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv.contourArea)
		((x, y), radius) = cv.minEnclosingCircle(c)
		M = cv.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		# only proceed if the radius meets a minimum size
		if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv.circle(frame, (int(x), int(y)), int(radius),
					   (0, 255, 255), 2)
			cv.circle(frame, center, 5, (0, 0, 255), -1)
	# update the points queue
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
		cv.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
	# show the frame to our screen
	cv.imshow(WINDOW_NAME, mask)
	key = cv.waitKey(1) & 0xFF
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

cap.release()
# close all windows
cv.destroyAllWindows()
