import cv2
import numpy as np
import matplotlib
import collections
import itertools
import os
import math
import argparse
import skimage
import copy
import imutils

from imutils.video import VideoStream, FPS
from collections import defaultdict
from matplotlib import pyplot as plt
from skimage.draw import line, circle, rectangle


# HSV boundaries to detect various portion of the map: the ice, yellow_stone, red_stone, gray color, 
# the bands between two ice surface, the scoring table, all kinds of red, all kinds of yellow.
# the inner_circle (button) and the outer_circle (the house)
BOUNDARIES = {
	"ice": [([0, 0, 175], [179, 46, 255])],
	"yellow_stone": [([21, 93, 140], [54, 255, 255])],
	"red_stone": [([176, 214, 63], [179, 255, 183])],
	"gray": [([15, 9, 68], [179, 71, 125])],
	"bands": [([98, 106, 0], [120, 182, 100])],
	"score_table": [([94, 10, 22], [107, 255, 145])],
	"all_red": [([168, 124, 12], [179, 255, 255])],
	"all_yellow": [([25, 65, 101], [32, 255, 255])],
	"button": [([167, 209, 130], [179, 255, 218])],
	"house": [([78, 98, 94], [96, 255, 134])]
}

# Helper function used to find HSV boundaries for each color of interest.
def find_hsv(image):
	def nothing(x):
		pass

	image = image.copy()

	cv2.namedWindow('HSV')

	# Creating track bar
	cv2.createTrackbar('h_min', 'HSV', 0, 179, nothing)
	cv2.createTrackbar('h_max', 'HSV', 0, 179, nothing)
	cv2.createTrackbar('s_min', 'HSV', 0, 255, nothing)
	cv2.createTrackbar('s_max', 'HSV', 0, 255, nothing)
	cv2.createTrackbar('v_min', 'HSV', 0, 255, nothing)
	cv2.createTrackbar('v_max', 'HSV', 0, 255, nothing)
	
	while True:
	    frame = image.copy()

	    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	    h_min = cv2.getTrackbarPos('h_min', 'HSV')
	    h_max = cv2.getTrackbarPos('h_max', 'HSV')
	    s_min = cv2.getTrackbarPos('s_min', 'HSV')
	    s_max = cv2.getTrackbarPos('s_max', 'HSV')
	    v_min = cv2.getTrackbarPos('v_min', 'HSV')
	    v_max = cv2.getTrackbarPos('v_max', 'HSV')

	    print("[{}, {}, {}], [{}, {}, {}]".format(h_min, s_min, v_min, h_max, s_max, v_max))

	    lower = np.array([h_min, s_min, v_min])
	    upper = np.array([h_max, s_max, v_max])

	    mask = cv2.inRange(hsv, lower, upper)

	    result = cv2.bitwise_and(frame, frame, mask=mask)

	    cv2.imshow('HSV MAP',result)

	    k = cv2.waitKey(5) & 0xFF
	    if k == 27:
	        break

# Computes masks from an image based on HSV boundaries for multiple colors. Several mechanisms to 
# remove noise are used: erosion, dilation and other functions such as get_score_table or remove_small_areas.
def get_color_masks(image, types_list=None, verbose=0):
	res = {}
	image = image.copy()
	hsv = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)

	colors = [key for key in BOUNDARIES]
	if types_list is not None:
		colors = [color for color in colors if color in types_list]

	for color in colors:	
		masks_list = []
		for (lower, upper) in BOUNDARIES[color]:
			lower = np.array(lower)
			upper = np.array(upper)

			mask = cv2.inRange(hsv.copy(), lower, upper)
			masks_list.append(mask)
			
		final_mask = masks_list[0]
		for i in range(1, len(masks_list)):
			final_mask = final_mask | masks_list[i]

		if color in ("score_table"):
			_, final_mask = get_score_table(image=image, mask=final_mask)
		elif color in ("bands"):
			final_mask = cv2.erode(final_mask, np.ones((7, 7)), iterations=2)
			final_mask = remove_small_areas(mask=final_mask, threshold=1000, kernel_close=7, iterations=2)
			final_mask = cv2.dilate(final_mask, np.ones((1, 5)), iterations=2)
			final_mask = cv2.dilate(final_mask, np.ones((3, 1)), iterations=30)
			final_mask = remove_small_areas(mask=final_mask, threshold=1000, kernel_close=7, iterations=2)
		elif color in ("button", "house"):
			final_mask = cv2.erode(final_mask, np.ones((5, 5)), iterations=1)
			final_mask = remove_small_areas(mask=final_mask, threshold=400, kernel_close=7, iterations=2)
			final_mask = cv2.dilate(final_mask, np.ones((5, 5)), iterations=1)
			final_mask = remove_small_areas(mask=final_mask, threshold=400, kernel_close=7, iterations=2)

		image_aux = cv2.bitwise_and(image.copy(), image.copy(), mask=final_mask)
		
		res[color] = (image_aux, final_mask)
		
	if verbose:
		show_images(images_list=[image] + [res[key][0] for key in res], nrows=3, ncols=3)
	return res

# Find how many points (1 or 255) from the mask overlapped with a given circle.
def find_circle_overlapping(circle, mask):
	mask = mask.copy()
	(center, radius) = circle
	rr, cc = skimage.draw.circle(r=center[1], c=center[0], radius=radius, shape=mask.shape[:2])
	
	total_matchings = sum([1 for (x, y) in zip(rr, cc) if mask[x][y] > 0])

	return (total_matchings, int(math.pi * radius * radius))

# Find how many points (1 or 255) from the mask overlapped with a given rectangle.
def find_rectangle_overlapping(rectangle, mask):
	mask = mask.copy()
	(top_left, bottom_right) = rectangle
	indeces = []
	for i in range(top_left[0], bottom_right[0] + 1):
		for j in range(top_left[1], bottom_right[1] + 1):
			indeces.append((i, j))
	
	total_matchings = sum([1 for (x, y) in indeces if mask[y][x] > 0])

	return (total_matchings, len(indeces))

# Use HoughCircle to find multiple circles. Then, an heuristics based on the 
# percentage of red/yellow points inside the circles is further used to remove FPs.
def get_hough_circles(image, min_radius, max_radius, minDist, dp, param1, param2, verbose):
	image = image.copy()
	image_all_circles, image_filtered_circles = image.copy(), image.copy()

	colors_tracked = get_color_masks(image=image, types_list=["yellow_stone", "red_stone", "gray"], verbose=verbose)
	
	mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 
							   minDist=minDist, dp=dp,
							   param1=param1, param2=param2,
							   minRadius=min_radius, maxRadius=max_radius)
	
	circles_list = []
	red_circles, yellow_circles = [], []
	if circles is not None:
		circles = np.uint16(np.around(circles))
		for i in circles[0, :]:
			center, radius = (int(i[0]), int(i[1])), int(i[2])
			radius_smaller = int(radius * 0.8) 
			circle = (center, radius_smaller)
			
			(a_yellow, b_yellow) = find_circle_overlapping(circle=circle, mask=colors_tracked["yellow_stone"][1])
			(a_red, b_red) = find_circle_overlapping(circle=circle, mask=colors_tracked["red_stone"][1])
			(a_gray, b_gray) = find_circle_overlapping(circle=circle, mask=colors_tracked["gray"][1])
			p_red, p_yellow, p_gray = a_red / b_red, a_yellow / b_yellow, a_gray / b_gray

			# if p_red >= 4% (we used a more aggresive HSV boundary for red_stone, so the percentage is lower)
			if p_red >= 0.04:
				s = "[Accepted red] radius: {}, p_red:{}, p_yellow:{}, p_gray:{}".format(radius, p_red, p_yellow, p_gray)
				red_circles.append((center, radius))
				cv2.circle(image_filtered_circles, center, radius, (255, 0, 255), 3)
				circles_list.append((center, radius))
			# if p_red >= 40%
			elif p_yellow >= 0.4: 
				s = "[Accepted yellow] radius: {}, p_red:{}, p_yellow:{}, p_gray:{}".format(radius, p_red, p_yellow, p_gray)
				yellow_circles.append((center, radius))
				cv2.circle(image_filtered_circles, center, radius, (255, 0, 255), 3)
				circles_list.append((center, radius))
			else:
				s = "[Discarded] radius: {}, p_red:{}, p_yellow:{}, p_gray:{}".format(radius, p_red, p_yellow, p_gray)
			
			cv2.circle(image_all_circles, center, radius, (255, 0, 255), 3)

			if verbose:
				print(s)
			
	return image_all_circles, image_filtered_circles, {"red_circles": red_circles, "yellow_circles": yellow_circles, "all_circles": red_circles + yellow_circles}

# Computes a histogram based on the possible colors from a matrix/image.
def get_histogram(matrix):
	histogram = {}
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[1]):
			histogram[matrix[i][j]] = histogram.get(matrix[i][j], 0) + 1
	return histogram

# Computes a mask for the scoring table (we needed it because there were red/yellow
# stones there that could be taken as FPs by my algorithm using HoughCircles).
# We will match on the contour color (green-ish), using a closing operation and then find the 
# largest component (we assume that if such scoring table exists, it will be the largest).
def get_score_table(image, mask):
	image, mask = image.copy(), mask.copy()

	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=5)
	
	# find largest component	
	max_area = 0
	seed_points = {}
	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):
			if mask[i][j] == 1 or mask[i][j] == 255:
				(area, _, _, _) = cv2.floodFill(mask, None, seedPoint=(j, i), newVal=128)
				seed_points[(j, i)] = area
				if area > max_area:
					seedPoint = (j, i)
					max_area = area
	
	# modify the largest component to 255 (here we use an heuristic based on max_area > 5000 
	# (since there are images that doesnt contain any scoring tables, so this function should return a 0-matrix).
	# Also, since the scoring table is always situated almsot at the top left corner, we force the seedPoint to be
	# in the top-left quadrant of the image. 
	if max_area > 5000 and seedPoint[0] < mask.shape[1] / 2 and seedPoint[1] < mask.shape[0] / 2:
		cv2.floodFill(mask, None, seedPoint=seedPoint, newVal=255)

	# make all the non-255 values 0.
	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):
			if mask[i][j] != 255:
				mask[i][j] = 0

	final_image = cv2.bitwise_and(image.copy(), image.copy(), mask=mask)

	return (np.uint8(final_image), mask)

# Used to cover the small holes in the image(noise).
# The idea is to use a given threshold for the accepted areas and filter based on that.
def remove_small_areas(mask, threshold, kernel_close, iterations):
	mask = mask.copy()

	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((kernel_close, kernel_close), np.uint8), iterations=iterations)
	
	# group by components and store the area for each of them.
	seed_points = {}
	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):
			if mask[i][j] == 1 or mask[i][j] == 255:
				(area, _, _, _) = cv2.floodFill(mask, None, seedPoint=(j, i), newVal=128)
				seed_points[(j, i)] = area

	# for each component, if the are is greater than the threshold, make it 255 again
	for point in seed_points:
		if seed_points[point] > threshold: 
			cv2.floodFill(mask, None, seedPoint=point, newVal=255)

	# make all the non-255 values 0.
	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):
			if mask[i][j] != 255:
				mask[i][j] = 0

	return mask

# Used to get only the center map, without the scoring table (some photos contain 3 ice surfaces,
# so we want our algorithm to be limited only to the relevant ice-surface and remove the non relevant ones).
# The idea is to combine the masks from the scoring_table and bands, to invert the last matrix and 
# to compute the largest component. This should be always the center map.
def get_map(image, verbose):
	image = image.copy()
	colors_tracked = get_color_masks(image=image, types_list=["score_table", "bands"], verbose=verbose)
	mask = colors_tracked["score_table"][1] | colors_tracked["bands"][1]

	inverted_mask = cv2.bitwise_not(mask)
	
	# compute each component, while tracking the max one.
	max_area = 0
	seed_points = {}
	for i in range(inverted_mask.shape[0]):
		for j in range(inverted_mask.shape[1]):
			if inverted_mask[i][j] == 1 or inverted_mask[i][j] == 255:
				(area, _, _, _) = cv2.floodFill(inverted_mask, None, seedPoint=(j, i), newVal=128)
				seed_points[(j, i)] = area
				if area > max_area:
					seedPoint = (j, i)
					max_area = area

	# restore the largest component
	cv2.floodFill(inverted_mask, None, seedPoint=seedPoint, newVal=255)

	# make all the non-255 values 0.
	for i in range(inverted_mask.shape[0]):
		for j in range(inverted_mask.shape[1]):
			if inverted_mask[i][j] != 255:
				inverted_mask[i][j] = 0

	final_image = cv2.bitwise_and(image.copy(), image.copy(), mask=inverted_mask)

	return np.uint8(final_image)

# Helper that shows multiple images in a grid format (specified by nrows and ncols)
# It supports both black and white / BGR.
def show_images(images_list, nrows=2, ncols=3):
	for i, image in enumerate(images_list):
		plt.subplot(nrows, ncols, i + 1)
		if len(image.shape) == 3:
			showed_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
			plt.imshow(showed_image)	
		else:
			showed_image = image.copy()
			plt.imshow(showed_image, cmap='gray', vmin=0, vmax=255)
	plt.show()
