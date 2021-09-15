import utils
from utils import *

# Arguments available
def parse_args():
	parser = argparse.ArgumentParser(description='Task2')
	parser.add_argument('--video_path', type=str, default=None,
	                    help='Path to a video on which to apply Task2 (absolute or relative path)')
	parser.add_argument('--save_path', type=str, default=None,
	                    help='Path where to save the output of the algorithm (absolute or relative path)')
	parser.add_argument('--path_dir', type=str, default="./dataset/Task2/",
	                    help='Path to the directory where we want to apply Task2 (absolute or relative path)')
	parser.add_argument('--save_dir', type=str, default='./dataset/predictions/Task2/',
	                    help='Path where to save the directory where we want to apply Task2 (absolute or relative path)')
	parser.add_argument('--no_file', type=str, default=None,
	                    help='Apply the algorithm on the video specified by this number, that is located on path_dir. The output is saved on save_dir location')
	parser.add_argument('--verbose', type=str, default='0',
	                    help='Print intermediate output from the algorithm. Choose 0/1')
	args = parser.parse_args()
	return args

# Computes the center of the button (inner-circle) + the radius to the house.
# It combines the masks related to the button and the house. To compute the center of the button,
# we first find the mean coordinates of all red points (bear in mind that at this point erosion 
# and remove_small_areas were already applied, so small holes should have been removes). Then, we 
# only keep the red points that are inside circle with the center calcualted before and radius=200 (hyperparameter).
# We then find the margin red points and compute all the distances between each 2 pairs. We then choose the largest 
# 100 and compute their mean on each coordinate. This would be saved as the center of the button.
# Then, to find the radius to the house, we calculate the distance from all the blue (the color of the outer circle) 
# points, remove the outliers and compute the mean of the largest 100 remaining. This would be the desired radius.
def find_map_details(image, verbose):
	image = image.copy()
	colors_tracked = get_color_masks(image=image, types_list=["button", "house"], verbose=verbose)

	mask_button = colors_tracked["button"][1]
	mask_house= colors_tracked["house"][1]
	
	red_points = {}
	for i in range(mask_button.shape[0]):
		for j in range(mask_button.shape[1]):
			if mask_button[i][j] == 1 or mask_button[i][j] == 255:
				red_points[(i, j)] = 1

	x_center = int(np.mean([i for (i, _) in red_points]))
	y_center = int(np.mean([j for (_, j) in red_points]))

	red_points = {(i, j): 1 for (i, j) in red_points if (i - x_center) ** 2 + (j - y_center) ** 2 <= 200 ** 2}

	x_center_mean = int(np.mean([i for (i, _) in red_points]))
	y_center_mean = int(np.mean([j for (_, j) in red_points]))
	
	margin_red_points = []
	for (i, j) in red_points:
		if not ((i + 1, j) in red_points and (i - 1, j) in red_points and (i, j - 1) in red_points and (i, j + 1) in red_points):
			margin_red_points.append((i, j))	
	
	red_distances = []
	for i in range(len(margin_red_points)):
		for j in range(i + 1, len(margin_red_points)):
			a = np.asarray(margin_red_points[i])
			b = np.asarray(margin_red_points[j])
			dist = np.linalg.norm(a - b)
			middle_point = ((a[0] + b[0])/2, (a[1] + b[1])/2)
			red_distances.append((dist, middle_point))

	red_distances.sort(key=lambda x: x[0], reverse=True)
	x_center_heuristics = int(np.mean([x for (_, (x, _)) in red_distances[:100]]))
	y_center_heuristics = int(np.mean([y for (_, (_, y)) in red_distances[:100]]))
	
	center_button = (y_center_heuristics, x_center_heuristics)
	#center_button = ((y_center_mean + y_center_heuristics) // 2, (x_center_mean + x_center_heuristics) // 2)
	
	radius_list = []
	for i in range(mask_house.shape[0]):
		for j in range(mask_house.shape[1]):
			if mask_house[i][j] == 1 or mask_house[i][j] == 255:
				radius = np.linalg.norm(np.asarray((j, i)) - np.asarray(center_button))
				radius_list.append(radius)
	
	mean_radius = np.mean(radius_list)
	std_radius = np.std(radius_list)
	radius_list = [r for r in radius_list if r >= mean_radius - 3 * std_radius and r <= mean_radius + 3 * std_radius]
	radius_list.sort(reverse=True)

	radius_house = np.mean(radius_list[:500])
	radius_house = int(radius_house)
	
	return center_button, radius_house

# It groups all the circles corresponding to the stones. Then, it filters them, keeping only the ones that touch the house.
# Then, these stones are sorted based on their relative distance to the center. The score will be equal to how many consecutive stones
# of the same color as the closest one there are (starting from the closest one).
def algorithm(circles_dict, center_button, radius_house):
	scores = {"red": 0, "yellow": 0}
	all_circles = [(center, r, "yellow") for (center, r) in circles_dict["yellow_circles"]] + [(center, r, "red") for (center, r) in circles_dict["red_circles"]]
	
	epsilon = 5
	all_circles = [(np.linalg.norm(np.asarray(center) - np.asarray(center_button)), r, color) for (center, r, color) in all_circles]
	all_circles = [(distance, color) for (distance, r, color) in all_circles if distance <= radius_house + r - epsilon]
	all_circles.sort(key=lambda x: x[0])

	i = 0
	while(i < len(all_circles) and all_circles[i][1] == all_circles[0][1]):
		scores[all_circles[0][1]] += 1
		i += 1

	print("all_circles in house: ", all_circles)
	print("radius_house: ", radius_house)
	print("scores: ", scores)

	return scores

# The logic behind task2. It takes a video, choose the last frame and does the following:
# - applies get_map on the last frame (to get only the relevant ice-surface and remove the scoring table).
# - calls find_map_details to find the center of the button and the radius to the house.
# - applies get_hough_circles to detect the circles that correspond to the stones.
# - calls algorithm to compute the final score.
# Finally, it saves the result in the specified file.
def task2(video_path, save_path=None, verbose=0):
	vs = cv2.VideoCapture(video_path)
	fps = FPS().start()
	initialize = True
	
	curr_frame = vs.read()[1]
	while True:
		next_frame = vs.read()[1]
		if next_frame is None:
			break
		curr_frame = next_frame

	image = get_map(image=curr_frame, verbose=verbose)
	center_button, radius_house = find_map_details(image=image, verbose=verbose)
	
	if verbose:
		image_with_button = image.copy()
		cv2.circle(image_with_button, center_button, 10, (255, 0, 255), 3)
		cv2.circle(image_with_button, center_button, radius_house, (255, 0, 255), 3)
		utils.show_images([image, image_with_button], nrows=1, ncols=2)

	_, image_filtered_circles, circles_dict = get_hough_circles(image=image, min_radius=10, max_radius=25, minDist=30, dp=1, param1=150, param2=15,verbose=verbose)
	scores = algorithm(circles_dict=circles_dict, center_button=center_button, radius_house=radius_house)
	
	string_to_write_in_file = "\n".join([str(scores["red"]), str(scores["yellow"])])

	if save_path != None and save_path != "":
		with open(save_path, "w+") as f:
			f.write(string_to_write_in_file)
		print("The output was saved at location: {}!".format(save_path))
		print(string_to_write_in_file)

		#cv2.circle(image_filtered_circles, center_button, 10, (255, 0, 255), 3)
		#cv2.circle(image_filtered_circles, center_button, radius_house, (255, 0, 255), 3)
		#image_path = save_path.replace(".txt", ".png")
		#cv2.imwrite(image_path, image_filtered_circles)
	
	return scores
	
if __name__ == "__main__":
	args = parse_args()
	verbose = ord(args.verbose) - ord('0')

	if args.video_path != None:
		try:
			task2(video_path=args.video_path, 
				  save_path=args.save_path,
				  verbose=verbose)
		except:
			raise Exception("An exception occured during the execution of Task2!")

	else:
		os.makedirs(args.save_dir, exist_ok=True)

		if args.no_file != None:
			try:
				video_path = os.path.join(args.path_dir, "{}.mp4".format(args.no_file))
				save_path = os.path.join(args.save_dir, "{}_predicted.txt".format(args.no_file))
				print("Processing the video located at: {}".format(video_path))
				task2(video_path=video_path,
				 	  verbose=verbose, 
				 	  save_path=save_path)
			except:
				raise Exception("An exception occured during the execution of Task2 for the video located at: {}!".format(video_path))
		
		else:
			for no_file in range(1, 11):
				try:
					video_path = os.path.join(args.path_dir, "{}.mp4".format(no_file))
					save_path = os.path.join(args.save_dir, "{}_predicted.txt".format(no_file))
					print("Processing the video located at: {}".format(video_path))
					task2(video_path=video_path,
				 	  	  verbose=verbose, 
				 	      save_path=save_path)
				except:
					raise Exception("An exception occured during the execution of Task2 for the video located at: {}!".format(video_path))