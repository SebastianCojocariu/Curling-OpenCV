import utils
from utils import *

# Arguments available
def parse_args():
	parser = argparse.ArgumentParser(description='Task3')
	parser.add_argument('--video_path', type=str, default=None,
						 help='Path to a video on which to apply Task3 (absolute or relative path)')
	parser.add_argument('--initial_file_path', type=str, default=None,
						help='Path to the initial configuration of the video on which to apply Task3 (absolute or relative path)')
	parser.add_argument('--save_path', type=str, default=None,
						help='Path where to save the output of the algorithm (absolute or relative path)')
	parser.add_argument('--path_dir', type=str, default="./dataset/Task3/",
						help='Path to the directory where we want to apply Task3 (absolute or relative path)')
	parser.add_argument('--save_dir', type=str, default='./dataset/predictions/Task3/',
						help='Path where to save the directory where we want to apply Task3 (absolute or relative path)')
	parser.add_argument('--no_file', type=str, default=None,
						help='Apply the algorithm on the video specified by this number, that is located on path_dir. The output is saved on save_dir location')
	parser.add_argument('--verbose', type=str, default='0',
						help='Print intermediate output from the algorithm. Choose 0/1')
	args = parser.parse_args()
	return args

# It reduces the bounding box calculated by the CSRT tracker. It tries to shrink the rectangle using red_stone/yellow_stone masks.
# Before reducing the rectangle, we remove small areas of color from the mask (to remove noise). Additionally, it expands the calculated 
# rectangle within an epsilon_width/epsilon_height range (to account for the gray contour for instance).
def reduce_bounding_box(bbox, image, stone_color, verbose):
	def get_y_bottom(bbox, mask):
		(x_top, y_top, x_bottom, y_bottom) = bbox
		diff = y_bottom - (y_bottom - y_top) // 3
		j = y_bottom
		for j in range(y_bottom, diff - 1, -1):
			for i in range(x_top, x_bottom + 1):
				if mask[j][i] > 0:
					return j
		return j

	def get_y_top(bbox, mask):
		(x_top, y_top, x_bottom, y_bottom) = bbox
		diff = y_top + (y_bottom - y_top) // 3
		j = y_top
		for j in range(y_top, diff):
			for i in range(x_top, x_bottom + 1):
				if mask[j][i] > 0:
					return j
		return j
	
	def get_x_bottom(bbox, mask):
		(x_top, y_top, x_bottom, y_bottom) = bbox
		diff = x_bottom - (x_bottom - x_top) // 3
		i = x_bottom
		for i in range(x_bottom, diff - 1, -1):
			for j in range(y_top, y_bottom + 1):
				if mask[j][i] > 0:
					return i
		return i
	

	def get_x_top(bbox, mask):
		(x_top, y_top, x_bottom, y_bottom) = bbox
		diff = x_top + (x_bottom - x_top) // 3
		i = x_top
		for i in range(x_top, diff):
			for j in range(y_top, y_bottom + 1):
				if mask[j][i] > 0:
					return i
		return i
	
	image = image.copy()
	
	(x_top, y_top, width, height) = bbox
	x_bottom = x_top + width
	y_bottom = y_top + height

	colors_tracked = get_color_masks(image=image, types_list=[stone_color], verbose=0)
	mask = colors_tracked[stone_color][1]
	mask[y_top: y_bottom + 1, x_top: x_bottom + 1] = remove_small_areas(mask=mask[y_top: y_bottom + 1, x_top: x_bottom + 1], threshold=50, kernel_close=5, iterations=2)
	
	bbox = (x_top, y_top, x_bottom, y_bottom) 
	y_bottom_new = get_y_bottom(bbox=bbox, mask=mask)
	y_top_new = get_y_top(bbox=bbox, mask=mask)
	x_bottom_new = get_x_bottom(bbox=bbox, mask=mask)
	x_top_new = get_x_top(bbox=bbox, mask=mask)

	epsilon_width = 0.15
	epsilon_height = 0.3
	
	width_new = x_bottom_new - x_top_new
	height_new = y_bottom_new - y_top_new

	x_bottom_new = min(x_bottom, int(x_bottom_new + (width - width_new) * epsilon_width))
	y_bottom_new = min(y_bottom, int(y_bottom_new + (height - height_new) * epsilon_height))
	x_top_new = max(x_top, int(x_top_new - (width - width_new)  * epsilon_width))
	y_top_new = max(y_top, int(y_top_new - (height - height_new) * epsilon_height))

	if verbose:
		cv2.rectangle(image, (x_top_new, y_top_new), (x_bottom_new, y_bottom_new), 255, 3)
		cv2.imshow("Frame", image)
		#cv2.rectangle(mask, (x_top_new, y_top_new), (x_bottom_new, y_bottom_new), 255, 3)
		#cv2.imshow("Frame", mask)
		cv2.waitKey(1) & 0xFF

	return [str(x_top_new), str(y_top_new), str(x_bottom_new), str(y_bottom_new)]

# Computes the logic behind task3. it takes a video, instantiates a CSRT tracker on the 
# initial rectangle received and for each frame it does the following:
# - uses CSRT tracker to compute the current bounding box.
# - applies reduce_bounding_box to further reduce the bbox.
# Finally, it saves the result in the specified file.
def task3(video_path, initial_file_path, save_path=None, verbose=0):
	tracker = cv2.TrackerCSRT_create()
	
	result = []
	with open(initial_file_path, "r") as f:
		lines = f.read().splitlines()
		lines = [[int(number) for number in line.split() if number != ""] for line in lines]
		result.append(" ".join([str(x) for x in lines[0]]))
		
		x_top, y_top, x_bottom, y_bottom = lines[1][1:]
		width = x_bottom - x_top
		height = y_bottom - y_top 
		initBB = (x_top, y_top, width, height)
		
		fps = FPS().start()

	video = cv2.VideoCapture(video_path)

	curr_frame_idx = 0
	while True:
		frame = video.read()[1]

		if frame is None:
			break

		if curr_frame_idx == 0:
			tracker.init(frame, initBB)

			# find the color or the stone
			colors_tracked = utils.get_color_masks(image=frame, types_list=["all_yellow", "all_red"], verbose=0)
			a_red, _ = find_rectangle_overlapping(rectangle=((x_top, y_top), (x_bottom, y_bottom)), mask=colors_tracked["all_red"][1])
			a_yellow, _ = find_rectangle_overlapping(rectangle=((x_top, y_top), (x_bottom, y_bottom)), mask=colors_tracked["all_yellow"][1])
	
			if a_red > a_yellow:
				stone_color = "red_stone"
			else:
				stone_color = "yellow_stone"

		(success, bbox) = tracker.update(frame)
		
		fps.update()
		fps.stop()

		reduced_bbox = reduce_bounding_box(bbox=bbox, image=frame, stone_color=stone_color, verbose=verbose)
		result.append(" ".join([str(curr_frame_idx)] + reduced_bbox))

		curr_frame_idx += 1

	string_to_write_in_file = "\n".join(result)

	if save_path != None and save_path != "":
		with open(save_path, "w+") as f:
			f.write(string_to_write_in_file)
		print("The output was saved at location: {}!".format(save_path))

if __name__ == "__main__":
	args = parse_args()
	verbose = ord(args.verbose) - ord('0')

	if args.video_path != None and args.initial_file_path != None:
		try:
			task3(video_path=args.video_path,
				  initial_file_path=args.initial_file_path,	
				  verbose=verbose, 
				  save_path=args.save_path)
		except:
			raise Exception("An exception occured during the execution of Task3!")

	else:
		os.makedirs(args.save_dir, exist_ok=True)

		if args.no_file != None:
			try:
				video_path = os.path.join(args.path_dir, "{}.mp4".format(args.no_file))
				initial_file_path = os.path.join(args.path_dir, "{}.txt".format(args.no_file))
				save_path = os.path.join(args.save_dir, "{}_predicted.txt".format(args.no_file))
				print("Processing the video located at: {} with initial file located at: {}".format(video_path, initial_file_path))
				task3(video_path=video_path,
					  initial_file_path=initial_file_path,	
				 	  verbose=verbose, 
				 	  save_path=save_path)
			except:
				raise Exception("An exception occured during the execution of Task3 for the video located at: {} with initial config file at: {} !".format(video_path, initial_file_path))
		
		else:
			for no_file in range(1, 6):
				try:
					video_path = os.path.join(args.path_dir, "{}.mp4".format(no_file))
					initial_file_path = os.path.join(args.path_dir, "{}.txt".format(no_file))
					save_path = os.path.join(args.save_dir, "{}_predicted.txt".format(no_file))
					print("Processing the video located at: {} with initial file located at: {}".format(video_path, initial_file_path))
					task3(video_path=video_path,
						  initial_file_path=initial_file_path,	
				 		  verbose=verbose, 
				 		  save_path=save_path)
				except:
					raise Exception("An exception occured during the execution of Task3 for the video located at: {} with initial config file at: {} !".format(video_path, initial_file_path))