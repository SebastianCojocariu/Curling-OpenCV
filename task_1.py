import utils
from utils import *

# Arguments available
def parse_args():
	parser = argparse.ArgumentParser(description='Task1')
	parser.add_argument('--image_path', type=str, default=None,
	                    help='Path to an image on which to apply Task1 (absolute or relative path)')
	parser.add_argument('--save_path', type=str, default=None,
	                    help='Path where to save the output of the algorithm (absolute or relative path)')
	parser.add_argument('--path_dir', type=str, default="./dataset/Task1/",
	                    help='Path to the directory where we want to apply Task1 (absolute or relative path)')
	parser.add_argument('--save_dir', type=str, default='./dataset/predictions/Task1/',
	                    help='Path where to save the directory where we want to apply Task1 (absolute or relative path)')
	parser.add_argument('--no_file', type=str, default=None,
	                    help='Apply the algorithm on the image specified by this number, that is located on path_dir. The output is saved on save_dir location')
	parser.add_argument('--verbose', type=str, default='0',
	                    help='Print intermediate output from the algorithm. Choose 0/1')
	args = parser.parse_args()
	return args

# Computes the logic behind task1. 
# - first apply get_map to remove the scoring table and non-relevant ice-surfaces.
# - it finds and filters multiple circles extracted using houghCircles algorithm. 
# Finally, it saves the result in the specified file.
def task1(image_path, save_path=None, verbose=0):
	image = cv2.imread(image_path)
	image = get_map(image=image, verbose=verbose)
	image_all_circles, image_filtered_circles, circles_dict = get_hough_circles(image=image, min_radius=10, max_radius=25, minDist=30, dp=1, param1=150, param2=15,verbose=verbose)
	
	if verbose:
		utils.show_images([image, image_all_circles, image_filtered_circles], nrows=2, ncols=2)

	string_to_write_in_file = "\n".join([str(len(circles_dict ["all_circles"])), str(len(circles_dict ["red_circles"])), str(len(circles_dict ["yellow_circles"]))])

	if save_path != None and save_path != "":
		with open(save_path, "w+") as f:
			f.write(string_to_write_in_file)
		print("The output was saved at location: {}!".format(save_path))
		print(string_to_write_in_file)
		
		#image_path = save_path.replace(".txt", ".png")
		#cv2.imwrite(image_path, image_filtered_circles)

	return circles_dict
	
if __name__ == "__main__":
	args = parse_args()
	verbose = ord(args.verbose) - ord('0')

	if args.image_path != None:
		try:
			task1(image_path=args.image_path,
				  save_path=args.save_path,
				  verbose=verbose)
		except:
			raise Exception("An exception occured during the execution of Task1!")

	else:
		os.makedirs(args.save_dir, exist_ok=True)
		
		if args.no_file != None:
			try:
				image_path = os.path.join(args.path_dir, "{}.png".format(args.no_file))
				save_path = os.path.join(args.save_dir, "{}_predicted.txt".format(args.no_file))
				print("Processing the image located at: {}".format(image_path))
				task1(image_path=image_path,
				 	  verbose=verbose, 
				 	  save_path=save_path)
			except:
				raise Exception("An exception occured during the execution of Task1 for the image located at: {}!".format(image_path))
		
		else:
			for no_file in range(1, 26):
				try:
					image_path = os.path.join(args.path_dir, "{}.png".format(no_file))
					save_path = os.path.join(args.save_dir, "{}_predicted.txt".format(no_file))
					print("Processing the image located at: {}".format(image_path))
					task1(image_path=image_path,
				 	  	  verbose=verbose, 
				 	      save_path=save_path)
				except:
					raise Exception("An exception occured during the execution of Task1 for the image located at: {}!".format(image_path))
