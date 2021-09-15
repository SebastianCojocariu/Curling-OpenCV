## REQUIREMENTS

opencv-contrib-python==4.5.2.54
numpy==1.18.5
matplotlib=3.2.2
sortedcollections==1.2.1
more-itertools==8.4.0
mpmath==1.1.0
argparse=1.1
skimage=0.16.2
imutils=0.5.4


## TASK1
Python script: task_1.py
Function of interest: task1(image_path, save_path=None, verbose=0)
Supports the following command line arguments:
--image_path: path to an image we want to apply the algorithm. Default is None
--save_path: path where to save the output. Default is None
--path_dir: path from where to extract the files to run the algorithm on. Default is ./dataset/Task1/
--save_dir: path where to save the output for the files from path_dir. Default is ./dataset/predictions/Task1/
--no_file: the number of file from path_dir where we want to apply our algorithm. It saves the output on save_dir (it must be specified). Default is None
--verbose: a flag to show different data along the running of the algorithm. Default is 0

To run all the tests, just run: python3 task_1.py (This will get all the 25 images from ./dataset/Task1/and save the outputs on ./dataset/predictions/Task1/)
To run a specific image (specified by path), just run: python3 task_1.py --image_path <path_to_the_image> --verbose <0/1>
To run a specific image (specified by number from path_dir), just run: python3 task_1.py --no_file <the number of file> --verbose <0/1>

## TASK2
Python script: task_2.py
Function of interest: task2(video_path, save_path=None, verbose=0)
Supports the following command line arguments:
--video_path: path to the video we want to apply the algorithm. Default is None
--save_path: path where to save the output. Default is None
--path_dir: path from where to extract the files to run the algorithm on. Default is ./dataset/Task2/
--save_dir: path where to save the output for the files from path_dir. Default is ./dataset/predictions/Task2/
--no_file: the number of file from path_dir where we want to apply our algorithm. It saves the output on save_dir (it must be specified). Default is None
--verbose: a flag to show different data along the running of the algorithm. Default is 0

To run all the tests, just run: python3 task_2.py (This will get all the 10 videos from ./dataset/Task2/ and save the outputs on ./dataset/predictions/Task2/)
To run a specific image (specified by path), just run: python3 task_2.py --video_path <path_to_the_video> --verbose <0/1>
To run a specific image (specified by number from path_dir), just run: python3 task_2.py --no_file <the number of file>  --verbose <0/1>

## TASK3
Python script: task_3.py
Function of interest: task3(video_path, initial_file_path, save_path=None, verbose=0)
Supports the following command line arguments:
--video_path: path to the video we want to apply the algorithm. Default is None
--initial_file_path: path to the initial configuration of the video on which to apply Task3 (absolute or relative path). Default is None.
--save_path: path where to save the output. Default is None
--path_dir: path from where to extract the files to run the algorithm on. Default is ./dataset/Task3/
--save_dir: path where to save the output for the files from path_dir. Default is ./dataset/predictions/Task3/
--no_file: the number of file from path_dir where we want to apply our algorithm. It saves the output on save_dir (it must be specified). Default is None
--verbose: a flag to show different data along the running of the algorithm. Default is 0

To run all the tests, just run: python3 task_3.py (This will get all the 5 videos from ./dataset/Task3/ and save the outputs on ./dataset/predictions/Task3/)
To run a specific image (specified by path), just run: python3 task_3.py --video_path <path_to_the_image> --initial_file <path_to_initial_file> --verbose <0/1>
To run a specific image (specified by number from path_dir), just run: python3 task_3.py --no_file <the number of file> --verbose <0/1>

