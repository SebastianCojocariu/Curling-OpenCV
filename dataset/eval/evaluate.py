import numpy as np

def evaluate_results_task1(predictions_path,ground_truth_path,verbose = 0):
    total_correct_number_stones_all = 0
    total_correct_number_red_and_yellow = 0

    for i in range(1, 26):
        try:
            filename_predictions = predictions_path + "/" + str(i) + "_predicted.txt"
            p = open(filename_predictions,"rt")        
            filename_ground_truth = ground_truth_path + "/" + str(i) + "_gt.txt"
            gt = open(filename_ground_truth,"rt")
            
            correct_number_stones_all = 1        
            #read the first line - number of total stones
            p_line = p.readline()
            gt_line = gt.readline()
            if (p_line[:1] != gt_line[:1]):
                    correct_number_stones_all = 0                
            
            correct_number_red_and_yellow = 1
            #read the second and third lines - number of red and yellow stones        
            p_line = p.readline()
            gt_line = gt.readline()
            if (p_line[:1] != gt_line[:1]):
                    correct_number_red_and_yellow = 0
            p_line = p.readline()
            gt_line = gt.readline()
            if (p_line[:1] != gt_line[:1]):
                    correct_number_red_and_yellow = 0                
            p.close()
            gt.close()
            
            if verbose:
                print("Task 1 -Counting stones + their color: for test example number ", str(i), " the prediction is :", (1-correct_number_stones_all) * "in" + "correct for number of total stones and ",(1-correct_number_red_and_yellow) * "in" + "correct for number of red and yellow stones" + "\n")
                   
            total_correct_number_stones_all = total_correct_number_stones_all + correct_number_stones_all
            total_correct_number_red_and_yellow = total_correct_number_red_and_yellow + correct_number_red_and_yellow    
        except:
            pass

    points = total_correct_number_stones_all * 0.03 + total_correct_number_red_and_yellow * 0.03

    return total_correct_number_stones_all, total_correct_number_red_and_yellow, points

def evaluate_results_task2(predictions_path,ground_truth_path,verbose = 0):
    total_correct_scores = 0
    
    for i in range(1, 11):
        try:
            filename_predictions = predictions_path + "/" + str(i) + "_predicted.txt"
            p = open(filename_predictions,"rt")        
            filename_ground_truth = ground_truth_path + "/" + str(i) + "_gt.txt"
            gt = open(filename_ground_truth,"rt")
            correct_scores = 1
            
            #read the first and second lines - giving the score
            p_line = p.readline()
            gt_line = gt.readline()
            if (p_line[:1] != gt_line[:1]):
                    correct_scores = 0
            p_line = p.readline()
            gt_line = gt.readline()
            if (p_line[:1] != gt_line[:1]):
                    correct_scores = 0                
            p.close()
            gt.close()
            
            
            if verbose:
                print("Task 2 -Assessing correct score: for test example number ", str(i), " the prediction is :", (1-correct_scores) * "in" + "correct" + "\n")
                   
            total_correct_scores = total_correct_scores + + correct_scores                
        except:
            pass
    
    points = total_correct_scores * 0.1

    return total_correct_scores, points


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def compute_percentage_tracking(gt_bboxes, predicted_bboxes, num_frames):
    """
    This function compute the percentage of detected bounding boxes based on the ground-truth bboxes and the predicted ones.
    :param gt_bboxes. The ground-truth bboxes with the format: frame_idx, x_min, y_min, x_max, y_max.
    :param predicted_bboxes. The predicted bboxes with the format: frame_idx, x_min, y_min, x_max, y_max
    :param num_frames. The total number of frames in the video.
    """
    
    num_frames = int(num_frames)
    
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    gt_dict = {}
    for gt_box in gt_bboxes:
        gt_dict[gt_box[0]] = gt_box[1:]
    
    pred_dict = {}
    for pred_bbox in predicted_bboxes:
        pred_dict[pred_bbox[0]] = pred_bbox[1:]
        
    for i in range(num_frames):
        if gt_dict.get(i, None) is None and pred_dict.get(i, None) is None: # the stone is not on the ice surface
            tn += 1 
        
        elif gt_dict.get(i, None) is not None and pred_dict.get(i, None) is None: # the stone is not detected
            fn += 1
            
        elif gt_dict.get(i, None) is None and pred_dict.get(i, None) is not None: # the stone is not on the ice surface, but it is 'detected'
            fp += 1
            
        elif gt_dict.get(i, None) is not None and pred_dict.get(i, None) is not None: # the stone is on the ice surface and it is detected
            
            iou = bb_intersection_over_union(gt_dict[i], pred_dict[i])
            if iou >= 0.2:
                tp += 1
            else:
                fp += 1 
                         
    print(f'tp = {tp}, tn = {tn}, fp = {fp},fn = {fn}')
    assert tn + fn + tp + fp == num_frames
    perc = (tp + tn) / (tp + fp + tn + fn)
    
    return perc

def evaluate_results_task3(predictions_path,ground_truth_path, verbose = 0):
    total_correct_tracked_videos = 0
    for i in range(1, 6):
        try:
            filename_predictions = predictions_path + "/" + str(i) + "_predicted.txt"        
            filename_ground_truth = ground_truth_path + "/" + str(i) + "_gt.txt"
            
            p = np.loadtxt(filename_predictions)
            num_frames = p[0][0]
            predicted_bboxes = p[1:]
            
            gt = np.loadtxt(filename_ground_truth)
            gt_bboxes = gt[1:]
            
            percentage = compute_percentage_tracking(gt_bboxes, predicted_bboxes, num_frames)
            
            correct_tracked_videos = 1
            if percentage < 0.8:
                 correct_tracked_videos = 0
            
            print("percentage = ", percentage)
            if verbose:
                print("Task 3 - Tracking a stone in constrained scenario: for test example number ", str(i), " the prediction is :", (1-correct_tracked_videos) * "in" + "correct", "\n")
            
            total_correct_tracked_videos = total_correct_tracked_videos + correct_tracked_videos
        except:
            pass

    points = total_correct_tracked_videos * 0.1
        
    return total_correct_tracked_videos,points 


#change this on your machine
predictions_path_root = "../predictions/"
ground_truth_path_root = "../"

#task1
verbose = 1
predictions_path = predictions_path_root + "Task1/"
ground_truth_path = ground_truth_path_root + "Task1/ground-truth/"
total_correct_number_stones_all, total_correct_number_red_and_yellow, points_task1 = evaluate_results_task1(predictions_path,ground_truth_path,verbose)

#task2
verbose = 1
predictions_path = predictions_path_root + "Task2/"
ground_truth_path = ground_truth_path_root + "Task2/ground-truth/"
total_correct_scores,points_task2 = evaluate_results_task2(predictions_path,ground_truth_path,verbose)


#task3
verbose = 1
predictions_path = predictions_path_root + "Task3/"
ground_truth_path = ground_truth_path_root + "Task3/ground-truth/"
total_correct_tracked_videos_task3,points_task3 = evaluate_results_task3(predictions_path,ground_truth_path,verbose)


print("Task 1 = ", points_task1, "\nTask 2 = ",points_task2, "\nTask 3 = ", points_task3, "\n")
