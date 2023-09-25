# %% Import dependencies ----
import cv2 as cv
import numpy as np

# %% identify_contours_closest_neighbours() ----
def identify_contours_closest_neighbours(list_of_reference_contours, list_of_neighbours_contours):
    
    def find_cnt_centroid(cnt):
        moment = cv.moments(cnt)
        cx = int(moment['m10']/moment['m00'])
        cy = int(moment['m01']/moment['m00'])
        return np.array([cx, cy])
    
    def get_distance_pt1_pt2(pt1, pt2):
        return np.linalg.norm(pt1 - pt2, axis=0)
        
    def get_closest_neighbour_for_single_pt(single_pt, list_of_neighbours):
        distances = np.array(list(map(lambda x: get_distance_pt1_pt2(pt1=single_pt, pt2=x), list_of_neighbours)))
        min_distance_idx = distances.argsort()[0]
        min_distance_val = distances[min_distance_idx]
        out = {'closest_neighbour_idx':min_distance_idx,
               'distance_to_closest_neighbour':min_distance_val}
        return out
    
    def get_closest_neighbour_for_list_of_pts(list_of_pts, list_of_neighbours):
        out = []
        for idx, pt in enumerate(list_of_pts):
            res = get_closest_neighbour_for_single_pt(single_pt=pt, list_of_neighbours=list_of_neighbours)
            res['ref_idx'] = idx
            res = {k:res[k] for k in ['ref_idx', 'closest_neighbour_idx', 'distance_to_closest_neighbour']}
            out.append(res)
        return out    
    
    reference_centroids = list(map(lambda x: find_cnt_centroid(x), list_of_reference_contours))
    
    neighbour_centroids = list(map(lambda x: find_cnt_centroid(x), list_of_neighbours_contours))
    
    out = get_closest_neighbour_for_list_of_pts(list_of_pts=reference_centroids, 
                                                list_of_neighbours=neighbour_centroids)
    
    return out

# %% filter_contours_specific_closest_neighbours() ----
def filter_contours_specific_closest_neighbours(list_of_neighbours_data_dicts, dist_threshold):
    true_nei_data_dict_list = [i for i in list_of_neighbours_data_dicts if i['distance_to_closest_neighbour'] <= dist_threshold]
    true_nei_data_dict_list = sorted(true_nei_data_dict_list,
                                     key = lambda x: x['distance_to_closest_neighbour'])
    nei_idx_list = [i['closest_neighbour_idx'] for i in true_nei_data_dict_list]
    nei_idx_set = set(nei_idx_list)
    true_nei_data_dict_list_min_dist_idx = [nei_idx_list.index(i) for i in nei_idx_set]
    min_dist_true_nei_data_dict_list = [true_nei_data_dict_list[i] for i in true_nei_data_dict_list_min_dist_idx]
    min_dist_true_nei_data_dict_list = sorted(min_dist_true_nei_data_dict_list,
                                              key = lambda x: x['ref_idx'])
    return min_dist_true_nei_data_dict_list

# %% check_contours_closest_neighbours_overlap() ----
def check_contours_closest_neighbours_overlap(list_of_contours_to_be_overlapped, 
                                              list_of_contours_to_overlap_by,
                                              list_of_neighbours_data_dicts,
                                              overlap_neighbours_by_references=True,
                                              shape=(1080,1080)):
    
    def check_contours_overlap(contours_to_be_overlapped,
                               contours_to_overlap_by,
                               shape=shape):
        
        submask_to_be_overlapped = cv.cvtColor(src=cv.drawContours(image=cv.cvtColor(src=np.zeros(shape=shape, dtype=np.uint8), 
                                                                                     code=cv.COLOR_GRAY2RGB),
                                                                   contours=[contours_to_be_overlapped],
                                                                   contourIdx=-1,
                                                                   color=(255,255,255),
                                                                   thickness=-1), 
                                               code=cv.COLOR_RGB2GRAY)
        
        submask_to_overlap_by = cv.cvtColor(src=cv.drawContours(image=cv.cvtColor(src=np.zeros(shape=shape, dtype=np.uint8),  
                                                                                  code=cv.COLOR_GRAY2RGB),
                                                                contours=[contours_to_overlap_by], 
                                                                contourIdx=-1,
                                                                color=(255,255,255),
                                                                thickness=-1), 
                                            code=cv.COLOR_RGB2GRAY)
        
        submasks_overlap = np.where(submask_to_be_overlapped==255, submask_to_overlap_by, 0)
        
        contours_overlap =  cv.findContours(image=submasks_overlap,
                                            mode=cv.RETR_LIST, 
                                            method=cv.CHAIN_APPROX_NONE)[0]
        
        if not len(contours_overlap):
            contours_overlap_area = 0
        else:
            contours_overlap_area = cv.contourArea(contours_overlap[0][:,0,:])
            
        contours_to_be_overlapped_area = cv.contourArea(contours_to_be_overlapped)
        overlap_pct = (contours_overlap_area/contours_to_be_overlapped_area)*100
        
        return overlap_pct
    
    if overlap_neighbours_by_references:
        cnt_to_be_overlapped_idx = 'closest_neighbour_idx'
        cnt_to_overlap_by_idx = 'ref_idx'
    else:
        cnt_to_be_overlapped_idx = 'ref_idx'
        cnt_to_overlap_by_idx = 'closest_neighbour_idx'
        
    for neighbours_data_dict in list_of_neighbours_data_dicts:
        contours_to_be_overlapped = list_of_contours_to_be_overlapped[neighbours_data_dict[cnt_to_be_overlapped_idx]]
        contours_to_overlap_by = list_of_contours_to_overlap_by[neighbours_data_dict[cnt_to_overlap_by_idx]]
        overlap_pct = check_contours_overlap(contours_to_be_overlapped=contours_to_be_overlapped,
                                             contours_to_overlap_by=contours_to_overlap_by, 
                                             shape=shape)
        
        neighbours_data_dict['overlap_pct'] = overlap_pct
    
    return list_of_neighbours_data_dicts

# %% filter_overlapping_neighbours() ----
def filter_overlapping_neighbours(list_of_neighbours_data_dicts, overlap_threshold):
    output = [i for i in list_of_neighbours_data_dicts if i['overlap_pct'] >= overlap_threshold]
    return output

# %% identify_contours_to_rescue() ----
def identify_contours_to_rescue(list_of_reference_contours, list_of_neighbours_contours, list_of_neighbours_data_dicts):
    
    ref_idx = list(range(len(list_of_reference_contours)))
    selected_ref_idx = [i['ref_idx'] for i in list_of_neighbours_data_dicts]
    ref_to_rescue_idx = [i for i in ref_idx if i not in selected_ref_idx]
    
    nei_idx = list(range(len(list_of_neighbours_contours)))
    selected_nei_idx = [i['closest_neighbour_idx'] for i in list_of_neighbours_data_dicts]
    nei_to_rescue_idx = [i for i in nei_idx if i not in selected_nei_idx]
    
    contours_to_rescue_idx = {'ref_to_rescue_idx':ref_to_rescue_idx,
                              'nei_to_rescue_idx':nei_to_rescue_idx}
    
    return contours_to_rescue_idx

# %% retreive_contours_values_from_contours_to_rescue_data() ----
def retreive_contours_values_from_contours_to_rescue_data(list_of_contours, 
                                                          contours_to_rescue_data_dict,
                                                          contours_type):
    
    if contours_type == 'reference':
        contours_type_idx = 'ref_to_rescue_idx'
    elif contours_type == 'neighbour':
        contours_type_idx = 'nei_to_rescue_idx'
    out = []
    for i in contours_to_rescue_data_dict[contours_type_idx]:
        out.append(list_of_contours[i])
    return out

# %% merge_closest_neighbours_data_lists() ----
def merge_closest_neighbours_data_lists(ordered_list_of_lists_of_neighbours_data_dicts):
    for idx, list_of_neighbours_data_dicts in enumerate(ordered_list_of_lists_of_neighbours_data_dicts):
        for neighbours_data_dict in list_of_neighbours_data_dicts:
            neighbours_data_dict['pass'] = idx
    output = [d for list_of_dicts in ordered_list_of_lists_of_neighbours_data_dicts for d in list_of_dicts]
    return output
        
# %% retreive_contours_values_from_closest_neighbours_data() ----
def retreive_contours_values_from_closest_neighbours_data(list_of_contours, 
                                                          list_of_neighbours_data_dicts,
                                                          contours_type):
    if contours_type == 'reference':
        contours_type_idx = 'ref_idx'
    elif contours_type == 'neighbour':
        contours_type_idx = 'closest_neighbour_idx'
    out = []
    for i in list_of_neighbours_data_dicts:
        out.append(list_of_contours[i[contours_type_idx]])
    return out

