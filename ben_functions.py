# %% Import dependencies
import os
import re
import copy
import numpy as np
import pandas as pd
import skimage.io 
import cv2 as cv
from cellpose import models, core, utils, io, models, plot

# import skimage.exposure
# import xarray as xr


# %% STORE PHENIX IMAGE FILE NAMES IN A DICT
def phenix_names_dict(path, solve_zstack=False, nest_zstack=False):
    
    # Stop execution if arguments are invalid
    # Stop if provided path does not exist
    if not os.path.exists(path):
        print('Execution stopped: provided path does not exist')
        return
     
    # Stop if provided zstack is invalid
    if solve_zstack not in [False, True]:
        print('Execution stopped: zstack is invalid')
        return
    
    # Stop if provided nest_zstack is invalid
    if nest_zstack not in [False, True]:
        print('Execution stopped: nest_zstack is invalid')
        return
    
    # List image file names, channels and row / column / FOV combinations
    image_names = sorted([x for x in os.listdir(path) if ".tiff" in x])
    def try_extract(pattern):
      try:
        return sorted({re.search(pattern, x).group() for x in image_names})
      except:
        return []
    channels = try_extract("ch\d{1}")
    row_col_fov = try_extract("r\d{2}c\d{2}f\d{2}")
    
    # Stop execution if target directory does not contain .tiff files with valid names
    lists_lengths = [len(x) for x in [image_names, channels, row_col_fov]]  
    if 0 in lists_lengths:
        print('Execution stopped: no .tiff files with valid names in target folder')
        return
    
    # List zstack positions:  
    if solve_zstack:
      position = try_extract("p\d{2}")
    
    # Stop execution if solve_zstack=True but file names do not contain zstack position  
    if solve_zstack and len(position) == 0:
      print('Execution stopped: zstack position not found in file names')
      return
    
    # Return a dict with indexed image names
    layer1, layer2 =  channels, row_col_fov
    if solve_zstack:
      layer3 = position
      
    output = {}
    # Case when zstack=True and nest_zstack=False
    if solve_zstack and not nest_zstack:
      for key1 in layer1:
        dict2 = {}
        for key2 in layer2:
          for key3 in layer3:
            value = [x for x in image_names if key1 in x if key2 in x if key3 in x]
            dict2[(key2, key3)] = value
        output[key1] = dict2
    # Case when zstack=True and nest_zstack=True
    elif solve_zstack and nest_zstack:
      for key1 in layer1:
        dict2 = {}
        for key2 in layer2:
          dict3 = {}
          for key3 in layer3:
            value3 = [x for x in image_names if key1 in x if key2 in x if key3 in x]
            dict3[key3] = value3
          dict2[key2] = dict3
        output[key1] = dict2
    # Case when zstack=False    
    else:
      if nest_zstack:
        print('Warning: "nest_zstack=True" ignored as solve_zstack is False') 
      for key1 in layer1:
        dict2 = {}
        for key2 in layer2:
          value = [x for x in image_names if key1 in x if key2 in x]
          dict2[key2] = value
        output[key1] = dict2
            
    return output

# %% RESOLVE MAX DEPTH OF A DICT
def dict_depth(d, level=0):
    if not isinstance(d, dict) or not d:
      return level
    return max([dict_depth(d=d[k], level=level+1) for k in d])

# %% READ AND STORE PHENIX IMAGE ARRAYS IN A DICT
def phenix_images_dict(path, names_dict=None, solve_zstack=False, nest_zstack=False):
    
  # Stop execution if arguments are invalid
  # Stop if provided path does not exist
  if not os.path.exists(path):
    print('Execution stopped: provided path does not exist')
    return
  
  # Prepare warning message is solve_zstack and/or nest_zstack set to True while providing names_dict
  if names_dict is not None and (solve_zstack or nest_zstack):
    return_zstack_arg_warning = True
  else:
    return_zstack_arg_warning = False
  
  # Construct names_dict if not provided
  if names_dict is None:
    names_dict = phenix_names_dict(path=path, solve_zstack=solve_zstack, nest_zstack=nest_zstack)
    if names_dict is None:
      return
    
  # Stop if provided names_dict is not a dict or is an empty dict
  if not isinstance(names_dict, dict) or not names_dict:
    print('Execution stopped: names_dict is not a dict or is an empty dict')
    return
  
  # Stop if provided names_dict is of invalid depth    
  input_depth = dict_depth(d=names_dict)
  if input_depth not in [2, 3]:
    print('Execution stopped: names_dict is of invalid depth')
    return
  
  # Return a dict with indexed image arrays
  output = copy.deepcopy(names_dict)

  def try_imread(path_to_file):
      try:
        return skimage.io.imread(path_to_file)
      except:
        return None
      
  # Case when names_dict does not contain nested zstack positions
  if input_depth == 2:
    for key1 in output.keys():
      for key2 in output[key1].keys():
        # Case when names_dict contains unsolved zstack positions
        if len(output[key1][key2]) > 1:
          output[key1][key2] = [try_imread(path+output[key1][key2][x]) for x in range(len(output[key1][key2]))]
          if any([isinstance(i, type(None)) for i in output[key1][key2]]):
            print('Execution stopped: one or more names_dict element(s) are missing in target directory')
            return
        # Case when names_dict does not contain zstack positions or contains solved but not nested zstack positions   
        else:
          output[key1][key2] = try_imread(path+output[key1][key2][0])
          if output[key1][key2] is None:
            print('Execution stopped: one or more names_dict element(s) are missing in target directory')
            return
  # Case when names_dict contains nested zstack positions
  else:
    for key1 in output.keys():
      for key2 in output[key1].keys():
        for key3 in output[key1][key2].keys():
          output[key1][key2][key3] = try_imread(path+output[key1][key2][key3][0])
          if output[key1][key2][key3] is None:
            print('Execution stopped: one or more names_dict element(s) are missing in target directory')
            return
  
  # Print warning message is solve_zstack and/or nest_zstack set to True while providing names_dict      
  if return_zstack_arg_warning:
    print('Warning: solve_zstack and nest_zstack are ignored when names_dict is provided')
    
  return output  

# %% CONVERT A PHENIX DICT TO A DATAFRAME
def phenix_dict_to_df(phenix_dict, ch_dict=None, sample_dict=None):
  
  # Stop execution if arguments are invalid
  # Stop if provided phenix_dict is not a dict or is an empty dict
  if not isinstance(phenix_dict, dict) or not phenix_dict:
    print('Execution stopped: phenix_dict is not a dict or is an empty dict')
    return
  
  # Stop if provided phenix_dict is of invalid depth    
  input_depth = dict_depth(d=phenix_dict)
  if input_depth not in [2, 3]:
    print('Execution stopped: phenix_dict is of invalid depth')
    return
  
  # Check if provided phenix_dict is zstack_indexed (zstack positions can be nested or not nested)
  if input_depth == 3:
    zstack_indexed_input = True
  else:
    zstack_indexed_input = []
    for i in phenix_dict.keys():
        for j in list(phenix_dict[i].keys()):
            if len(j) == 2:
                zstack_indexed_input.append(True)
            else:
                zstack_indexed_input.append(False)
    zstack_indexed_input = list(set(zstack_indexed_input))[0]
  
  # Stop if provided ch_dict is not a dict or is an empty dict
  if ch_dict is not None and (not isinstance(ch_dict, dict) or not ch_dict):
    print('Execution stopped: ch_dict is not a dict or is an empty dict')
    return
  
  # Stop if provided sample_dict is not a dict or is an empty dict
  if sample_dict is not None and (not isinstance(sample_dict, dict) or not sample_dict):
    print('Execution stopped: sample_dict is not a dict or is an empty dict')
    return
  
  # Stop if provided sample_dict contains non str key(s)
  if sample_dict is not None:
    str_check_sample_dict_keys = {isinstance(i, str) for i in sample_dict.keys()}
    if len(str_check_sample_dict_keys) > 1 or str_check_sample_dict_keys == {False}:
      print('Execution stopped: sample_dict keys must be str')
      return
  
  # Convert phenix_dict to a pd.DataFrame  
  # Case when phenix_dict does not contain nested zstack positions
  if input_depth == 2:
    output = pd.DataFrame(phenix_dict)
  # Case when names_dict contains nested zstack positions
  else:
    output = {}
    for k in phenix_dict.keys():
      output[k] = pd.DataFrame(phenix_dict[k]).unstack()
    output = pd.DataFrame(output)
  
  # Change column names if a valid ch_dict was provided
  if ch_dict is not None:  
    output.rename(columns=ch_dict, inplace=True)
  
  # Change index if a valid sample_dict was provided  
  if sample_dict is not None:
    # Case when phenix_dict is not zstack_indexed
    if input_depth == 2 and not zstack_indexed_input:
      sample_idx = pd.Series(output.index)
      for i in sample_dict:
        sample_idx = pd.Series(np.where(sample_idx.str.contains(i), sample_dict[i], sample_idx))
      output.index = [sample_idx, output.index]
    # Case when phenix_dict is zstack_indexed  
    else:
      sample_idx = pd.Series([i[0] for i in output.index])
      for i in sample_dict:
          sample_idx = pd.Series(np.where(sample_idx.str.contains(i), sample_dict[i], sample_idx))
      zstack_idx = pd.Series([i[1] for i in output.index])
      output.index = [sample_idx, [i[0] for i in output.index], zstack_idx]
  
  return output

# %% READ AND STORE PHENIX IMAGE ARRAYS IN A DATAFRAME
def phenix_images_df(path, names_dict=None, solve_zstack=False, ch_dict=None, sample_dict=None):
  
  # Call phenix_images_dict()
  output = phenix_images_dict(path=path, names_dict=names_dict, solve_zstack=solve_zstack)
  
  # Stop execution if phenix_images_dict() fails
  if output is None:
    return 
  # Otherwise Call phenix_dict_to_df()
  else:
    return phenix_dict_to_df(phenix_dict=output, ch_dict=ch_dict, sample_dict=sample_dict)
    
# %% Store image maximum projections in dict
def make_max_proj(img_dict, path):
    
    # Stop execution if arg values are invalid
    if not os.path.exists(path):
        print('Execution stopped: provided path does not exist')
        return
    if len(img_dict) == 0:
        print('Execution stopped: provided dict is empty')
        return
    
    # Return dict with maximum projection for each image
    output = {}

    for outerkey in img_dict:
        innerdict = {}
        for innerkey in img_dict[outerkey]:
            innervalues = [skimage.io.imread(path + x) for x in img_dict[outerkey][innerkey]]
            innervalues = xr.DataArray(innervalues)
            innervalues = xr.concat(innervalues, dim = "pln")
            innervalues = np.max(innervalues, axis = 0)
            innerdict[innerkey] = innervalues
        output[outerkey] = innerdict
    print("Done!")    
    
    return output
    
# %% Rescale intensity of an image
def rescale_intensity(img, minp=0.5, maxp=99.5):
    
  out = skimage.exposure.rescale_intensity(img, in_range=tuple(np.percentile(img, (minp, maxp))))
  
  return out

# %% Adjust gamma of an image
def gamma_adjust(src, gamma):
    lut = np.round(np.multiply(np.power(np.divide(np.arange(0,256), 255), (1/gamma)), 255)).astype(np.uint8)
    out = cv.LUT(src, lut)
    return out

# %% Write maximum projection stored in dict to file
def save_maximum_projections(proj_dict, path):
    
    # Stop execution if arg values are invalid
    if not os.path.exists(path):
        print('Execution stopped: provided path does not exist')
        return
    if len(proj_dict) == 0:
        print('Execution stopped: provided dict is empty')
        return
    
    # Write dict with maximum projections to disk
    for outerkey in proj_dict:
        for innerkey in proj_dict[outerkey]:
            skimage.io.imsave(path + innerkey + "_" + outerkey + ".tiff", proj_dict[outerkey][innerkey])
    
    return print("Done!")

# %% make_float_rgb_images
def make_float_rgb_images(red_img_list, green_img_list, blue_img_list):
  rgb_img_list = []
  for i in range(len(red_img_list)):
    rgb_img_list.append(np.zeros(shape=(1080,1080, 3), dtype='uint16'))
    rgb_img_list[i][:,:,0]=red_img_list[i][:,:]
    rgb_img_list[i][:,:,1]=green_img_list[i][:,:]    
    rgb_img_list[i][:,:,2]=blue_img_list[i][:,:]  
    rgb_img_list[i] = skimage.util.img_as_float(rgb_img_list[i])
  return rgb_img_list

# %% make_mask_outlines_and_submasks
def make_mask_outlines_and_submasks(list_of_main_colored_masks):
  outer_outlines = []
  for i in range(len(list_of_main_colored_masks)):
    inner_outlines = utils.outlines_list(list_of_main_colored_masks[i])
    inner_outlines = [inner_outlines[i] for i in range(len(inner_outlines)) if inner_outlines[i].any()]
    outer_outlines.append(inner_outlines)
  outer_masks = []
  for i in range(len(list_of_main_colored_masks)):
    inner_masks = []
    for j in range(len(outer_outlines[i])):
      mask = skimage.draw.polygon2mask(image_shape=list_of_main_colored_masks[i].shape, polygon=outer_outlines[i][j])
      mask = mask.T
      inner_masks.append(mask)
    outer_masks.append(inner_masks)
  return outer_outlines, outer_masks

# %% get_submasks_mean_pixel_intensity
def get_submasks_mean_pixel_intensity(list_of_lists_of_submasks, list_of_imgs):
  outer_list = []
  for i in range(len(list_of_lists_of_submasks)):
    inner_list = []
    for j in range(len(list_of_lists_of_submasks[i])):
      submask_loc = list_of_lists_of_submasks[i][j]
      pixels_in_submask = skimage.util.img_as_ubyte(list_of_imgs[i])[submask_loc]
      mean_submask_pixel_intensity = np.mean(pixels_in_submask)
      inner_list.append(mean_submask_pixel_intensity)
    outer_list.append(inner_list)
  return outer_list

# %% get_submasks_area
def get_submasks_area(list_of_lists_of_submasks):
  outer_list = []
  for i in range(len(list_of_lists_of_submasks)):
    inner_list = []
    for j in range(len(list_of_lists_of_submasks[i])):
      submask_pixel_count = np.count_nonzero(list_of_lists_of_submasks[i][j])
      inner_list.append(submask_pixel_count)
    outer_list.append(inner_list)
  return outer_list

# %% keep_drop_2d_arrays_based_on_percentile
def keep_drop_2d_arrays_based_on_percentile(list_of_lists_of_2d_arrays, list_of_lists_of_parameter_values, percentile_threshold):
  percentile_parameter_value = np.percentile(a = [x for i in list_of_lists_of_parameter_values for x in i], q=percentile_threshold)
  outer_recovered = []
  outer_dropped = []
  for i in range(len(list_of_lists_of_2d_arrays)):
    inner_recovered = []
    inner_dropped = []
    for j in range(len(list_of_lists_of_2d_arrays[i])):
      if list_of_lists_of_parameter_values[i][j] > percentile_parameter_value:
        inner_recovered.append(list_of_lists_of_2d_arrays[i][j])
      else:
        inner_dropped.append(list_of_lists_of_2d_arrays[i][j])
    outer_recovered.append(inner_recovered)
    outer_dropped.append(inner_dropped)
  return outer_recovered, outer_dropped

# %% make_set_of_2d_array
def make_set_of_2d_arrays(main_list, sublist1, sublist2, test_op):
  outer_sublist1_bool = []
  outer_sublist2_bool = []
  for i in range(len(main_list)):
    inner_sublist1_bool = []
    for j in range(len(sublist1)):
      inner_sublist1_bool.append(np.array_equal(main_list[i], sublist1[j]))
    inner_sublist1_bool = any(inner_sublist1_bool)
    outer_sublist1_bool.append(inner_sublist1_bool)
    inner_sublist2_bool = []
    for k in range(len(sublist2)):
      inner_sublist2_bool.append(np.array_equal(main_list[i], sublist2[k]))
    inner_sublist2_bool = any(inner_sublist2_bool)
    outer_sublist2_bool.append(inner_sublist2_bool)
  if test_op == "both_in":
    fct = all
  elif test_op == "any_in":
    fct = any
  sublist1_sublist2_bool = [fct(i) for i in zip(outer_sublist1_bool, outer_sublist2_bool)]
  output = [main_list[i] for i in range(len(main_list)) if sublist1_sublist2_bool[i]]
  return output

# %% make_imgs_from_submasks
def make_imgs_from_submasks_outlines(list_of_imgs, list_of_lists_of_outlines):
  imgs_in_glob_masks = []
  for i in range(len(list_of_imgs)):
    img = list_of_imgs[i]
    outlines = list_of_lists_of_outlines[i]
    glob_mask = np.zeros(shape = img.shape, dtype = 'bool')
    for j in range(len(outlines)):
      submask = skimage.draw.polygon2mask(image_shape=img.shape, polygon=outlines[j])
      submask = submask.T
      glob_mask = np.ma.mask_or(m1=glob_mask, m2=submask)
    img_in_glob_mask = np.where(glob_mask == 1, img, 0)
    imgs_in_glob_masks.append(img_in_glob_mask)
  return imgs_in_glob_masks

# %% make_masks_from_outlines
def make_masks_from_outlines(list_of_lists_of_outlines, shape):
  masks_outer_list = []
  for i in range(len(list_of_lists_of_outlines)):
    masks_inner_list = []
    for j in range(len(list_of_lists_of_outlines[i])):
      mask = skimage.draw.polygon2mask(image_shape=shape, polygon=list_of_lists_of_outlines[i][j])
      mask = mask.T
      masks_inner_list.append(mask)
    masks_outer_list.append(masks_inner_list)
  return masks_outer_list

# %% make_main_mask_from_submasks
def make_main_mask_from_submasks(list_of_lists_of_submasks, shape):
  list_of_main_masks = []
  for i in range(len(list_of_lists_of_submasks)):
    main_mask = np.zeros(shape=shape, dtype='bool')
    for j in range(len(list_of_lists_of_submasks[i])):
      main_mask = np.where(list_of_lists_of_submasks[i][j] == 1, list_of_lists_of_submasks[i][j], main_mask)
    list_of_main_masks.append(main_mask)
  return list_of_main_masks

# %% make_masks_overlap
def make_masks_overlap(list_of_masks1, list_of_masks2):
  list_of_masks_overlaps = []
  for i in range(len(list_of_masks1)):
    overlap = np.where(list_of_masks1[i] == 1, list_of_masks2[i], 0)
    list_of_masks_overlaps.append(overlap)
  return list_of_masks_overlaps

# %% keep_submasks_touching_main_mask
def keep_submasks_touching_main_mask(list_of_lists_of_submasks, list_of_main_masks):
  outer_list = []
  for i in range(len(list_of_lists_of_submasks)):
    inner_list = []
    for j in range(len(list_of_lists_of_submasks[i])):
      touching_part = np.where(list_of_lists_of_submasks[i][j] == 1, list_of_main_masks[i], 0)
      touching_part = np.any(touching_part)
      if touching_part == True:
        inner_list.append(list_of_lists_of_submasks[i][j])
    outer_list.append(inner_list)
  return outer_list

# %% get_submasks1_overlapped_by_submasks2_beta
def get_submasks1_overlapped_by_submasks2_beta(list_of_lists_of_submasks_to_filter, list_of_lists_of_submasks_to_filter_by, pct_overlap):
  outer_list = []
  for i in range(len(list_of_lists_of_submasks_to_filter)):
    inner_list = []
    for j in range(len(list_of_lists_of_submasks_to_filter[i])):
      submask_pixel_count = np.count_nonzero(list_of_lists_of_submasks_to_filter[i][j])
      n_touching_parts_above_threshold = 0
      for k in range(len(list_of_lists_of_submasks_to_filter_by[i])):
        touching_part = np.where(list_of_lists_of_submasks_to_filter[i][j] == 1,
                                 list_of_lists_of_submasks_to_filter_by[i][k],
                                 0)
        touching_part_bool = np.any(touching_part)
        if touching_part_bool == True:
          touching_part_pixel_count = np.count_nonzero(touching_part)
          touching_part_area_pass_threshold = (touching_part_pixel_count / submask_pixel_count) >= pct_overlap
          if touching_part_area_pass_threshold:
            n_touching_parts_above_threshold += 1
      if n_touching_parts_above_threshold > 0:
        inner_list.append(list_of_lists_of_submasks_to_filter[i][j])
    outer_list.append(inner_list)
  return outer_list

# %% get_submasks1_overlapped_by_submasks2
def get_submasks1_overlapped_by_submasks2(list_of_lists_of_submasks_to_filter, list_of_lists_of_submasks_to_filter_by, min_pct_overlap):
  overlapped_submasks1 = []
  overlapping_submasks2 = []
  for i in range(len(list_of_lists_of_submasks_to_filter)):
    inner_overlapped_submasks1 = [False] * len(list_of_lists_of_submasks_to_filter[i])
    inner_overlapping_submasks2 = [False] * len(list_of_lists_of_submasks_to_filter_by[i])
    for j in range(len(list_of_lists_of_submasks_to_filter[i])):
      submask_pixel_count = np.count_nonzero(list_of_lists_of_submasks_to_filter[i][j])
      any_touching_parts_above_threshold = False
      for k in range(len(list_of_lists_of_submasks_to_filter_by[i])):
        touching_part = np.where(list_of_lists_of_submasks_to_filter[i][j] == 1,
                                 list_of_lists_of_submasks_to_filter_by[i][k],
                                 0)
        touching_part_bool = np.any(touching_part)
        if touching_part_bool:
          touching_part_pixel_count = np.count_nonzero(touching_part)
          touching_part_area_pass_threshold = (touching_part_pixel_count / submask_pixel_count) >= min_pct_overlap
          if touching_part_area_pass_threshold:
            any_touching_parts_above_threshold = True
            inner_overlapping_submasks2[k] = True             
      if any_touching_parts_above_threshold:
        inner_overlapped_submasks1[j] = True
    inner_overlapped_submasks1 = [list_of_lists_of_submasks_to_filter[i][x] for x in range(len(inner_overlapped_submasks1)) if inner_overlapped_submasks1[x] == True]
    overlapped_submasks1.append(inner_overlapped_submasks1)
    inner_overlapping_submasks2 = [list_of_lists_of_submasks_to_filter_by[i][x] for x in range(len(inner_overlapping_submasks2)) if inner_overlapping_submasks2[x] == True]
    overlapping_submasks2.append(inner_overlapping_submasks2)
  return overlapped_submasks1, overlapping_submasks2

# %% make_submasks_outlines
def make_submasks_outlines(list_of_lists_of_submasks):
  outer_list = []
  for i in range(len(list_of_lists_of_submasks)):
    inner_list = []
    for j in range(len(list_of_lists_of_submasks[i])):
      inner_list.extend(utils.outlines_list(list_of_lists_of_submasks[i][j]))
    outer_list.append(inner_list)
  return outer_list