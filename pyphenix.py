
# %% Import dependencies ----
import os
import re
import copy
import numpy as np
import pandas as pd
import skimage.io 

# %% phenix_names_dict(): store phenix image file names in a dict ----
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

# %% dict_depth(): resolve max depth of a dict ----
def dict_depth(d, level=0):
    if not isinstance(d, dict) or not d:
      return level
    return max([dict_depth(d=d[k], level=level+1) for k in d])

# %% phenix_images_dict(): read and store phenix image arrays in a dict ----
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

# %% phenix_dict_to_df(): convert a phenix dict to a dataframe ----
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

# %% phenix_images_df(): read and store phenix image arrays in a dataframe ----
def phenix_images_df(path, names_dict=None, solve_zstack=False, ch_dict=None, sample_dict=None):
  
  # Call phenix_images_dict()
  output = phenix_images_dict(path=path, names_dict=names_dict, solve_zstack=solve_zstack)
  
  # Stop execution if phenix_images_dict() fails
  if output is None:
    return 
  # Otherwise Call phenix_dict_to_df()
  else:
    return phenix_dict_to_df(phenix_dict=output, ch_dict=ch_dict, sample_dict=sample_dict)
    
