# %% Import dependencies ----
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import skimage.util, skimage.draw
import cellpose.utils, cellpose.models
import itertools

# %% make_float_rgb_images() ----
def make_float_rgb_images(red_img_list, green_img_list, blue_img_list):
  rgb_img_list = []
  for i in range(len(red_img_list)):
    rgb_img_list.append(np.zeros(shape=(1080,1080, 3), dtype='uint16'))
    rgb_img_list[i][:,:,0]=red_img_list[i][:,:]
    rgb_img_list[i][:,:,1]=green_img_list[i][:,:]    
    rgb_img_list[i][:,:,2]=blue_img_list[i][:,:]  
    rgb_img_list[i] = skimage.util.img_as_float(rgb_img_list[i])
  return rgb_img_list

# %% make_mask_outlines_and_submasks() ----
def make_outlines_and_submasks_from_main_masks(list_of_main_masks):
  outer_outlines = []
  for i in range(len(list_of_main_masks)):
    inner_outlines = cellpose.utils.outlines_list(list_of_main_masks[i])
    inner_outlines = [inner_outlines[i] for i in range(len(inner_outlines)) if inner_outlines[i].any()]
    outer_outlines.append(inner_outlines)
  outer_masks = []
  for i in range(len(list_of_main_masks)):
    inner_masks = []
    for j in range(len(outer_outlines[i])):
      mask = skimage.draw.polygon2mask(image_shape=list_of_main_masks[i].shape, polygon=outer_outlines[i][j])
      mask = mask.T
      inner_masks.append(mask)
    outer_masks.append(inner_masks)
  return outer_outlines, outer_masks

# Alternative version using map() instead of for loops
def make_outlines_and_submasks_from_main_masks2(list_of_main_masks):
  
  def make_submasks(outlines, image_shape):
    submask = skimage.draw.polygon2mask(image_shape=image_shape, polygon=outlines)
    submask = submask.T
    return submask
  
  def make_outlines_and_submasks(main_mask):
    outlines = cellpose.utils.outlines_list(main_mask)
    outlines = [outlines[i] for i in range(len(outlines)) if outlines[i].any()]
    submasks = list(map(make_submasks, 
                    outlines, 
                    itertools.repeat(main_mask.shape, len(outlines))))
    return outlines, submasks
  
  outlines_submasks = list(map(make_outlines_and_submasks, list_of_main_masks))
  outlines = [i[0] for i in outlines_submasks]
  submasks = [i[1] for i in outlines_submasks]
  
  return outlines, submasks

# Function to make outlines only from a single mask
def make_outlines(mask):
    outlines = cellpose.utils.outlines_list(mask)
    outlines = [outlines[i] for i in range(len(outlines)) if outlines[i].any()]
    return outlines

# Function to make submasks only from the outlines of a single mask
def make_submasks(outlines, image_shape):
    submasks = list(map(lambda x: skimage.draw.polygon2mask(image_shape=image_shape, polygon=x).T,
                        outlines))
    return submasks
  
# %% get_submasks_mean_pixel_intensity() ----
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

# %% get_submasks_quantile_pixel_intensity() ----
def get_submasks_quantile_pixel_intensity(list_of_lists_of_submasks, list_of_imgs, q):
  outer_list = []
  for i in range(len(list_of_lists_of_submasks)):
    inner_list = []
    for j in range(len(list_of_lists_of_submasks[i])):
      submask_loc = list_of_lists_of_submasks[i][j]
      pixels_in_submask = skimage.util.img_as_ubyte(list_of_imgs[i])[submask_loc]
      quantile_submask_pixel_intensity = np.quantile(a=pixels_in_submask, q=q)
      inner_list.append(quantile_submask_pixel_intensity)
    outer_list.append(inner_list)
  return outer_list

# %% get_submasks_pixel_count() ----
def get_submasks_pixel_count(list_of_lists_of_submasks):
  outer_list = []
  for i in range(len(list_of_lists_of_submasks)):
    inner_list = []
    for j in range(len(list_of_lists_of_submasks[i])):
      submask_pixel_count = np.count_nonzero(list_of_lists_of_submasks[i][j])
      inner_list.append(submask_pixel_count)
    outer_list.append(inner_list)
  return outer_list

# %% keep_drop_2d_arrays_based_on_percentile() ----
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

# %% make_set_of_2d_arrays() ----
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

# %% make_imgs_from_submasks_outlines() ----
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

# %% make_masks_from_outlines() ----
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

# %% make_main_mask_from_submasks() ----
def make_main_mask_from_submasks(list_of_lists_of_submasks, shape):
  list_of_main_masks = []
  for i in range(len(list_of_lists_of_submasks)):
    main_mask = np.zeros(shape=shape, dtype='bool')
    for j in range(len(list_of_lists_of_submasks[i])):
      main_mask = np.where(list_of_lists_of_submasks[i][j] == 1, list_of_lists_of_submasks[i][j], main_mask)
    list_of_main_masks.append(main_mask)
  return list_of_main_masks

# %% make_masks_overlap() ----
def make_masks_overlap(list_of_masks1, list_of_masks2):
  list_of_masks_overlaps = []
  for i in range(len(list_of_masks1)):
    overlap = np.where(list_of_masks1[i] == 1, list_of_masks2[i], 0)
    list_of_masks_overlaps.append(overlap)
  return list_of_masks_overlaps

# %% keep_submasks_touching_main_mask() ----
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

# %% get_submasks1_overlapped_by_submasks2() ----
def get_submasks1_overlapped_by_submasks2(list_of_lists_of_submasks_to_be_overlapped, 
                                          list_of_lists_of_submasks_to_overlap_by, 
                                          min_pct_overlap):
  
  # Create an empty outer list of overlapped submasks 
  overlapped_submasks = []
  # Create an empty outer list of overlapping submasks 
  overlapping_submasks = []
  # For each inner list of submasks to be overlapped
  # (Note that n inner list of submasks to be overlapped == n inner list of submasks to overlap by)
  for i in range(len(list_of_lists_of_submasks_to_be_overlapped)):
    # Create an inner list of overlapped submasks with length = n submasks in inner list of submasks to be overlapped
    # => Default all element values to False
    inner_overlapped_submasks = [False] * len(list_of_lists_of_submasks_to_be_overlapped[i])
    # Create an inner list of overlapping submasks with length = n submasks in inner list of submasks to overlap by
    # => Default all element values to False
    inner_overlapping_submasks = [False] * len(list_of_lists_of_submasks_to_overlap_by[i])
    # For each submask in current inner list of submasks to be overlapped
    for j in range(len(list_of_lists_of_submasks_to_be_overlapped[i])):
      # Compute submask area
      submask_pixel_count = np.count_nonzero(list_of_lists_of_submasks_to_be_overlapped[i][j])
      # Define variable any_touching_parts_above_threshold
      # => Indicates if current submask to be overlapped if overlapped by any submask to overlap by
      # => Default to False
      any_touching_parts_above_threshold = False
      # For each submask in current inner list of submasks to overlap by
      for k in range(len(list_of_lists_of_submasks_to_overlap_by[i])):
        # Extract the part of current submask to overlap by which overlaps the current submask to be overlapped
        touching_part = np.where(list_of_lists_of_submasks_to_be_overlapped[i][j] == 1,
                                 list_of_lists_of_submasks_to_overlap_by[i][k],
                                 0)
        # Record if there is overlap 
        touching_part_bool = np.any(touching_part)
        # If there is overlap
        if touching_part_bool:
          # Compute overlap area
          touching_part_pixel_count = np.count_nonzero(touching_part)
          # Record if overlap area >= user-defined threshold
          touching_part_area_pass_threshold = (touching_part_pixel_count / submask_pixel_count) >= min_pct_overlap
          # if overlap area >= user-defined threshold
          if touching_part_area_pass_threshold:
            # Reset any_touching_parts_above_threshold to True
            # => Indicates that current submask to be overlapped if overlapped by at least one submask to overlap by
            any_touching_parts_above_threshold = True
            # In current inner list of overlapping submasks, reset element corresponding to current submask to overlap by to True 
            inner_overlapping_submasks[k] = True   
      # If current submask to be overlapped if overlapped by at least one submask
      if any_touching_parts_above_threshold:
        # In current inner list of overlapped submasks, reset element corresponding to current submask to be overlapped to True 
        inner_overlapped_submasks[j] = True
    # Fill current inner list of overlapped submasks with actual submasks values (replace True values with values of corresponding submasks to be overlapped) 
    inner_overlapped_submasks = [list_of_lists_of_submasks_to_be_overlapped[i][x] for x in range(len(inner_overlapped_submasks)) if inner_overlapped_submasks[x] == True]
    # Add current inner list of overlapped submasks to outer list of overlapped submasks 
    overlapped_submasks.append(inner_overlapped_submasks)
    # Fill current inner list of overlapping submasks with actual submasks values (replace True values with values of corresponding submasks to overlap by) 
    inner_overlapping_submasks = [list_of_lists_of_submasks_to_overlap_by[i][x] for x in range(len(inner_overlapping_submasks)) if inner_overlapping_submasks[x] == True]
    # Add current inner list of overlapping submasks to outer list of overlapping submasks 
    overlapping_submasks.append(inner_overlapping_submasks)
  # Return outer lists of overlapped submasks and overlapping submasks
  return overlapped_submasks, overlapping_submasks

# %% get_submasks_overlapped_by_mask() ----
def get_submasks_overlapped_by_masks(list_of_lists_of_submasks_to_be_overlapped, 
                                     list_of_masks_to_overlap_by, 
                                     min_pct_overlap):
    
    overlapped_submasks = []
    
    for i in range(len(list_of_lists_of_submasks_to_be_overlapped)):
        
        inner_overlapped_submasks = [False] * len(list_of_lists_of_submasks_to_be_overlapped[i])
        
        for j in range(len(list_of_lists_of_submasks_to_be_overlapped[i])):
            
            touching_part = np.where(list_of_lists_of_submasks_to_be_overlapped[i][j] == 1,
                                     list_of_masks_to_overlap_by[i],
                                     0)
            
            if np.any(touching_part):
                
                submask_pixel_count = np.count_nonzero(list_of_lists_of_submasks_to_be_overlapped[i][j])
                
                touching_part_pixel_count = np.count_nonzero(touching_part)
                
                touching_part_area_pass_threshold = (touching_part_pixel_count / submask_pixel_count) >= min_pct_overlap
                
                if touching_part_area_pass_threshold:
                    
                    inner_overlapped_submasks[j] = True
                    
        inner_overlapped_submasks = [list_of_lists_of_submasks_to_be_overlapped[i][x] for x in range(len(inner_overlapped_submasks)) if inner_overlapped_submasks[x] == True]
        
        overlapped_submasks.append(inner_overlapped_submasks)
        
    return overlapped_submasks

# %% make_submasks_outlines() ----
def make_submasks_outlines(list_of_lists_of_submasks):
  outer_list = []
  for i in range(len(list_of_lists_of_submasks)):
    inner_list = []
    for j in range(len(list_of_lists_of_submasks[i])):
      inner_list.extend(cellpose.utils.outlines_list(list_of_lists_of_submasks[i][j]))
    outer_list.append(inner_list)
  return outer_list

# %% plot_img_set_from_multiindex_frame() ----
def plot_img_set_from_multiindex_frame(frame, channel, fov, sides_inch=10, title_upper=False, title_size=None):
  title_size = title_size if title_size is not None else 12
  if len(frame.index[0]) == 3:
    nrows=len({i[0] for i in frame.index})
    ncols=len({i[1] for i in frame.index})
  else:
    nrows=len({i[0] for i in frame.index})*len({i[2] for i in frame.index})
    ncols=len({i[1] for i in frame.index})
  fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
  fig.set_size_inches(ncols*sides_inch, nrows*sides_inch)
  fig.subplots_adjust(wspace = 0, hspace = 0.1)
  x,y=(0,0)
  slice_op = pd.IndexSlice[:,:,fov] if len(frame.index[0]) == 3 else pd.IndexSlice[:,:,:,fov]
  for i,j in zip(frame.loc[slice_op, channel],
                 [' '.join(i) for i in frame.loc[slice_op, channel].index]):
    axs[x,y].imshow(i)
    axs[x,y].axis('off')
    if title_upper:
      axs[x,y].set_title(j.upper(), fontdict={'fontsize': title_size})
    else:
      axs[x,y].set_title(j, fontdict={'fontsize': title_size})
    if y==ncols-1:
      y=0
      x+=1
    else:
      y+=1
  return fig

# Old version of plot_img_set_from_multiindex_frame():
# def plot_img_set_from_multiindex_frame(frame, channel, fov, sides_inch=10, title_upper=False, title_size=None):
#   title_size = title_size if title_size is not None else 12
#   nrows=len({i[0] for i in frame.index})*len({i[2] for i in frame.index})
#   ncols=len({i[1] for i in frame.index})
#   fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
#   fig.set_size_inches(ncols*sides_inch, nrows*sides_inch)
#   fig.subplots_adjust(wspace = 0, hspace = 0.1)
#   x,y=(0,0)
#   for i,j in zip(frame.loc[pd.IndexSlice[:,:,:,fov], channel],
#                  [' '.join(i) for i in frame.loc[pd.IndexSlice[:,:,:,fov], channel].index]):
#     axs[x,y].imshow(i)
#     axs[x,y].axis('off')
#     if title_upper:
#       axs[x,y].set_title(j.upper(), fontdict={'fontsize': title_size})
#     else:
#       axs[x,y].set_title(j, fontdict={'fontsize': title_size})
#     if y==ncols-1:
#       y=0
#       x+=1
#     else:
#       y+=1
#   return fig

# %% get_submasks_idx_touching_main_mask() ----
def get_submasks_idx_touching_main_mask(list_of_lists_of_submasks, list_of_main_masks):
        
    def does_submask_touch_main_mask(submask, main_mask):
        return np.any(np.where(submask, main_mask, 0)) 
    
    submask_touch_main_mask = list(map(lambda x, y: list(map(lambda z: does_submask_touch_main_mask(submask=z, main_mask=y), x)),
                                       list_of_lists_of_submasks,
                                       list_of_main_masks))
    
    def get_touching_index(list_of_submasks):
        return [idx for idx, touch_bool in enumerate(list_of_submasks) if touch_bool]
    
    return list(map(lambda x: get_touching_index(list_of_submasks=x), submask_touch_main_mask))
  
# %% run_cp3d_from_df() ----
# Function to run cellpose 3D from a dataframe and return cellpose masks 
def run_cp3d_from_df(df, channel, model_type='cyto2', diameter=30, cellprob_threshold=0.0, min_size=15, stitch_threshold=0.5):
    
    # Convert channel stacks to np.array
    if not isinstance(df[channel][0], np.ndarray):
        df[channel] = df[channel].map(np.array)
        
    # Define cellpose model
    model = cellpose.models.Cellpose(gpu=True, model_type=model_type)
    
    masks, *_ = model.eval(x=df.loc[:, channel].to_list(), 
                          channels=[0,0], 
                          do_3D=False, 
                          diameter=diameter,
                          cellprob_threshold=cellprob_threshold,
                          min_size=min_size,
                          stitch_threshold=stitch_threshold)
    del _
        
    # Store cellpose masks in dataframe
    masks_df = pd.DataFrame({'cp_masks':masks}, 
                            index=df.index)
    
    return masks_df

# %% cp3d_masks_to_outlines() ----
# Function to retreive outlines from cellpose 3D masks and to combine them across planes
def cp3d_masks_to_outlines(cp_masks_df):
        
    def create_outlines_df(masks):
        outlines = [cellpose.utils.outlines_list(i) for i in masks]
        outlines_idx = [np.delete(np.unique(i), 0) for i in masks]
        out_dict = {}
        for idx in range(len(outlines_idx)):
            in_dict = {'mask'+str(outlines_idx[idx][i]):outlines[idx][i] for i in range(len(outlines_idx[idx]))}
            out_dict['plane'+str(idx)] = in_dict
        df = pd.DataFrame(out_dict)
        df.index.name='mask_id'
        df.columns.name='plane_id'
        return df
    
    def combine_outlines(outlines):
        combined_outlines = np.vstack(outlines.dropna())
        combined_outlines = np.unique(combined_outlines, axis=0)
        mask = cv.drawContours(image=np.zeros(shape=cp_masks_df['cp_masks'][0][0].shape, dtype=np.uint8),
                               contours=[combined_outlines], 
                               contourIdx=-1, 
                               color=255, 
                               thickness=-1)
        combined_outlines = cv.findContours(image=mask,
                                            mode=cv.RETR_LIST, 
                                            method=cv.CHAIN_APPROX_NONE)[0][0][:,0,:]
        return combined_outlines
    
    cp_outlines_df = []
    
    for idx in cp_masks_df.index:
        df = create_outlines_df(masks=cp_masks_df.loc[idx, 'cp_masks'])
        df['all_planes'] = df.apply(combine_outlines, axis=1)
        df.index = pd.MultiIndex.from_tuples(i+tuple([j]) for i, j in zip(itertools.repeat(idx, len(df.index)), df.index))
        cp_outlines_df.append(df)
    cp_outlines_df = pd.concat(cp_outlines_df)
    cp_outlines_df.index.names = cp_masks_df.index.names+['cp_mask_id']

    return cp_outlines_df
  
# %% find_and_tidy_contours() ----
def find_and_tidy_contours(image, mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_NONE):
  
  out = cv.findContours(image=image,
                        mode=mode, 
                        method=method)[0]
  if not len(out):
      out = []
  else:
    out = list(map(lambda x: x[:,0,:], out))
  return out   

# %% get_submasks_pct_px_above_intens_thres() ----
def get_submasks_pct_px_above_intens_thres(list_of_lists_of_submasks, list_of_imgs, px_intens_thres):
  outer_list = []
  for i in range(len(list_of_lists_of_submasks)):
    inner_list = []
    for j in range(len(list_of_lists_of_submasks[i])):
      submask_loc = list_of_lists_of_submasks[i][j]
      pixels_in_submask = skimage.util.img_as_ubyte(list_of_imgs[i])[submask_loc]
      pct_px_above_thres = (len([i for i in pixels_in_submask if i >= px_intens_thres]) / len(pixels_in_submask))* 100
      inner_list.append(pct_px_above_thres)
    outer_list.append(inner_list)
  return outer_list

# %% sort_objs_with_pct_px_above_intens_thres() ----
def sort_objs_with_pct_px_above_intens_thres(list_of_lists_of_objs,
                                             list_of_lists_of_pct_px_above_intens_thres,
                                             pct_px_intens_thres):
    outer_rec_list = []
    outer_dis_list = []
    for i in range(len(list_of_lists_of_objs)):
        inner_rec_list = []
        inner_dis_list = []
        for j in range(len(list_of_lists_of_objs[i])):
            if list_of_lists_of_pct_px_above_intens_thres[i][j] >= pct_px_intens_thres:
                inner_rec_list.append(list_of_lists_of_objs[i][j])
            else:
                inner_dis_list.append(list_of_lists_of_objs[i][j])
        outer_rec_list.append(inner_rec_list)
        outer_dis_list.append(inner_dis_list)
    return outer_rec_list, outer_dis_list