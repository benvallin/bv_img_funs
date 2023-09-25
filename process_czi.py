# %% Import dependencies ----
import os, re, czifile
import pandas as pd
import skimage.util
import imageio

# %% czi_to_df(): read and store czi files into a dataframe ----
def czi_to_df(path, pattern=None):
  if pattern is None:
    img_nm = sorted([x for x in os.listdir(path) if bool(re.search(pattern='^.*\\.czi$', string=x))])
  else:
    img_nm = sorted([x for x in os.listdir(path) if bool(re.search(pattern=pattern, string=x))])
  img_list = map(lambda x: czifile.imread(path+x), img_nm)
  output = pd.DataFrame({'czi_img': img_list}, index=img_nm)
  return output

# %% czi_df_to_img_df(): extract and store images into channel-specific columns  ----
def czi_df_to_img_df(df, channels, uint8_convert=False):
    for i, j in enumerate(channels):
        df[j] = df['czi_img'].map(lambda x: x[0,0,i,0,0,:,:,0])    
        if uint8_convert:
            df[j] = df[j].map(lambda x: list(map(lambda y: skimage.util.img_as_ubyte(y), x)))
    return df

# %% imwrite_from_img_df(): write images stored into channel-specific columns to files  ----
def imwrite_from_img_df(df, channels, outpath, format='tif'):
    for ch in channels:
        for img_nm in df.index:
            imageio.imwrite(uri=outpath+df.loc[[img_nm],[ch]].index[0][0:-4]+'_'+ch+'.'+format,
                            im=df.loc[img_nm,ch])