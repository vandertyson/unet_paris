

import skimage.transform
import shapely.wkt
import tables as tb
import tqdm
import pandas as pd
import numpy as np
from matplotlib import pyplot
import gdal


# FN_MASK_CSV = "/data/train/AOI_3_Paris_Train/summaryData/AOI_3_Paris_Train_Building_Solutions.csv"
FN_MASK_CSV = "./AOI_3_Paris_Train_Building_Solutions.csv"
FN_RGB = "/data/train/AOI_3_Paris"
INPUT_SIZE = 256

def resize_original_im(img):
    img = np.array(img).swapaxes(0,1).swapaxes(1,2)    
    return img

def prepare_mask_tif():
    df_summary = pd.read_csv(FN_MASK_CSV)
    df = pd.read_csv(FN_MASK_CSV, usecols=[0]).iloc[:]
    ids = pd.DataFrame(df)['ImageId'].unique()
    for image_id in ids[0:10]:        
        mask = image_mask_resized_from_summary(df_summary, image_id)
        im_path = "/data/train/AOI_3_Paris/RGB_PanSharpen_" + image_id + ".tif"
        # im_path = "./RGB-PanSharpen_AOI_3_Paris_img197.tif"
        out_path1 = "./out_path/mask/" + image_id + "_mask.tif"
        # im_path = "/data/train/AOI_3_Paris/RGB_PanSharpen_" + image_id + ".tif"


        ds = gdal.Open(im_path)
        driver = gdal.GetDriverByName("GTiff")
        ## get resized mask
        dst_ds1 = driver.Create(out_path1, 650, 650, (1), gdal.GDT_Byte)                          
        dst_ds1.GetRasterBand(1).WriteArray(mask)
        dst_ds1.GetRasterBand(1).ComputeStatistics(False)
        dst_ds1.SetProjection(ds.GetProjection())
        dst_ds1.SetGeoTransform(ds.GetGeoTransform())

        ## get resized img
        out_path2 = "./out_path/resized/" + image_id + ".tif"
        img2 = ds.ReadAsArray()
        img2 = resize_original_im(img2)
        dst_ds2 = driver.Create(out_path2,650,650,3,gdal.GDT_Byte)
        for i in range(1,img2.shape[2]+1):
            dst_ds2.GetRasterBand(i).WriteArray(img2[:,:,i-1])
            dst_ds2.GetRasterBand(i).ComputeStatistics(False)
        dst_ds2.SetProjection(ds.GetProjection())
        dst_ds2.SetGeoTransform(ds.GetGeoTransform())

def image_mask_resized_from_summary(df, image_id):
    im_mask = np.zeros((650, 650))
    for idx, row in df[df.ImageId == image_id].iterrows():
        shape_obj = shapely.wkt.loads(row.PolygonWKT_Pix)
        if shape_obj.exterior is not None:
            coords = list(shape_obj.exterior.coords)
            x = [round(float(pp[0])) for pp in coords]
            y = [round(float(pp[1])) for pp in coords]
            yy, xx = skimage.draw.polygon(y, x, (650, 650))
            im_mask[yy, xx] = 1

            interiors = shape_obj.interiors
            for interior in interiors:
                coords = list(interior.coords)
                x = [round(float(pp[0])) for pp in coords]
                y = [round(float(pp[1])) for pp in coords]
                yy, xx = skimage.draw.polygon(y, x, (650, 650))
                im_mask[yy, xx] = 0
    # im_mask = skimage.transform.resize(im_mask, (INPUT_SIZE, INPUT_SIZE))
    im_mask = (im_mask > 0.5).astype(np.uint8)
    return im_mask


prepare_mask_tif()
