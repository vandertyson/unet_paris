from tf_unet import unet, util, image_util
import numpy as np
import gdal

output_path = "out_path/model/model.cpkt"
net = unet.Unet(layers=3, features_root=64, channels=3, n_class=2)

#verification
img_path = "./RGB-PanSharpen_AOI_3_Paris_img197.tif"
ds = gdal.Open(img_path)
arr = ds.ReadAsArray()
arr = np.reshape(arr,(1,650,650,3))

prediction = net.predict(output_path, arr)
print(prediction)
# print(prediction)
# unet.error_rate(prediction, util.crop_to_shape(label, prediction.shape))

# img = util.combine_img_prediction(data, label, prediction)
# util.save_image(img, "prediction.jpg")
