from tf_unet import unet, util, image_util
import numpy as np

#preparing data loading
data_provider = image_util.ImageDataProvider(search_path="out_path/resized/",data_suffix=u"/resized/*.tif",mask_suffix=u"out_path/mask/*_mask.tif")

#setup & training
output_path = "./out_path/model"
net = unet.Unet(layers=3, features_root=64, channels=3, n_class=2)
trainer = unet.Trainer(net)
path = trainer.train(data_provider, output_path, training_iters=32, epochs=10, restore=True, dropout=0.75, display_step=2)
