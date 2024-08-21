from PIL import Image
import numpy as np
from utils import RGB_YCbCr as util
import torch
import numpy 

path = '../datasets/Sony/rgb/long/gt_00017_00_10s.png'
img = numpy.array(Image.open(path).convert('RGB'))
print(img.shape)
img = img[None,:,:,:]
ut = util()
YCrCb = ut.RGB2YCrCb(torch.tensor(img).permute(0,3,1,2))
np_img = YCrCb.squeeze(0).numpy()
Y = np_img[0:1,:,:]
# Y = Y[None,:,:]
Cr = np_img[1:2,:,:]
Cb = np_img[2:3,:,:]

print(Y.max(), Y.min())

Y += 50
Y = Y.clip(0, 255)
print(Y.max(), Y.min())

print(Y.shape, Cr.shape, Cb.shape)
YCrCb_fuse = numpy.concatenate((Y,Cr,Cb), 0)
RGB_fuse = ut.YCrCb2RGB(torch.tensor(YCrCb_fuse).unsqueeze(0))
RGB_fuse = RGB_fuse[0].permute(1,2,0).numpy()
RGB_fuse = Image.fromarray(RGB_fuse.astype(np.uint8))

RGB_fuse.save("test3.png") 



exit()

# read image
img = Image.open('experiments/test_diffusion_sid_240810_220225/results/test/0/10003_00_10s.ARW.png')
# convert to numy
img = np.array(img)   # BGR
img = img[:, :, [0, 1, 2]]
# convert to PIL
img = Image.fromarray(img)
# save image
img.save('experiments/test_diffusion_sid_240810_220225/results/test/0/10003_00_10s.ARW_order.png')