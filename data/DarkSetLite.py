import torch
from torch.utils.data import Dataset
import numpy as np
import rawpy
import glob
import random
import os
from datetime import datetime
import time
from tqdm import tqdm
from torch import nn

class DarkData(Dataset):
    def __init__(self, patch_size=512, type="train", data_root='dataset/Sony/', pick_random=True, compress=True, first_n=None):
        self.patch_size   = patch_size                  # patch size for training
        self.type_code    = '0'                         # type of the dataset to be loaded. (0:train, 1:test, 2:validation)
        self.data_root    = data_root                   # dataset root
        self.n_burst      = 8                           # number of input images to be retrieved (fixed value 8)
        self.pick_random  = pick_random                 # True=pick input images randomly, False=pick first N input images
        self.input_dir    = self.data_root + 'short/'   # short (input) image path
        self.gt_dir       = self.data_root + 'long/'    # long (output/ground truth) image path
        #self.input_images = []
        #self.gt_images    = []
        self.train_ids    = []
        self.compress = compress
        self.first_n = first_n
        
        if type == 'train':
            self.type_code = '0'
        elif type == 'test':
            self.type_code = '1'
        elif type == 'validation':
            self.type_code = '2'
        else:
            raise Exception('Invalid type parameter given for data loader. Only <train>, <test> or <validation> values are accepted')
        
        if compress:
            self.compress_dataset()
        self.load_dataset()
        
        self.data_len = len(self.train_ids)
    
    def compress_dataset(self):
        # check if .npz files already present
        gt_fnames = glob.glob(self.gt_dir + '*.ARW.npz')
        in_fnames = glob.glob(self.input_dir + '*s.ARW.npz')        
        if len(gt_fnames) > 0 and len(in_fnames) > 0:
            print(f"WARNING: .npz files are alredy present in both {self.gt_dir} and {self.input_dir} ")
            print(f"WARNING: No compressin is done")
            return

        fnames = glob.glob(self.gt_dir + '*.ARW')
        # self.train_ids holds the list of ids for short-long image relationship
        ids = sorted([int(os.path.basename(train_fn)[0:5]) for train_fn in fnames])#[:1]
        fc = []
        t0 = time.time()
        for id in tqdm(ids, desc="Compressing"):
            # use train_id to get the list of output/ground truth images
            gt_files = glob.glob(self.gt_dir + '%05d_00*.ARW' % id)
            # get the first and the only image path for given train_id 
            gt_path = gt_files[0]            
            gt_raw = rawpy.imread(gt_path)
            gt_im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_im = np.float32(gt_im/65535.0)
            gt_im = np.minimum(gt_im, 1.0)
            gt_im = np.expand_dims(gt_im, axis=0)            
            np.savez_compressed(gt_path, data=gt_im)   
            # remove the path and get the file names only
            gt_fn = os.path.basename(gt_path)          
            gt_exposure = float(gt_fn[9:-5])

            in_files = sorted(glob.glob(self.input_dir + '%05d_*' % id + 's.ARW'))
            fc.append(len(in_files))
            for in_pth in in_files:
                raw = rawpy.imread(in_pth)
                in_fn = os.path.basename(in_pth) 
                in_exposure = float(in_fn[9:-5])                              
                ratio = min(gt_exposure / in_exposure, 300)                              
                im = self.transform_raw(raw.raw_image_visible, ratio)
                np.savez_compressed(in_pth, data=im)
                               
        t1 = time.time()
        print(f"{sum(fc)} files are compressed in {t1-t0} seconds")        


    def load_dataset(self):
        # get target set of image names in ground truth folder.
        train_fns = glob.glob(self.gt_dir + self.type_code + '*.ARW.npz')
        # print("GT", len(train_fns))
        
        # self.train_ids holds the list of ids for short-long image relationship
        if self.first_n:
            self.train_ids = sorted([int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns])[:self.first_n] # [17:20]
        else:
            self.train_ids = sorted([int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns])
                

        t0 = time.time()
        n_gt = len(train_fns)
        self.Y = np.zeros((n_gt, 1, 2848, 4256, 3), dtype=np.float32)
        self.file_counts = np.zeros((n_gt), dtype=np.uint8)
        self.gt_exposures = np.zeros((n_gt), dtype=np.float16)
        self.in_exposures = []
        self.gt_paths = [] 
        self.in_pths = [] 
        print("Estimating stats using npz files...")
        for idx,train_id in tqdm(enumerate(self.train_ids), desc="Loading..."):
            # use train_id to get the list of output/ground truth images
            gt_files = glob.glob(self.gt_dir + '%05d_00*.ARW.npz' % train_id)
            # get the first and the only image path for given train_id 
            gt_path = gt_files[0]
            self.Y[idx] = np.load(gt_path)["data"]
            self.gt_paths.append(gt_path)                      
            # remove the path and get the file names only
            gt_fn = os.path.basename(gt_path)          
            gt_exposure = float(gt_fn[9:-9])
            self.gt_exposures[idx] = gt_exposure
            in_files = sorted(glob.glob(self.input_dir + '%05d_*' % train_id + 's.ARW.npz'))
            self.in_pths += in_files            
            self.file_counts[idx] = len(in_files)
            for in_pth in in_files:
                in_fn = os.path.basename(in_pth) 
                in_exposure = float(in_fn[9:-9])               
                self.in_exposures.append(in_exposure)        
        t1 = time.time()
        self.in_exposures = np.array(self.in_exposures, dtype=np.float16)
        print(f"{len(self.in_exposures)} input file stats are loaded to ram in {t1-t0} seconds")
        print(f" Input files count {len(self.in_pths)} Ground truth files count {len(self.gt_paths)}")
        print(f" File counts {self.file_counts}")    
    
    """
    Input:
    Output: Length of data set
    """
    def __len__(self):
        return self.data_len
            

    def __getitem__(self, index):
        # a = time.time()
        # get the data set with given index
        train_id = self.train_ids[index]       
        # read ground truth
        gt_pth = self.gt_paths[index]
        # gt_image = np.load(gt_pth)["data"]          # 1, 2848, 4256, 3 
        gt_image = self.Y[index]          # 1, 2848, 4256, 3 

        # print(f" gt image load {time.time()-a}")
        
        # a = time.time()
        in_start = np.sum(self.file_counts[:index])
        in_end = in_start + self.file_counts[index]
        # get input exposure values for the current index
        exposures = self.in_exposures[in_start:in_end]
        # pick a random exposure from uniqu exposure values    
        in_exposure = np.random.choice(np.unique(exposures))
        # print("in_exposure", in_exposure)  

        # get ground truth exposure
        gt_exposure = self.gt_exposures[index]
        # print("gt_exposure", gt_exposure) 

        # calculate ratio
        ratio = min(gt_exposure / in_exposure, 300)

        # indices of input images with exposure value eq to in_exposure
        indices = np.where(exposures == in_exposure)[0]
        indices += in_start
      
        # we have at least self.n_burst input images
        if len(indices) >= self.n_burst:
            input_image_indices = np.random.choice(range(in_start, in_end), 
                                                   size=self.n_burst,
                                                   replace=False)            
        # we don't have enough input images
        # make sure we have burst_size images
        else:
            input_image_indices = np.random.choice(range(in_start, in_end), 
                                                   size=self.n_burst,
                                                   replace=True)   

        # print(f"making burst indices ready {time.time()-a}")         

        # a = time.time()
        # in_paths = [self.in_pths[idx] for idx in input_image_indices]
        # input_images = self.path_to_raws2(in_paths, ratio)
        in_paths = [self.in_pths[idx] for idx in input_image_indices]
        # N, 1, 1424, 2128, 4
        input_images = np.array([np.load(fn)["data"] for fn in in_paths])
        # print("np.max(input_images)", np.max(input_images), "np.min(input_images)", np.min(input_images), flush=True)


        # print(f"making burst ready {time.time()-a}")
        
        # print("len(indices) = %d, len(input_images) = %d,  self.n_burst = %d, train_id=%s" %(len(indices), len(input_images),  self.n_burst, train_id))
        # print("input_images[0].shape", input_images[0].shape )

        # 1:test, 2:validation
        if self.type_code == '1' or self.type_code == '2':
            # print(f"input_images.shape {input_images.shape}")   # 8, 1, 1424, 2128, 4     
            # print(f"gt_image.shape {gt_image.shape}")    # 1, 2848, 4256, 3
            input_patches = input_images
            input_patches = np.minimum(input_patches, 1.0)     
            input_patches = np.squeeze(input_patches, axis=1)   # 8, 1424, 2128, 4                 
            gt_patch = gt_image
            # resize down to input dimensions for UNET
            _, _, H, W , _ = input_images.shape 
            gt_patch = torch.from_numpy(gt_patch)
            # first to bchw then back to bhwc
            gt_patch = nn.functional.interpolate(gt_patch.permute(0,3,1,2), size=(H, W), mode='bilinear', align_corners=True).permute(0,2,3,1)
            gt_patch = gt_patch.numpy()            
            xx = 0
            yy = 0
        # 0:train
        else:
            # return input_images, gt_image, train_id, 0, 0
            #get patches out of images
            # a = time.time()
            input_patches, gt_patch, xx, yy = self.crop_samples(input_images, gt_image, self.patch_size)    
            # print(f"crop samples {time.time()-a}")
            #augment patches
            # a = time.time()
            input_patches, gt_patch = self.augment_samples(input_patches, gt_patch)
            # print(f"augment {time.time()-a}")        
            #shuffle

            # a = time.time()
            input_patches = self.shuffle_samples(input_patches)
            # print(f"shuffle samples {time.time()-a}")                       
            #shift
            # input_patches = self.shift_samples(input_patches,max_err=8)            
            
            # a = time.time()
            input_patches = np.squeeze(input_patches, axis=1)
            input_patches = np.minimum(input_patches, 1.0)
            # print(f"minimum patches {time.time()-a}")
            

        input_patches = self.numpy_to_torch(input_patches)        
        gt_patch = self.numpy_to_torch(gt_patch)        
        gt_patch = np.squeeze(gt_patch, axis=0)
        # print("input_patches.shape", input_patches.shape) # [8, 4, 128, 128]
        # print("gt_patch.shape", gt_patch.shape)           # [3, 256, 256]
        # print("train_id", train_id)
        # print("xx", xx)
        # print("yy", yy)
        # print("ratio", ratio)
        ret = {}
        ret['gt_image'] = gt_patch
        _, _, H, W = input_patches.shape
        ret['cond_image'] =  input_patches.view(-1, H, W)
        ret['path'] = gt_pth
        ret['imgid']   = train_id     
        return ret
        # return input_patches, gt_patch, train_id, xx, yy, ratio

        
    """
    Input: object returned from rawpy.imread()
    Output: numpy array in shape (1424, 2128, 4)
    """
    def pack_raw_rgbg(self, raw):
        im = raw.astype(np.float32)   # shape of (2848, 4256)
        im = np.maximum(im - 512, 0) / (16383 - 512)    # subtract the black level
        im = np.expand_dims(im, axis=2)                 # shape of (2848, 4256, 1)

        img_shape = im.shape # (H, W, 1)
        H = img_shape[0]
        W = img_shape[1]
        
        # Pack into 4 channels
        red     = im[0:H:2,0:W:2,:]
        green_1 = im[0:H:2,1:W:2,:]
        blue    = im[1:H:2,1:W:2,:]
        green_2 = im[1:H:2,0:W:2,:]
        
        # Final shape: (1424, 2128, 4)
        out = np.concatenate((red, green_1, blue, green_2), axis=2)
        
        return out

    """
    Input: image path
    Output: raw image
    """
    def path_to_raw(self, pth, ratio, k=1):
        # ger raw output image and add into the input images list
        raw = rawpy.imread(pth)
        raw = self.pack_raw_rgbg(raw) * (ratio * k)
        raw = np.expand_dims(raw, 0)
        raw = np.minimum(raw, 1.0)
        
        return raw
    
    """
    Input: image path
    Output: raw image
    """
    def transform_raw(self, raw, ratio=1, k=1):       
        raw = self.pack_raw_rgbg(raw) * (ratio * k)
        raw = np.minimum(raw, 1.0)        
        raw = np.expand_dims(raw, 0)        
        return raw    


    """
    Input: numpy array (B x H x W x C)
    Output: torch tensor (B x C x H x W)
    """
    def numpy_to_torch(self, image):
        #print("image.shape : ", image.shape)
        image = image.transpose((0, 3, 1, 2))
        torch_tensor = torch.from_numpy(image.copy())
        return torch_tensor

    # ========        AUGMENTATIONS    ==========
    
    def shuffle_samples(self, input_images):
        indices = np.arange(input_images.shape[0])
        np.random.shuffle(indices)
        return input_images[indices]


    def augment_samples(self, input_images, gt_image):
        t1, t2, t3 = False, False, False
        if np.random.randint(2, size=1)[0] == 1:  # random flip
            gt_image = np.flip(gt_image, axis=1)
            #gt_image_raw = np.flip(gt_image_raw, axis=1)
            t1 = True
        if np.random.randint(2, size=1)[0] == 1:
            gt_image = np.flip(gt_image, axis=2)
            #gt_image_raw = np.flip(gt_image_raw, axis=2)
            t2 = True
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            gt_image = np.transpose(gt_image, (0, 2, 1, 3))
            #gt_image_raw = np.transpose(gt_image_raw, (0, 2, 1, 3))
            t3 = True

        new_images = []
        for i in range(len(input_images)):
            img = input_images[i]
            if t1 == True:
                img = np.flip(img, axis=1)
            if t2 == True:
                img = np.flip(img, axis=2)
            if t3 == True:
                img = np.transpose(img, (0, 2, 1, 3))
            new_images.append(img)

        return np.array(new_images), gt_image

    def shift_samples(self, input_images, max_err=4):
        shifted_images = []
        error_x = np.random.randint(0, max_err)
        error_y = np.random.randint(0, max_err)        
        for i in range(0,len(input_images)):
            img = input_images[i][0,:,:,:]
            shifted = np.pad(img, ((max_err//2, max_err//2), (max_err//2,max_err//2), (0, 0)), mode='reflect')
            shifted = shifted[error_y:shifted.shape[0]-(max_err-error_y), error_x:shifted.shape[1]-(max_err-error_x), :]
            shifted_images.append(np.expand_dims(shifted, 0))
            
        return np.array(shifted_images)

    def crop_samples(self, input_images, gt_image, ps=256, raw_ratio=2):
        _, _, H, W, _ = input_images.shape  # 8, 4, H, W
        xx = np.random.randint(0, W - ps)
        yy = np.random.randint(0, H - ps)
        input_patches = input_images[:, :, yy:yy + ps, xx:xx + ps, :]
        gt_patch = gt_image[:, yy*raw_ratio:yy*raw_ratio +ps*raw_ratio, xx*raw_ratio:xx*raw_ratio +ps*raw_ratio, :]
        # resize from 512,512 to 256, 256 for UNET
        gt_patch = torch.from_numpy(gt_patch)
        gt_patch = nn.functional.interpolate(gt_patch.permute(0,3,1,2), size=(ps, ps), mode='bilinear', align_corners=True).permute(0,2,3,1)
        gt_patch = gt_patch.numpy()
        return input_patches, gt_patch, xx, yy
