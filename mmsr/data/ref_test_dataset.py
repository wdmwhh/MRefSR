import cv2
import os
import numpy as np
import glob
import random
import mmcv
import torch
from PIL import Image
from torch.utils.data import Dataset

from mmsr.data.transforms import mod_crop, totensor


class CUFEDSet(Dataset):
    def __init__(self, opt, ref_level=1):
        super().__init__()
        self.opt = opt
        self.input_list = sorted(glob.glob(os.path.join(opt['dataroot_in'], '*_0.png')) )
        self.ref_list = sorted(glob.glob(os.path.join(opt['dataroot_ref'], f'*_{ref_level}.png')) )

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        scale = self.opt['scale']

        img_in = cv2.imread(self.input_list[idx])
        img_ref = cv2.imread(self.ref_list[idx])
        ref_path = self.ref_list[idx]

        img_in = mod_crop(img_in, scale)
        img_in_gt = img_in.copy()
        img_ref = mod_crop(img_ref, scale)
        img_in_h, img_in_w, _ = img_in.shape
        img_ref_h, img_ref_w, _ = img_ref.shape
        padding = False

        if img_in_h != img_ref_h or img_in_w != img_ref_w:
            padding = True
            target_h = max(img_in_h, img_ref_h)
            target_w = max(img_in_w, img_ref_w)
            img_in = mmcv.impad(
                img_in, shape=(target_h, target_w), pad_val=0)
            img_ref = mmcv.impad(
                img_ref, shape=(target_h, target_w), pad_val=0)

        gt_h, gt_w, _ = img_in.shape
        # downsample image using PIL bicubic kernel
        lq_h, lq_w = gt_h // scale, gt_w // scale
        img_in_lq = Image.fromarray(img_in).resize((lq_w, lq_h), Image.BICUBIC)
        img_ref_lq = Image.fromarray(img_ref).resize((lq_w, lq_h), Image.BICUBIC)
        img_in_up = img_in_lq.resize((gt_w, gt_h), Image.BICUBIC)
        img_ref_up = img_ref_lq.resize((gt_w, gt_h), Image.BICUBIC)

        img_in = img_in.astype(np.float32) / 255.
        img_ref = img_ref.astype(np.float32) / 255.
        img_in_gt = img_in_gt.astype(np.float32) / 255.
        img_in_lq = np.array(img_in_lq).astype(np.float32) / 255.
        img_in_up = np.array(img_in_up).astype(np.float32) / 255.
        img_ref_lq = np.array(img_ref_lq).astype(np.float32) / 255.
        img_ref_up = np.array(img_ref_up).astype(np.float32) / 255.

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_in, img_in_lq, img_in_up, img_ref, img_ref_lq, img_ref_up, img_in_gt = totensor(  # noqa: E501
            [img_in, img_in_lq, img_in_up, img_ref, img_ref_lq, img_ref_up, img_in_gt],
            bgr2rgb=True,
            float32=True)

        return_dict = {
            'img_in': img_in_gt,
            'img_in_lq': img_in_lq,
            'img_in_up': img_in_up,
            'img_ref': img_ref,
            'img_ref_lq': img_ref_lq,
            'img_ref_up': img_ref_up,
            'lq_path': ref_path,
            'padding': padding,
            'original_size': (img_in_h, img_in_w),
        }

        return return_dict


class CUFEDSet_multi(Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.input_list = sorted(glob.glob(os.path.join(opt['dataroot_in'], '*_0.png')) )
        self.ref_list1 = sorted(glob.glob(os.path.join(opt['dataroot_ref'], '*_1.png')) )
        self.ref_list2 = sorted(glob.glob(os.path.join(opt['dataroot_ref'], '*_2.png')) )
        self.ref_list3 = sorted(glob.glob(os.path.join(opt['dataroot_ref'], '*_3.png')) )
        self.ref_list4 = sorted(glob.glob(os.path.join(opt['dataroot_ref'], '*_4.png')) )
        self.ref_list5 = sorted(glob.glob(os.path.join(opt['dataroot_ref'], '*_5.png')) )

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        scale = self.opt['scale']

        img_in = cv2.imread(self.input_list[idx])
        img_ref1 = cv2.imread(self.ref_list1[idx])
        img_ref2 = cv2.imread(self.ref_list2[idx])
        img_ref3 = cv2.imread(self.ref_list3[idx])
        img_ref4 = cv2.imread(self.ref_list4[idx])
        img_ref5 = cv2.imread(self.ref_list5[idx])
        ref_path = self.ref_list1[idx].replace('_1.png', '_multi.png')

        img_in = mod_crop(img_in, scale)
        img_in_gt = img_in.copy()

        Refs = [img_ref1, img_ref2, img_ref3, img_ref4, img_ref5]
        h_new = 500
        w_new = 500
        Refs_pad = []
        for Ref in Refs:
            Ref_pad = np.zeros((h_new, w_new, 3), dtype=np.uint8)
            h, w, _ = Ref.shape
            Ref_pad[:h, :w, :] = Ref
            Refs_pad.append(Ref_pad)
        img_ref = cv2.vconcat(Refs_pad)
        img_ref = mod_crop(img_ref, scale)
        img_in_h, img_in_w, _ = img_in.shape
        img_ref_h, img_ref_w, _ = img_ref.shape
        padding = False

        if img_in_h != img_ref_h or img_in_w != img_ref_w:
            padding = True
            target_h = max(img_in_h, img_ref_h)
            target_w = max(img_in_w, img_ref_w)
            img_in = mmcv.impad(
                img_in, shape=(target_h, target_w), pad_val=0)
            img_ref = mmcv.impad(
                img_ref, shape=(target_h, target_w), pad_val=0)

        gt_h, gt_w, _ = img_in.shape
        # downsample image using PIL bicubic kernel
        lq_h, lq_w = gt_h // scale, gt_w // scale
        img_in_lq = Image.fromarray(img_in).resize((lq_w, lq_h), Image.BICUBIC)
        img_ref_lq = Image.fromarray(img_ref).resize((lq_w, lq_h), Image.BICUBIC)
        img_in_up = img_in_lq.resize((gt_w, gt_h), Image.BICUBIC)
        img_ref_up = img_ref_lq.resize((gt_w, gt_h), Image.BICUBIC)

        img_in = img_in.astype(np.float32) / 255.
        img_ref = img_ref.astype(np.float32) / 255.
        img_in_gt = img_in_gt.astype(np.float32) / 255.
        img_in_lq = np.array(img_in_lq).astype(np.float32) / 255.
        img_in_up = np.array(img_in_up).astype(np.float32) / 255.
        img_ref_lq = np.array(img_ref_lq).astype(np.float32) / 255.
        img_ref_up = np.array(img_ref_up).astype(np.float32) / 255.

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_in, img_in_lq, img_in_up, img_ref, img_ref_lq, img_ref_up, img_in_gt = totensor(  # noqa: E501
            [img_in, img_in_lq, img_in_up, img_ref, img_ref_lq, img_ref_up, img_in_gt],
            bgr2rgb=True,
            float32=True)

        return_dict = {
            'img_in': img_in_gt,
            'img_in_lq': img_in_lq,
            'img_in_up': img_in_up,
            'img_ref': img_ref,
            'img_ref_lq': img_ref_lq,
            'img_ref_up': img_ref_up,
            'lq_path': ref_path,
            'padding': padding,
            'original_size': (img_in_h, img_in_w),
        }

        return return_dict


class Urban100Set(Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.input_list = sorted(glob.glob(os.path.join(opt['dataroot_in'], '*.png')) )

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        scale = self.opt['scale']

        img_in = cv2.imread(self.input_list[idx])
        ref_path = self.input_list[idx]

        img_in = mod_crop(img_in, scale)
        img_in_gt = img_in.copy()
        img_in_h, img_in_w, _ = img_in.shape
        gt_h, gt_w, _ = img_in.shape
        # downsample image using PIL bicubic kernel
        lq_h, lq_w = gt_h // scale, gt_w // scale
        img_in_lq = Image.fromarray(img_in).resize((lq_w, lq_h), Image.BICUBIC)
        img_ref = np.array(img_in_lq).copy()
        img_ref_h, img_ref_w, _ = img_ref.shape
        padding = False

        if img_in_h != img_ref_h or img_in_w != img_ref_w:
            padding = True
            target_h = max(img_in_h, img_ref_h)
            target_w = max(img_in_w, img_ref_w)
            img_in = mmcv.impad(
                img_in, shape=(target_h, target_w), pad_val=0)
            img_ref = mmcv.impad(
                img_ref, shape=(target_h, target_w), pad_val=0)

        img_ref_lq = Image.fromarray(img_ref).resize((lq_w, lq_h), Image.BICUBIC)
        img_in_up = img_in_lq.resize((gt_w, gt_h), Image.BICUBIC)
        img_ref_up = img_ref_lq.resize((gt_w, gt_h), Image.BICUBIC)

        img_in = img_in.astype(np.float32) / 255.
        img_ref = img_ref.astype(np.float32) / 255.
        img_in_gt = img_in_gt.astype(np.float32) / 255.
        img_in_lq = np.array(img_in_lq).astype(np.float32) / 255.
        img_in_up = np.array(img_in_up).astype(np.float32) / 255.
        img_ref_lq = np.array(img_ref_lq).astype(np.float32) / 255.
        img_ref_up = np.array(img_ref_up).astype(np.float32) / 255.

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_in, img_in_lq, img_in_up, img_ref, img_ref_lq, img_ref_up, img_in_gt = totensor(  # noqa: E501
            [img_in, img_in_lq, img_in_up, img_ref, img_ref_lq, img_ref_up, img_in_gt],
            bgr2rgb=True,
            float32=True)

        return_dict = {
            'img_in': img_in_gt,
            'img_in_lq': img_in_lq,
            'img_in_up': img_in_up,
            'img_ref': img_ref,
            'img_ref_lq': img_ref_lq,
            'img_ref_up': img_ref_up,
            'lq_path': ref_path,
            'padding': padding,
            'original_size': (img_in_h, img_in_w),
        }

        return return_dict


class Sun80Set(Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.input_list = sorted(glob.glob(os.path.join(opt['dataroot_in'], 'Sun_Hays_SR_groundtruth/*.jpg')) )
        self.ref_folders = os.path.join(opt['dataroot_ref'], 'Sun_Hays_SR_scenematches/')

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        scale = self.opt['scale']

        img_in = cv2.imread(self.input_list[idx])
        HR_name = os.path.basename(self.input_list[idx])
        ref_folder = os.path.join(self.ref_folders, HR_name)
        ref_list = sorted(glob.glob(os.path.join(ref_folder, '*.jpg')))
        ref_path = random.sample(ref_list, 1)[0]
        img_ref = cv2.imread(ref_path)

        img_in = mod_crop(img_in, scale)
        img_in_gt = img_in.copy()
        img_ref = mod_crop(img_ref, scale)
        img_in_h, img_in_w, _ = img_in.shape
        img_ref_h, img_ref_w, _ = img_ref.shape
        padding = False

        if img_in_h != img_ref_h or img_in_w != img_ref_w:
            padding = True
            target_h = max(img_in_h, img_ref_h)
            target_w = max(img_in_w, img_ref_w)
            img_in = mmcv.impad(
                img_in, shape=(target_h, target_w), pad_val=0)
            img_ref = mmcv.impad(
                img_ref, shape=(target_h, target_w), pad_val=0)

        gt_h, gt_w, _ = img_in.shape
        # downsample image using PIL bicubic kernel
        lq_h, lq_w = gt_h // scale, gt_w // scale
        img_in_lq = Image.fromarray(img_in).resize((lq_w, lq_h), Image.BICUBIC)
        img_ref_lq = Image.fromarray(img_ref).resize((lq_w, lq_h), Image.BICUBIC)
        img_in_up = img_in_lq.resize((gt_w, gt_h), Image.BICUBIC)
        img_ref_up = img_ref_lq.resize((gt_w, gt_h), Image.BICUBIC)

        img_in = img_in.astype(np.float32) / 255.
        img_ref = img_ref.astype(np.float32) / 255.
        img_in_gt = img_in_gt.astype(np.float32) / 255.
        img_in_lq = np.array(img_in_lq).astype(np.float32) / 255.
        img_in_up = np.array(img_in_up).astype(np.float32) / 255.
        img_ref_lq = np.array(img_ref_lq).astype(np.float32) / 255.
        img_ref_up = np.array(img_ref_up).astype(np.float32) / 255.

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_in, img_in_lq, img_in_up, img_ref, img_ref_lq, img_ref_up, img_in_gt = totensor(  # noqa: E501
            [img_in, img_in_lq, img_in_up, img_ref, img_ref_lq, img_ref_up, img_in_gt],
            bgr2rgb=True,
            float32=True)

        return_dict = {
            'img_in': img_in_gt,
            'img_in_lq': img_in_lq,
            'img_in_up': img_in_up,
            'img_ref': img_ref,
            'img_ref_lq': img_ref_lq,
            'img_ref_up': img_ref_up,
            'lq_path': ref_path,
            'padding': padding,
            'original_size': (img_in_h, img_in_w),
        }

        return return_dict


class Manga109Set(Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.input_list = sorted(glob.glob(os.path.join(opt['dataroot_in'], '*.png')) )

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        scale = self.opt['scale']

        img_in = cv2.imread(self.input_list[idx])
        ref_path = random.sample(self.input_list, 1)[0]
        while ref_path == self.input_list[idx]:
            ref_path = random.sample(self.input_list, 1)[0]
        img_ref = cv2.imread(ref_path)

        img_in = mod_crop(img_in, scale)
        img_in_gt = img_in.copy()
        img_ref = mod_crop(img_ref, scale)
        img_in_h, img_in_w, _ = img_in.shape
        img_ref_h, img_ref_w, _ = img_ref.shape
        padding = False

        if img_in_h != img_ref_h or img_in_w != img_ref_w:
            padding = True
            target_h = max(img_in_h, img_ref_h)
            target_w = max(img_in_w, img_ref_w)
            img_in = mmcv.impad(
                img_in, shape=(target_h, target_w), pad_val=0)
            img_ref = mmcv.impad(
                img_ref, shape=(target_h, target_w), pad_val=0)

        gt_h, gt_w, _ = img_in.shape
        # downsample image using PIL bicubic kernel
        lq_h, lq_w = gt_h // scale, gt_w // scale
        img_in_lq = Image.fromarray(img_in).resize((lq_w, lq_h), Image.BICUBIC)
        img_ref_lq = Image.fromarray(img_ref).resize((lq_w, lq_h), Image.BICUBIC)
        img_in_up = img_in_lq.resize((gt_w, gt_h), Image.BICUBIC)
        img_ref_up = img_ref_lq.resize((gt_w, gt_h), Image.BICUBIC)

        img_in = img_in.astype(np.float32) / 255.
        img_ref = img_ref.astype(np.float32) / 255.
        img_in_gt = img_in_gt.astype(np.float32) / 255.
        img_in_lq = np.array(img_in_lq).astype(np.float32) / 255.
        img_in_up = np.array(img_in_up).astype(np.float32) / 255.
        img_ref_lq = np.array(img_ref_lq).astype(np.float32) / 255.
        img_ref_up = np.array(img_ref_up).astype(np.float32) / 255.

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_in, img_in_lq, img_in_up, img_ref, img_ref_lq, img_ref_up, img_in_gt = totensor(  # noqa: E501
            [img_in, img_in_lq, img_in_up, img_ref, img_ref_lq, img_ref_up, img_in_gt],
            bgr2rgb=True,
            float32=True)

        return_dict = {
            'img_in': img_in_gt,
            'img_in_lq': img_in_lq,
            'img_in_up': img_in_up,
            'img_ref': img_ref,
            'img_ref_lq': img_ref_lq,
            'img_ref_up': img_ref_up,
            'lq_path': ref_path,
            'padding': padding,
            'original_size': (img_in_h, img_in_w),
        }

        return return_dict


