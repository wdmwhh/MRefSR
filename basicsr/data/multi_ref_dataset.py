from ast import literal_eval
import cv2
import glob
import mmcv
import numpy as np
import os.path as osp
import pandas as pd
import random
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

from .transforms import augment, mod_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class MultiRefMegaDepthDataset(data.Dataset):
    """Multi-References based MegaDepth dataset for super-resolution.


    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_in (str): Data root path for input image.
        dataroot_ref (str): Data root path for ref image.
        ann_file (str): Path for annotation file.
        gt_size (int): Cropped patched size for gt patches.
        use_flip (bool): Use horizontal and vertical flips.
        use_rot (bool): Use rotation (use transposing h and w for
            implementation).

        scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(MultiRefMegaDepthDataset, self).__init__()
        self.opt = opt

        self.in_folder, self.ref_folder = opt['dataroot_in'], opt[
            'dataroot_ref']
        self.ann_file = opt['ann_file']
        self.load_annotations()

    def load_annotations(self):
        self.samples = []
        df = pd.read_csv(self.ann_file, dtype={"scene":"string"})
        for i in range(len(df)):
            target, H, M1, M2, L1, L2, p0, p1, p2, p3, p4, p5, scene = df.loc[i].tolist()
            target = osp.join(self.in_folder, scene, target)
            references = [
                osp.join(self.in_folder, scene, H),
                osp.join(self.in_folder, scene, M1),
                osp.join(self.in_folder, scene, M2),
                osp.join(self.in_folder, scene, L1),
                osp.join(self.in_folder, scene, L2)]
            p0 = np.array(literal_eval(p0))
            p_refs = [
                np.array(literal_eval(p1)),
                np.array(literal_eval(p2)),
                np.array(literal_eval(p3)),
                np.array(literal_eval(p4)),
                np.array(literal_eval(p5))]
            self.samples.append((target, references, p0, p_refs))
        print(len(self.samples))

    def __getitem__(self, index):

        scale = self.opt['scale']

        # Load in and ref images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1] float32.
        in_path, ref_paths, p0, p_refs = self.samples[index]
        img_in = Image.open(in_path).convert('RGB')
        Refs = [Image.open(ref_path).convert('RGB') for ref_path in ref_paths]
        img_in = np.array(img_in).astype(np.float32) / 255.
        Refs = [np.array(img_ref).astype(np.float32) / 255. for img_ref in Refs]

        gt_h, gt_w = self.opt['gt_size'], self.opt['gt_size']
        img_in = img_in[p0[1]-gt_h//2:p0[1]+gt_h//2, p0[0]-gt_w//2:p0[0]+gt_w//2]
        Refs = [
            img_ref[p_ref[1]-gt_h//2:p_ref[1]+gt_h//2, p_ref[0]-gt_w//2:p_ref[0]+gt_w//2]
            for img_ref, p_ref in zip(Refs, p_refs)
        ]
        random.shuffle(Refs)

        # data augmentation
        imgs = augment([img_in]+Refs, self.opt['use_flip'], self.opt['use_rot'])
        img_in = imgs[0]
        Refs = imgs[1:]

        # downsample image using PIL bicubic kernel
        lq_h, lq_w = gt_h // scale, gt_w // scale

        img_in_pil = img_in * 255
        img_in_pil = Image.fromarray(img_in_pil.astype(np.uint8))
        img_in_lq = img_in_pil.resize((lq_w, lq_h), Image.BICUBIC)
        img_in_up = img_in_lq.resize((gt_w, gt_h), Image.BICUBIC)
        Refs_lq = []
        Refs_up = []
        for img_ref in Refs:
            img_ref_pil = img_ref * 255
            img_ref_pil = Image.fromarray(img_ref_pil.astype(np.uint8))
            img_ref_lq = img_ref_pil.resize((lq_w, lq_h), Image.BICUBIC)
            img_ref_up = img_ref_lq.resize((gt_w, gt_h), Image.BICUBIC)
            Refs_lq.append(img_ref_lq)
            Refs_up.append(img_ref_up)

        img_in_lq = np.array(img_in_lq).astype(np.float32) / 255.
        img_in_up = np.array(img_in_up).astype(np.float32) / 255.
        Refs_lq = [np.array(img_ref_lq).astype(np.float32) / 255. for img_ref_lq in Refs_lq]
        Refs_up = [np.array(img_ref_up).astype(np.float32) / 255. for img_ref_up in Refs_up]

        # HWC to CHW, numpy to tensor
        img_in, img_in_lq, img_in_up = img2tensor(  # noqa: E501
            [img_in, img_in_lq, img_in_up],
            bgr2rgb=False,
            float32=True)
        Refs = img2tensor(Refs, bgr2rgb=False, float32=True)
        Refs_lq = img2tensor(Refs_lq, bgr2rgb=False, float32=True)
        Refs_up = img2tensor(Refs_up, bgr2rgb=False, float32=True)
        Refs = torch.stack(Refs)
        Refs_lq = torch.stack(Refs_lq)
        Refs_up = torch.stack(Refs_up)

        return_dict = {
            'img_in': img_in,
            'img_in_lq': img_in_lq,
            'img_in_up': img_in_up,
            'img_ref_list': Refs,
            'img_ref_lq_list': Refs_lq,
            'img_ref_up_list': Refs_up,
        }

        return return_dict

    def __len__(self):
        return len(self.samples)


@DATASET_REGISTRY.register()
class MultiRefCUFEDSet(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.input_list = sorted(glob.glob(osp.join(opt['dataroot_in'], '*_0.png')) )
        self.ref_list1 = sorted(glob.glob(osp.join(opt['dataroot_ref'], '*_1.png')) )
        self.ref_list2 = sorted(glob.glob(osp.join(opt['dataroot_ref'], '*_2.png')) )
        self.ref_list3 = sorted(glob.glob(osp.join(opt['dataroot_ref'], '*_3.png')) )
        self.ref_list4 = sorted(glob.glob(osp.join(opt['dataroot_ref'], '*_4.png')) )
        self.ref_list5 = sorted(glob.glob(osp.join(opt['dataroot_ref'], '*_5.png')) )

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
        Refs = [img_ref1, img_ref2, img_ref3, img_ref4, img_ref5]
        ref_path = self.ref_list1[idx].replace('_1.png', '_multi.png')

        img_in = mod_crop(img_in, scale)
        img_in_gt = img_in.copy()

        img_in_h, img_in_w, _ = img_in.shape
        padding = True
        gt_h, gt_w = 500, 500
        img_in = mmcv.impad(
            img_in, shape=(gt_h, gt_w), pad_val=0)
        Refs = [
            mmcv.impad(img_ref, shape=(gt_h, gt_w), pad_val=0)
            for img_ref in Refs
        ]

        gt_h, gt_w, _ = img_in.shape
        # downsample image using PIL bicubic kernel
        lq_h, lq_w = gt_h // scale, gt_w // scale
        img_in_lq = Image.fromarray(img_in).resize((lq_w, lq_h), Image.BICUBIC)
        img_in_up = img_in_lq.resize((gt_w, gt_h), Image.BICUBIC)
        Refs_lq = []
        Refs_up = []
        for img_ref in Refs:
            img_ref_lq = Image.fromarray(img_ref).resize((lq_w, lq_h), Image.BICUBIC)
            img_ref_up = img_ref_lq.resize((gt_w, gt_h), Image.BICUBIC)
            Refs_lq.append(img_ref_lq)
            Refs_up.append(img_ref_up)

        img_in = img_in.astype(np.float32) / 255.
        img_in_gt = img_in_gt.astype(np.float32) / 255.
        img_in_lq = np.array(img_in_lq).astype(np.float32) / 255.
        img_in_up = np.array(img_in_up).astype(np.float32) / 255.
        Refs = [img_ref.astype(np.float32) / 255. for img_ref in Refs]
        Refs_lq = [np.array(img_ref_lq).astype(np.float32) / 255. for img_ref_lq in Refs_lq]
        Refs_up = [np.array(img_ref_up).astype(np.float32) / 255. for img_ref_up in Refs_up]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_in, img_in_lq, img_in_up, img_in_gt = img2tensor(  # noqa: E501
            [img_in, img_in_lq, img_in_up, img_in_gt],
            bgr2rgb=True,
            float32=True)
        Refs = img2tensor(Refs, bgr2rgb=True, float32=True)
        Refs_lq = img2tensor(Refs_lq, bgr2rgb=True, float32=True)
        Refs_up = img2tensor(Refs_up, bgr2rgb=True, float32=True)
        Refs = torch.stack(Refs)
        Refs_lq = torch.stack(Refs_lq)
        Refs_up = torch.stack(Refs_up)

        return_dict = {
            'img_in': img_in_gt,
            'img_in_lq': img_in_lq,
            'img_in_up': img_in_up,
            'img_ref_list': Refs,
            'img_ref_lq_list': Refs_lq,
            'img_ref_up_list': Refs_up,
            'lq_path': ref_path,
            'padding': padding,
            'original_size': (img_in_h, img_in_w),
        }

        return return_dict

