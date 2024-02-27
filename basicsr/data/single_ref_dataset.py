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
class SingleRefMegaDepthDataset(data.Dataset):
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
        super(SingleRefMegaDepthDataset, self).__init__()
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
        perm = np.random.permutation(5)
        img_in = Image.open(in_path).convert('RGB')
        img_ref = Image.open(ref_paths[perm[0]]).convert('RGB')
        img_in = np.array(img_in).astype(np.float32) / 255.
        img_ref = np.array(img_ref).astype(np.float32) / 255.

        gt_h, gt_w = self.opt['gt_size'], self.opt['gt_size']
        img_in = img_in[p0[1]-gt_h//2:p0[1]+gt_h//2, p0[0]-gt_w//2:p0[0]+gt_w//2]
        img_ref = img_ref[p_refs[perm[0]][1]-gt_h//2:p_refs[perm[0]][1]+gt_h//2, p_refs[perm[0]][0]-gt_w//2:p_refs[perm[0]][0]+gt_w//2]

        # data augmentation
        img_in, img_ref = augment([img_in, img_ref], self.opt['use_flip'], self.opt['use_rot'])

        # downsample image using PIL bicubic kernel
        lq_h, lq_w = gt_h // scale, gt_w // scale

        img_in_pil = img_in * 255
        img_in_pil = Image.fromarray(img_in_pil.astype(np.uint8))
        img_in_lq = img_in_pil.resize((lq_w, lq_h), Image.BICUBIC)
        img_in_up = img_in_lq.resize((gt_w, gt_h), Image.BICUBIC)
        img_ref_pil = img_ref * 255
        img_ref_pil = Image.fromarray(img_ref_pil.astype(np.uint8))
        img_ref_lq = img_ref_pil.resize((lq_w, lq_h), Image.BICUBIC)
        img_ref_up = img_ref_lq.resize((gt_w, gt_h), Image.BICUBIC)

        img_in_lq = np.array(img_in_lq).astype(np.float32) / 255.
        img_in_up = np.array(img_in_up).astype(np.float32) / 255.
        img_ref_lq = np.array(img_ref_lq).astype(np.float32) / 255.
        img_ref_up = np.array(img_ref_up).astype(np.float32) / 255.

        # HWC to CHW, numpy to tensor
        img_in, img_in_lq, img_in_up, img_ref, img_ref_lq, img_ref_up = img2tensor(  # noqa: E501
            [img_in, img_in_lq, img_in_up, img_ref, img_ref_lq, img_ref_up],
            bgr2rgb=False,
            float32=True)

        return_dict = {
            'img_in': img_in,
            'img_in_lq': img_in_lq,
            'img_in_up': img_in_up,
            'img_ref': img_ref,
            'img_ref_lq': img_ref_lq,
            'img_ref_up': img_ref_up,
        }

        return return_dict

    def __len__(self):
        return len(self.samples)
