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

from mmsr.data.transforms import augment, mod_crop, totensor


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
        img_in, img_in_lq, img_in_up = totensor(  # noqa: E501
            [img_in, img_in_lq, img_in_up],
            bgr2rgb=False,
            float32=True)
        Refs = totensor(Refs, bgr2rgb=False, float32=True)
        Refs_lq = totensor(Refs_lq, bgr2rgb=False, float32=True)
        Refs_up = totensor(Refs_up, bgr2rgb=False, float32=True)
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
        img_in, img_in_lq, img_in_up, img_in_gt = totensor(  # noqa: E501
            [img_in, img_in_lq, img_in_up, img_in_gt],
            bgr2rgb=True,
            float32=True)
        Refs = totensor(Refs, bgr2rgb=True, float32=True)
        Refs_lq = totensor(Refs_lq, bgr2rgb=True, float32=True)
        Refs_up = totensor(Refs_up, bgr2rgb=True, float32=True)
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


class MultiRefMegaDepthTestSet(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.img_folder = osp.join(opt['dataroot_in'], 'test1600Pairs')
        self.ann_file = opt['ann_file']
        self.load_annotations()

    def __len__(self):
        return len(self.samples)

    def load_annotations(self):
        self.samples = []
        df = pd.read_csv(self.ann_file, dtype={"scene":"string"})
        for i in range(len(df)):
            target, refs = df.loc[i].tolist()
            refs = literal_eval(refs)
            target = osp.join(self.img_folder, target)
            references = [osp.join(self.img_folder, ref) for ref in refs]
            self.samples.append((target, references))

    def __getitem__(self, idx):
        scale = self.opt['scale']

        # Load in and ref images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1] float32.
        in_path, ref_paths = self.samples[idx]
        img_in = cv2.imread(in_path)
        Refs = [cv2.imread(ref_path) for ref_path in ref_paths]

        ref_path = in_path.replace('.jpg', '_mrsr.png')

        img_in = mod_crop(img_in, scale)
        img_in_gt = img_in.copy()

        img_in_h, img_in_w, _ = img_in.shape
        """
        Refs = [
            img_ref.transpose(1, 0, 2)
            if (img_in_h>img_in_w) != (img_ref.shape[0]>img_ref.shape[0])
            else img_ref
            for img_ref in Refs
        ]
        """
        padding = True
        gt_h, gt_w = img_in_h, img_in_w
        for img_ref in Refs:
            im_h, im_w, _ = img_ref.shape
            gt_h = max(gt_h, (im_h+3)//4*4)
            gt_w = max(gt_w, (im_w+3)//4*4)
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
        img_in, img_in_lq, img_in_up, img_in_gt = totensor(  # noqa: E501
            [img_in, img_in_lq, img_in_up, img_in_gt],
            bgr2rgb=True,
            float32=True)
        Refs = totensor(Refs, bgr2rgb=True, float32=True)
        Refs_lq = totensor(Refs_lq, bgr2rgb=True, float32=True)
        Refs_up = totensor(Refs_up, bgr2rgb=True, float32=True)
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


class MultiRefSun80Set(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.num_ref = opt['num_ref']
        self.input_list = sorted(glob.glob(osp.join(opt['dataroot_in'], 'Sun_Hays_SR_groundtruth/*.jpg')) )
        self.ref_folders = osp.join(opt['dataroot_ref'], 'Sun_Hays_SR_scenematches/')

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        scale = self.opt['scale']

        in_path = self.input_list[idx]
        img_in = cv2.imread(in_path)
        HR_name = osp.basename(in_path)
        ref_folder = osp.join(self.ref_folders, HR_name)
        ref_paths = sorted(glob.glob(osp.join(ref_folder, '*.jpg')) )
        ref_paths = ref_paths[::20//self.num_ref]
        Refs = [cv2.imread(ref_path) for ref_path in ref_paths]

        ref_path = in_path.replace('.jpg', '_mrsr.png')

        img_in = mod_crop(img_in, scale)
        img_in_gt = img_in.copy()

        img_in_h, img_in_w, _ = img_in.shape
        """
        Refs = [
            img_ref.transpose(1, 0, 2)
            if (img_in_h>img_in_w) != (img_ref.shape[0]>img_ref.shape[0])
            else img_ref
            for img_ref in Refs
        ]
        """
        padding = True
        gt_h, gt_w = img_in_h, img_in_w
        for img_ref in Refs:
            im_h, im_w, _ = img_ref.shape
            gt_h = max(gt_h, (im_h+3)//4*4)
            gt_w = max(gt_w, (im_w+3)//4*4)
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
        img_in, img_in_lq, img_in_up, img_in_gt = totensor(  # noqa: E501
            [img_in, img_in_lq, img_in_up, img_in_gt],
            bgr2rgb=True,
            float32=True)
        Refs = totensor(Refs, bgr2rgb=True, float32=True)
        Refs_lq = totensor(Refs_lq, bgr2rgb=True, float32=True)
        Refs_up = totensor(Refs_up, bgr2rgb=True, float32=True)
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


class MultiRefMegaDepthv3TestSet(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.data_folder = opt['data_folder']
        self.input_list = sorted(glob.glob(osp.join(self.data_folder, "*")) )

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        scale = self.opt['scale']

        folder_path = self.input_list[idx]
        HR_name = osp.basename(folder_path)
        in_path = osp.join(folder_path, HR_name[5:])
        img_in = cv2.imread(in_path)
        ref_paths = sorted(glob.glob(osp.join(folder_path, "*")) )
        Refs = [cv2.imread(ref_path) for ref_path in ref_paths if ref_path != in_path]

        ref_path = HR_name.split('.')[0] + '_mrsr.png'

        img_in = mod_crop(img_in, scale)
        img_in_gt = img_in.copy()

        img_in_h, img_in_w, _ = img_in.shape
        """
        Refs = [
            img_ref.transpose(1, 0, 2)
            if (img_in_h>img_in_w) != (img_ref.shape[0]>img_ref.shape[0])
            else img_ref
            for img_ref in Refs
        ]
        """
        padding = True
        gt_h, gt_w = img_in_h, img_in_w
        for img_ref in Refs:
            im_h, im_w, _ = img_ref.shape
            gt_h = max(gt_h, (im_h+3)//4*4)
            gt_w = max(gt_w, (im_w+3)//4*4)
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
        img_in, img_in_lq, img_in_up, img_in_gt = totensor(  # noqa: E501
            [img_in, img_in_lq, img_in_up, img_in_gt],
            bgr2rgb=True,
            float32=True)
        Refs = totensor(Refs, bgr2rgb=True, float32=True)
        Refs_lq = totensor(Refs_lq, bgr2rgb=True, float32=True)
        Refs_up = totensor(Refs_up, bgr2rgb=True, float32=True)
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


class MultiRefManga109Set(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.num_ref = opt['num_ref']
        self.input_list = sorted(glob.glob(osp.join(opt['dataroot_in'], '*.png')) )

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        scale = self.opt['scale']

        in_path = self.input_list[idx]
        img_in = cv2.imread(in_path)
        ref_paths = [ref_path for ref_path in self.input_list if ref_path != in_path]
        ref_paths = random.sample(ref_paths, self.num_ref)
        Refs = [cv2.imread(ref_path) for ref_path in ref_paths]

        ref_path = in_path.replace('.png', '_mrsr.png')

        img_in = mod_crop(img_in, scale)
        img_in_gt = img_in.copy()

        img_in_h, img_in_w, _ = img_in.shape
        """
        Refs = [
            img_ref.transpose(1, 0, 2)
            if (img_in_h>img_in_w) != (img_ref.shape[0]>img_ref.shape[0])
            else img_ref
            for img_ref in Refs
        ]
        """
        padding = True
        gt_h, gt_w = img_in_h, img_in_w
        for img_ref in Refs:
            im_h, im_w, _ = img_ref.shape
            gt_h = max(gt_h, (im_h+3)//4*4)
            gt_w = max(gt_w, (im_w+3)//4*4)
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
        img_in, img_in_lq, img_in_up, img_in_gt = totensor(  # noqa: E501
            [img_in, img_in_lq, img_in_up, img_in_gt],
            bgr2rgb=True,
            float32=True)
        Refs = totensor(Refs, bgr2rgb=True, float32=True)
        Refs_lq = totensor(Refs_lq, bgr2rgb=True, float32=True)
        Refs_up = totensor(Refs_up, bgr2rgb=True, float32=True)
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


class MultiRefWRSRSet(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.input_list = sorted(glob.glob(osp.join(opt['dataroot_in'], '*.png')) )
        self.ref_list = sorted(glob.glob(osp.join(opt['dataroot_ref'], '*.png')) )

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
            'img_ref_list': img_ref.unsqueeze(0),
            'img_ref_lq_list': img_ref_lq.unsqueeze(0),
            'img_ref_up_list': img_ref_up.unsqueeze(0),
            'lq_path': ref_path,
            'padding': padding,
            'original_size': (img_in_h, img_in_w),
        }

        return return_dict


class CUFEDSet_multi(data.Dataset):
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
            'img_ref_list': img_ref.unsqueeze(0),
            'img_ref_lq_list': img_ref_lq.unsqueeze(0),
            'img_ref_up_list': img_ref_up.unsqueeze(0),
            'lq_path': ref_path,
            'padding': padding,
            'original_size': (img_in_h, img_in_w),
        }

        return return_dict



