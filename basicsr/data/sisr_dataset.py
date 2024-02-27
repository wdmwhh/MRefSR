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

from .data_util import (paired_paths_from_ann_file,
                            paired_paths_from_folder, paired_paths_from_lmdb)
from .transforms import augment, mod_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class SISRMegaDepthDataset(data.Dataset):
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
        super(SISRMegaDepthDataset, self).__init__()
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
        img_in = np.array(img_in).astype(np.float32) / 255.

        gt_h, gt_w = self.opt['gt_size'], self.opt['gt_size']
        img_in = img_in[p0[1]-gt_h//2:p0[1]+gt_h//2, p0[0]-gt_w//2:p0[0]+gt_w//2]

        # data augmentation
        img_in = augment(img_in, self.opt['use_flip'], self.opt['use_rot'])

        # downsample image using PIL bicubic kernel
        lq_h, lq_w = gt_h // scale, gt_w // scale

        img_in_pil = img_in * 255
        img_in_pil = Image.fromarray(img_in_pil.astype(np.uint8))
        img_in_lq = img_in_pil.resize((lq_w, lq_h), Image.BICUBIC)
        img_in_lq = np.array(img_in_lq).astype(np.float32) / 255.

        # HWC to CHW, numpy to tensor
        img_in, img_in_lq = img2tensor(  # noqa: E501
            [img_in, img_in_lq],
            bgr2rgb=False,
            float32=True)

        return_dict = {
            'gt': img_in,
            'lq': img_in_lq,
        }

        return return_dict

    def __len__(self):
        return len(self.samples)


@DATASET_REGISTRY.register()
class SISRCUFEDDataset(data.Dataset):
    """Reference based CUFED dataset for super-resolution.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'ann_file': Use annotation file to generate paths.
        If opt['io_backend'] != lmdb and opt['ann_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The left.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_in (str): Data root path for input image.
        dataroot_ref (str): Data root path for ref image.
        ann_file (str): Path for annotation file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_flip (bool): Use horizontal and vertical flips.
        use_rot (bool): Use rotation (use transposing h and w for
            implementation).

        scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(SISRCUFEDDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.in_folder, self.ref_folder = opt['dataroot_in'], opt[
            'dataroot_ref']
        if 'filename_tmpl' in opt:  # only used for folder mode
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.in_folder, self.ref_folder]
            self.io_backend_opt['client_keys'] = ['in', 'ref']
            self.paths = paired_paths_from_lmdb(
                [self.in_folder, self.ref_folder], ['in', 'ref'])
        elif 'ann_file' in self.opt:
            self.paths = paired_paths_from_ann_file(
                [self.in_folder, self.ref_folder], ['in', 'ref'],
                self.opt['ann_file'])
        else:
            self.paths = paired_paths_from_folder(
                [self.in_folder, self.ref_folder], ['in', 'ref'],
                self.filename_tmpl)
        self.warp = transforms.RandomAffine(
            degrees=(10, 30),
            translate=(0.25, 0.5),
            scale=(1.2, 2.0),
            resample=Image.BICUBIC
        )

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load in and ref images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1] float32.
        in_path = self.paths[index]['in_path']
        img_bytes = self.file_client.get(in_path, 'in')
        img_in = mmcv.imfrombytes(img_bytes).astype(np.float32) / 255.
        ref_path = self.paths[index]['ref_path']
        img_bytes = self.file_client.get(ref_path, 'ref')
        img_ref = mmcv.imfrombytes(img_bytes).astype(np.float32) / 255.

        if self.opt['phase'] == 'train':
            gt_h, gt_w = self.opt['gt_size'], self.opt['gt_size']
            # some reference image in CUFED5_train have different sizes
            # resize reference image using PIL bicubic kernel
            img_ref = img_ref * 255
            img_ref = Image.fromarray(
                cv2.cvtColor(img_ref.astype(np.uint8), cv2.COLOR_BGR2RGB))
            img_ref = img_ref.resize((gt_w, gt_h), Image.BICUBIC)
            img_ref = cv2.cvtColor(np.array(img_ref), cv2.COLOR_RGB2BGR)
            img_ref = img_ref.astype(np.float32) / 255.
            # data augmentation
            img_in, img_ref = augment([img_in, img_ref], self.opt['use_flip'],
                                      self.opt['use_rot'])

        else:
            # for testing phase, zero padding to image pairs for same size
            img_in = mod_crop(img_in, scale)

            gt_h, gt_w, _ = img_in.shape

        # downsample image using PIL bicubic kernel
        lq_h, lq_w = gt_h // scale, gt_w // scale

        img_in_pil = img_in * 255
        img_in_pil = Image.fromarray(
            cv2.cvtColor(img_in_pil.astype(np.uint8), cv2.COLOR_BGR2RGB))

        img_in_lq = img_in_pil.resize((lq_w, lq_h), Image.BICUBIC)
        img_in_lq = cv2.cvtColor(np.array(img_in_lq), cv2.COLOR_RGB2BGR)
        img_in_lq = img_in_lq.astype(np.float32) / 255.

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_in, img_in_lq = img2tensor(  # noqa: E501
            [img_in, img_in_lq],
            bgr2rgb=True,
            float32=True)

        return_dict = {}

        if self.opt['phase'] != 'train':
            return_dict['gt'] = img_in
            return_dict['lq'] = img_in_lq
            return_dict['lq_path'] = ref_path

        return return_dict

    def __len__(self):
        return len(self.paths)


@DATASET_REGISTRY.register()
class SISRSun80Set(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.input_list = sorted(glob.glob(osp.join(opt['dataroot_in'], 'Sun_Hays_SR_groundtruth/*.jpg')) )

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        scale = self.opt['scale']

        img_in = cv2.imread(self.input_list[idx])
        ref_path = osp.basename(self.input_list[idx])

        img_in = mod_crop(img_in, scale)
        gt_h, gt_w, _ = img_in.shape
        # downsample image using PIL bicubic kernel
        lq_h, lq_w = gt_h // scale, gt_w // scale
        img_in_lq = Image.fromarray(img_in).resize((lq_w, lq_h), Image.BICUBIC)

        img_in = img_in.astype(np.float32) / 255.
        img_in_lq = np.array(img_in_lq).astype(np.float32) / 255.

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_in, img_in_lq = img2tensor(  # noqa: E501
            [img_in, img_in_lq],
            bgr2rgb=True,
            float32=True)

        return_dict = {
            'gt': img_in,
            'lq': img_in_lq,
            'lq_path': ref_path,
        }

        return return_dict


@DATASET_REGISTRY.register()
class SISRManga109Set(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.input_list = sorted(glob.glob(osp.join(opt['dataroot_in'], '*.png')) )

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        scale = self.opt['scale']

        img_in = cv2.imread(self.input_list[idx])
        ref_path = osp.basename(self.input_list[idx])

        img_in = mod_crop(img_in, scale)
        gt_h, gt_w, _ = img_in.shape
        # downsample image using PIL bicubic kernel
        lq_h, lq_w = gt_h // scale, gt_w // scale
        img_in_lq = Image.fromarray(img_in).resize((lq_w, lq_h), Image.BICUBIC)

        img_in = img_in.astype(np.float32) / 255.
        img_in_lq = np.array(img_in_lq).astype(np.float32) / 255.

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_in, img_in_lq = img2tensor(  # noqa: E501
            [img_in, img_in_lq],
            bgr2rgb=True,
            float32=True)

        return_dict = {
            'gt': img_in,
            'lq': img_in_lq,
            'lq_path': ref_path,
        }

        return return_dict


@DATASET_REGISTRY.register()
class SISRWRSRSet(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.input_list = sorted(glob.glob(osp.join(opt['dataroot_in'], '*.png')) )

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        scale = self.opt['scale']

        img_in = cv2.imread(self.input_list[idx])
        ref_path = osp.basename(self.input_list[idx])

        img_in = mod_crop(img_in, scale)
        gt_h, gt_w, _ = img_in.shape
        # downsample image using PIL bicubic kernel
        lq_h, lq_w = gt_h // scale, gt_w // scale
        img_in_lq = Image.fromarray(img_in).resize((lq_w, lq_h), Image.BICUBIC)

        img_in = img_in.astype(np.float32) / 255.
        img_in_lq = np.array(img_in_lq).astype(np.float32) / 255.

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_in, img_in_lq = img2tensor(  # noqa: E501
            [img_in, img_in_lq],
            bgr2rgb=True,
            float32=True)

        return_dict = {
            'gt': img_in,
            'lq': img_in_lq,
            'lq_path': ref_path,
        }

        return return_dict


@DATASET_REGISTRY.register()
class SISRMegaDepthTestSet(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.img_folder = osp.join(opt['dataroot_in'], 'test1600Pairs')
        self.pairs_frame = pd.read_csv(osp.join(opt['dataroot_in'], 'test1600Pairs.csv'))

    def __len__(self):
        return len(self.pairs_frame)

    def __getitem__(self, idx):
        scale = self.opt['scale']

        HR_name = self.pairs_frame.iloc[idx, 0]
        img_in = cv2.imread(osp.join(self.img_folder, HR_name))
        ref_path = HR_name

        img_in = mod_crop(img_in, scale)
        gt_h, gt_w, _ = img_in.shape
        # downsample image using PIL bicubic kernel
        lq_h, lq_w = gt_h // scale, gt_w // scale
        img_in_lq = Image.fromarray(img_in).resize((lq_w, lq_h), Image.BICUBIC)

        img_in = img_in.astype(np.float32) / 255.
        img_in_lq = np.array(img_in_lq).astype(np.float32) / 255.

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_in, img_in_lq = img2tensor(  # noqa: E501
            [img_in, img_in_lq],
            bgr2rgb=True,
            float32=True)

        return_dict = {
            'gt': img_in,
            'lq': img_in_lq,
            'lq_path': ref_path,
        }

        return return_dict


@DATASET_REGISTRY.register()
class SISRMegaDepthv3TestSet(data.Dataset):
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
        ref_path = HR_name

        img_in = mod_crop(img_in, scale)
        gt_h, gt_w, _ = img_in.shape
        # downsample image using PIL bicubic kernel
        lq_h, lq_w = gt_h // scale, gt_w // scale
        img_in_lq = Image.fromarray(img_in).resize((lq_w, lq_h), Image.BICUBIC)

        img_in = img_in.astype(np.float32) / 255.
        img_in_lq = np.array(img_in_lq).astype(np.float32) / 255.

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_in, img_in_lq = img2tensor(  # noqa: E501
            [img_in, img_in_lq],
            bgr2rgb=True,
            float32=True)

        return_dict = {
            'gt': img_in,
            'lq': img_in_lq,
            'lq_path': ref_path,
        }

        return return_dict

