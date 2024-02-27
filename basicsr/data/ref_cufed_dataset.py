import cv2
import mmcv
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

from .data_util import (paired_paths_from_ann_file,
                            paired_paths_from_folder, paired_paths_from_lmdb)
from .transforms import augment, mod_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class RefCUFEDDataset(data.Dataset):
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
        super(RefCUFEDDataset, self).__init__()
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
        # img_ref = img_in[:, :, ::-1]
        """
        img_ref_pil = img_ref * 255
        img_ref_pil = Image.fromarray(
            cv2.cvtColor(img_ref_pil.astype(np.uint8), cv2.COLOR_BGR2RGB))
        img_ref = cv2.cvtColor(np.array(self.warp(img_ref_pil)), cv2.COLOR_RGB2BGR)
        img_ref = img_ref.astype(np.float32) / 255.
        """

        # downsample image using PIL bicubic kernel
        lq_h, lq_w = gt_h // scale, gt_w // scale

        img_in_pil = img_in * 255
        img_in_pil = Image.fromarray(
            cv2.cvtColor(img_in_pil.astype(np.uint8), cv2.COLOR_BGR2RGB))

        img_ref_pil = img_ref * 255
        img_ref_pil = Image.fromarray(
            cv2.cvtColor(img_ref_pil.astype(np.uint8), cv2.COLOR_BGR2RGB))

        img_in_lq = img_in_pil.resize((lq_w, lq_h), Image.BICUBIC)
        img_ref_lq = img_ref_pil.resize((lq_w, lq_h), Image.BICUBIC)

        # bicubic upsample LR
        img_in_up = img_in_lq.resize((gt_w, gt_h), Image.BICUBIC)
        img_ref_up = img_ref_lq.resize((gt_w, gt_h), Image.BICUBIC)

        img_in_lq = cv2.cvtColor(np.array(img_in_lq), cv2.COLOR_RGB2BGR)
        img_in_lq = img_in_lq.astype(np.float32) / 255.
        img_in_up = cv2.cvtColor(np.array(img_in_up), cv2.COLOR_RGB2BGR)
        img_in_up = img_in_up.astype(np.float32) / 255.
        img_ref_lq = cv2.cvtColor(np.array(img_ref_lq), cv2.COLOR_RGB2BGR)
        img_ref_lq = img_ref_lq.astype(np.float32) / 255.
        img_ref_up = cv2.cvtColor(np.array(img_ref_up), cv2.COLOR_RGB2BGR)
        img_ref_up = img_ref_up.astype(np.float32) / 255.

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_in, img_in_lq, img_in_up, img_ref, img_ref_lq, img_ref_up = img2tensor(  # noqa: E501
            [img_in, img_in_lq, img_in_up, img_ref, img_ref_lq, img_ref_up],
            bgr2rgb=True,
            float32=True)

        return_dict = {
            'img_in': img_in,
            'img_in_lq': img_in_lq,
            'img_in_up': img_in_up,
            'img_ref': img_ref,
            'img_ref_lq': img_ref_lq,
            'img_ref_up': img_ref_up,
        }

        if self.opt['phase'] != 'train':
            img_in_gt = img2tensor(img_in_gt, bgr2rgb=True, float32=True)
            return_dict['img_in'] = img_in_gt
            return_dict['lq_path'] = ref_path
            return_dict['padding'] = padding
            return_dict['original_size'] = (img_in_h, img_in_w)

        return return_dict

    def __len__(self):
        return len(self.paths)


def image_pair_generation_perspective(img,
                          random_perturb_range=(0, 32),
                          cropping_window_size=160,
                          dsize=None):

    if img is not None:
        shape1 = img.shape
        h = shape1[0]
        w = shape1[1]
    else:
        h = 160
        w = 160

    # ===== in image-1
    cropS = cropping_window_size
    x_topleft = np.random.randint(random_perturb_range[1],
                                  max(w, w - cropS - random_perturb_range[1]))
    y_topleft = np.random.randint(random_perturb_range[1],
                                  max(h, h - cropS - random_perturb_range[1]))

    x_topright = x_topleft + cropS
    y_topright = y_topleft

    x_bottomleft = x_topleft
    y_bottomleft = y_topleft + cropS

    x_bottomright = x_topleft + cropS
    y_bottomright = y_topleft + cropS

    tl = (x_topleft, y_topleft)
    tr = (x_topright, y_topright)
    br = (x_bottomright, y_bottomright)
    bl = (x_bottomleft, y_bottomleft)

    rect1 = np.array([tl, tr, br, bl], dtype=np.float32)

    # ===== in image-2
    x2_topleft = x_topleft + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])
    y2_topleft = y_topleft + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])

    x2_topright = x_topright + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])
    y2_topright = y_topright + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])

    x2_bottomleft = x_bottomleft + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])
    y2_bottomleft = y_bottomleft + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])

    x2_bottomright = x_bottomright + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])
    y2_bottomright = y_bottomright + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])

    tl2 = (x2_topleft, y2_topleft)
    tr2 = (x2_topright, y2_topright)
    br2 = (x2_bottomright, y2_bottomright)
    bl2 = (x2_bottomleft, y2_bottomleft)

    rect2 = np.array([tl2, tr2, br2, bl2], dtype=np.float32)

    # ===== homography
    H = cv2.getPerspectiveTransform(src=rect1, dst=rect2)
    H_inverse = np.linalg.inv(H)

    if img is not None:
        if dsize is None:
            dsize = (w, h)
        img_warped = cv2.warpPerspective(src=img, M=H_inverse, dsize=dsize, flags=cv2.INTER_CUBIC)
        return img_warped, H, H_inverse
    else:
        return H_inverse


@DATASET_REGISTRY.register()
class RefCUFEDCVTDataset(RefCUFEDDataset):
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

            img_ref_hrp, _, H_inverse_ref = image_pair_generation_perspective(
                (img_ref * 255).astype(np.uint8),
                random_perturb_range=(5, 20),
                dsize=(160, 160))
            img_ref_hrp = img_ref_hrp.astype(np.float32) / 255.

        else:
            # for testing phase, zero padding to image pairs for same size
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
        # img_ref = img_in[:, :, ::-1]
        """
        img_ref_pil = img_ref * 255
        img_ref_pil = Image.fromarray(
            cv2.cvtColor(img_ref_pil.astype(np.uint8), cv2.COLOR_BGR2RGB))
        img_ref = cv2.cvtColor(np.array(self.warp(img_ref_pil)), cv2.COLOR_RGB2BGR)
        img_ref = img_ref.astype(np.float32) / 255.
        """

        # downsample image using PIL bicubic kernel
        lq_h, lq_w = gt_h // scale, gt_w // scale

        img_in_pil = img_in * 255
        img_in_pil = Image.fromarray(
            cv2.cvtColor(img_in_pil.astype(np.uint8), cv2.COLOR_BGR2RGB))

        img_ref_pil = img_ref * 255
        img_ref_pil = Image.fromarray(
            cv2.cvtColor(img_ref_pil.astype(np.uint8), cv2.COLOR_BGR2RGB))

        img_in_lq = img_in_pil.resize((lq_w, lq_h), Image.BICUBIC)
        img_ref_lq = img_ref_pil.resize((lq_w, lq_h), Image.BICUBIC)

        # bicubic upsample LR
        img_in_up = img_in_lq.resize((gt_w, gt_h), Image.BICUBIC)
        img_ref_up = img_ref_lq.resize((gt_w, gt_h), Image.BICUBIC)

        img_in_lq = cv2.cvtColor(np.array(img_in_lq), cv2.COLOR_RGB2BGR)
        img_in_lq = img_in_lq.astype(np.float32) / 255.
        img_in_up = cv2.cvtColor(np.array(img_in_up), cv2.COLOR_RGB2BGR)
        img_in_up = img_in_up.astype(np.float32) / 255.
        img_ref_lq = cv2.cvtColor(np.array(img_ref_lq), cv2.COLOR_RGB2BGR)
        img_ref_lq = img_ref_lq.astype(np.float32) / 255.
        img_ref_up = cv2.cvtColor(np.array(img_ref_up), cv2.COLOR_RGB2BGR)
        img_ref_up = img_ref_up.astype(np.float32) / 255.

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_in, img_in_lq, img_in_up, img_ref, img_ref_lq, img_ref_up = img2tensor(  # noqa: E501
            [img_in, img_in_lq, img_in_up, img_ref, img_ref_lq, img_ref_up],
            bgr2rgb=True,
            float32=True)

        return_dict = {
            'img_in': img_in,
            'img_in_lq': img_in_lq,
            'img_in_up': img_in_up,
            'img_ref': img_ref,
            'img_ref_lq': img_ref_lq,
            'img_ref_up': img_ref_up,
        }

        if self.opt['phase'] == 'train':
            img_ref_hrp_pil = img_ref_hrp * 255
            img_ref_hrp_pil = Image.fromarray(
                cv2.cvtColor(img_ref_hrp_pil.astype(np.uint8), cv2.COLOR_BGR2RGB))

            img_ref_hrp_lq = img_ref_hrp_pil.resize((lq_w, lq_h), Image.BICUBIC)

            img_ref_hrp_up = img_ref_hrp_lq.resize((gt_w, gt_h), Image.BICUBIC)

            img_ref_hrp_lq = cv2.cvtColor(np.array(img_ref_hrp_lq), cv2.COLOR_RGB2BGR)
            img_ref_hrp_lq = img_ref_hrp_lq.astype(np.float32) / 255.
            img_ref_hrp_up = cv2.cvtColor(np.array(img_ref_hrp_up), cv2.COLOR_RGB2BGR)
            img_ref_hrp_up = img_ref_hrp_up.astype(np.float32) / 255.

            img_ref_hrp, img_ref_hrp_lq, img_ref_hrp_up = img2tensor(
                [img_ref_hrp, img_ref_hrp_lq, img_ref_hrp_up],
                bgr2rgb=True,
                float32=True)
            return_dict['img_ref_hrp'] = img_ref_hrp
            return_dict['img_ref_hrp_lq'] = img_ref_hrp_lq
            return_dict['img_ref_hrp_up'] = img_ref_hrp_up
        else:
            img_in_gt = img2tensor(img_in_gt, bgr2rgb=True, float32=True)
            return_dict['img_in'] = img_in_gt
            return_dict['lq_path'] = ref_path
            return_dict['padding'] = padding
            return_dict['original_size'] = (img_in_h, img_in_w)

        return return_dict

