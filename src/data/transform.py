
import random
import cv2
import numpy as np


def box_xyxy_to_cxcywh(x):
    """box xyxy to cxcywh"""
    x0, y0, x1, y1 = np.array_split(x.T, 4)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return np.stack(b, axis=-1)[0]


def box_cxcywh_to_xyxy(x):
    """box cxcywh to xyxy"""
    x_c, y_c, w, h = np.array_split(x, 4, axis=-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return np.stack(b, axis=-1).squeeze(-2)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            return self.transforms1(image, target)
        return self.transforms2(image, target)


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = np.flip(img, 1)
            _, w, _ = img.shape

            target = target.copy()
            boxes = target["boxes"]
            boxes = boxes[:, [2, 1, 0, 3]] * np.array([-1, 1, -1, 1]) + np.array([w, 0, w, 0])
            target["boxes"] = boxes
        return img, target


def get_size(image_size, max_h, max_w):
    h, w = image_size
    scale = min(max_h/h, max_w/w)
    oh, ow = round(h*scale), round(w*scale)
    return oh, ow


class Resize(object):
    def __init__(self, max_size, size=None):
        if size:
            self.target_h, self.target_w = size, size
        else:
            self.target_h, self.target_w = max_size, max_size

    def __call__(self, img, target):
        h, w, _ = img.shape

        nh, nw = get_size((h, w), self.target_h, self.target_w)
        resize_pad_img = cv2.resize(img, (nw, nh))

        target = target.copy()
        # modify boxes
        ratio_width, ratio_height = float(nw)/float(w), float(nh)/float(h)
        boxes = target['boxes']
        boxes = boxes * np.array([ratio_width, ratio_height, ratio_width, ratio_height])
        target['boxes'] = boxes

        # modify size
        target['size'] = (nh, nw)

        return resize_pad_img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=1333):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target):
        size = random.choice(self.sizes)
        return Resize(max_size=self.max_size, size=size)(img, target)


class Pad(object):
    def __init__(self, tgt_h, tgt_w):
        self.tgt_h = tgt_h
        self.tgt_w = tgt_w

    def __call__(self, img, target):
        h, w, c = img.shape
        new_img = np.zeros((self.tgt_h, self.tgt_w, c), dtype=np.float32)
        new_img[:h, :w, :] = img
        new_mask = np.ones((self.tgt_h, self.tgt_w), dtype=np.float32)
        new_mask[:h, :w] = 0
        target['mask'] = new_mask
        target['size'] = (self.tgt_h, self.tgt_w)
        return new_img, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image, target):
        image = (image / 255)
        image = (image - self.mean) / self.std
        h, w, _ = image.shape

        target = target.copy()
        boxes = target["boxes"]
        boxes = box_xyxy_to_cxcywh(boxes)
        boxes = boxes / np.array([w, h, w, h], dtype=np.float32)
        target["boxes"] = boxes
        return image, target


class OutData(object):
    def __init__(self, is_training=True, max_size=1333):
        self.is_training = is_training
        self.pad_max_number = 100
        self.pad_func = Pad(max_size, max_size)

    def __call__(self, img, target):
        img, target = self.pad_func(img, target)
        img_data = img.transpose(2, 0, 1).astype(np.float32)
        mask = target['mask']
        if self.is_training:
            boxes = target['boxes'].astype(np.float32)
            labels = target['labels'].astype(np.int32)

            box_num = len(labels)
            gt_box = np.pad(boxes, ((0, self.pad_max_number - box_num), (0, 0)), mode="constant", constant_values=0)
            gt_label = np.pad(labels, (0, self.pad_max_number - box_num), mode="constant", constant_values=-1)
            gt_valid = np.zeros((self.pad_max_number,))
            gt_valid[:box_num] = 1
            gt_valid = gt_valid.astype(np.bool_)
            return img_data, mask, gt_box, gt_label, gt_valid
        else:
            image_id = target['image_id'].astype(np.int32)
            ori_size = np.array(target['ori_size'], dtype=np.int32)
            return img_data, mask, image_id, ori_size
