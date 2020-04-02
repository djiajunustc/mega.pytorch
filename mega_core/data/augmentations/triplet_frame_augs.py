import cv2
import types
import numpy as np
from numpy import random

import torch
from torchvision import transforms


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img_list, boxes=None, labels=None):
        for t in self.transforms:
            img_list, boxes, labels = t(img_list, boxes, labels)
        return img_list, boxes, labels

class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self, crop_pert=0.3, no_iou_limit=False):
        self.crop_pert = crop_pert
        self.no_iou_limit = no_iou_limit
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            # (None, None),
        )


    def __call__(self, image_list, boxes=None, labels=None):
        height, width, _ = image_list[0].shape
        aspect_ratio = float(height) / float(width)
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if self.no_iou_limit:
                mode = (None, None)
            if mode is None:
                return image_list, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                w = random.uniform(self.crop_pert * width, width)
                h = w * aspect_ratio

                # # aspect ratio constraint b/t .5 & 2
                # if h / w < 0.5 or h / w > 2:
                #     continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                
                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # cut the crop from the image
                num_image = len(image_list)
                new_image_list = []
                for i in range(num_image):
                    image = image_list[i]
                    cropped_image = image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]
                    new_image_list.append(cropped_image)

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]
                
                assert current_boxes.shape[0] == current_labels.shape[0]
                return new_image_list, current_boxes, current_labels
            
            return image_list, boxes, labels


class Expand(object):
    def __init__(self, mean, expand_scale=3.0):
        self.mean = mean
        self.expand_scale = expand_scale

    def __call__(self, image_list, boxes, labels):
        if random.randint(2):
            return image_list, boxes, labels

        height, width, depth = image_list[0].shape
        boxes_w = boxes[:, 2] - boxes[:, 0] + 1
        boxes_h = boxes[:, 3] - boxes[:, 1] + 1
        
        for _ in range(50):
            ratio = random.uniform(1, self.expand_scale)
            expand_width = width * ratio
            expand_height = height * ratio
            
            short_size = np.minimum(boxes_w/expand_width, boxes_h/expand_height)
            if np.min(short_size) < 0.02:
                continue

            left = random.uniform(0, width*ratio - width)
            top = random.uniform(0, height*ratio - height)

            num_image = len(image_list)
            new_image_list = []
            for i in range(num_image):
                image = image_list[i]
                expand_image = np.zeros(
                    (int(height*ratio), int(width*ratio), depth),
                    dtype=image.dtype)
                expand_image[:, :, :] = self.mean
                expand_image[int(top):int(top + height),
                            int(left):int(left + width)] = image
                new_image_list.append(expand_image)

            boxes = boxes.copy()
            boxes[:, :2] += (int(left), int(top))
            boxes[:, 2:] += (int(left), int(top))

            return new_image_list, boxes, labels
        return image_list, boxes, labels

class ConvertFromInts(object):
    def __call__(self, image_list, boxes=None, labels=None):
        num_image = len(image_list)
        new_image_list = []
        for i in range(num_image):
            image = image_list[i]
            image = image.astype(np.float32)
            new_image_list.append(image)
        return new_image_list, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image_list, boxes=None, labels=None):
        if random.randint(2):
            _alpha = random.uniform(self.lower, self.upper)

            num_image = len(image_list)
            new_image_list = []
            for i in range(num_image):
                image = image_list[i]
                # print('alpha: ', _alpha)
                image[:, :, 1] *= _alpha
                new_image_list.append(image)
            return new_image_list, boxes, labels
        else:
            return image_list, boxes, labels
                

class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image_list, boxes=None, labels=None):
        if random.randint(2):
            _beta = random.uniform(-self.delta, self.delta)

            num_image = len(image_list)
            new_image_list = []
            for i in range(num_image):
                image = image_list[i]
                image[:, :, 0] += _beta
                image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
                image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
                new_image_list.append(image)
            return new_image_list, boxes, labels
        else:
            return image_list, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image_list, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels

            num_image = len(image_list)
            new_image_list = []
            for i in range(num_image):
                image = image_list[i]
                image = shuffle(image)
                new_image_list.append(image)
            
            return new_image_list, boxes, labels
        else:
            return image_list, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image_list, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            num_image = len(image_list)
            new_image_list = []
            for i in range(num_image):
                image = image_list[i]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                new_image_list.append(image)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            num_image = len(image_list)
            new_image_list = []
            for i in range(num_image):
                image = image_list[i]
                image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
                new_image_list.append(image)
            # image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return new_image_list, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image_list, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            
            num_image = len(image_list)
            new_image_list = []
            for i in range(num_image):
                image = image_list[i]
                image *= alpha
                new_image_list.append(image)    
            # image *= alpha
            return new_image_list, boxes, labels
        else:
            return image_list, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image_list, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            
            num_image = len(image_list)
            new_image_list = []
            for i in range(num_image):
                image = image_list[i]
                image += delta
                new_image_list.append(image)
            # image += delta
            return new_image_list, boxes, labels
        else:
            return image_list, boxes, labels


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image_list, boxes, labels):
        im_list = [image.copy() for image in image_list]

        # add brightness noise
        im_list, boxes, labels = self.rand_brightness(im_list, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im_list, boxes, labels = distort(im_list, boxes, labels)
        return self.rand_light_noise(im_list, boxes, labels)


class DataAugmentation(object):
    def __init__(self, mean=(103.06, 115.90, 123.15)):
        self.augment = Compose([
            ConvertFromInts(),
            PhotometricDistort(),
            Expand(mean),
            RandomSampleCrop(),
        ])
    
    def __call__(self, img_list, boxes, labels):
        return self.augment(img_list, boxes, labels)