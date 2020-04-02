# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import single_transforms as SingleT
from . import triplet_transforms as TripleT


def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_horizontal_prob = cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN
        flip_vertical_prob = cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_horizontal_prob = 0.0
        flip_vertical_prob = 0.0
        
    to_bgr255 = cfg.INPUT.TO_BGR255 
    

    if not is_train:
        normalize_transform = SingleT.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
        )

        transform = SingleT.Compose(
            [
                SingleT.Resize(min_size, max_size),
                SingleT.RandomHorizontalFlip(flip_horizontal_prob),
                SingleT.RandomVerticalFlip(flip_vertical_prob),
                SingleT.ToTensor(),
                normalize_transform,
            ]
        )
    
    elif cfg.MODEL.VID.METHOD in ["rdn", "fgfa"]:
        normalize_transform = TripleT.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = TripleT.Compose(
            [
                TripleT.Resize(min_size, max_size),
                TripleT.RandomHorizontalFlip(flip_horizontal_prob),
                TripleT.RandomVerticalFlip(flip_vertical_prob),
                TripleT.ToTensor(),
                normalize_transform,
            ]
        )
        
    else:
        raise ValueError("not support {} for cfg.MODEL.VID.METHOD".format(cfg.MODEL.VID.METHOD))

    return transform
    