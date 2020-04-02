from . import single_frame_augs as SA
from . import triplet_frame_augs as TA


def build_augmentations(cfg):
    if not cfg.INPUT.WITH_AUGMENTATION:
        return None
        
    if cfg.MODEL.VID.METHOD in ["fgfa", "rdn"]:
        mean = cfg.INPUT.PIXEL_MEAN
        return TA.DataAugmentation(mean)

