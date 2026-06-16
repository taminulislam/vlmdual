# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class GasEmissionDataset(BaseSegDataset):
    """Gas Emission Dataset for Rumen Acidosis Detection.
    
    This dataset contains CO2 and CH4 gas emission images from rumen samples
    at different pH levels. The segmentation task identifies:
    - Background (0): Non-relevant areas
    - Tube (1): The tube/container structure
    - Gas (2): Gas emission regions (CO2 or CH4)
    
    The dataset is used to classify rumen health based on pH levels:
    - Healthy: pH 6.2-6.5
    - Transitional: pH 5.9
    - Acidotic: pH 5.0-5.6
    
    Args:
        img_suffix (str): Suffix of images. Default: '.png'
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
    """
    
    METAINFO = dict(
        classes=('background', 'tube', 'gas'),
        palette=[
            [0, 0, 0],        # Black for background
            [128, 128, 128],  # Gray for tube
            [255, 0, 0]       # Red for gas emission
        ]
    )
    
    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

