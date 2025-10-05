# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 01:31:31 2021

@author: Wong Yi Jie
"""
from depthai_utils import V4crosswalkPipeline

modelPath = r'models/frozen_darknet_yolov4_model_openvino_2021.4_6shave.blob'
label = ['crosswalk', 'guide_arrows']
v4CrosswalkPipeline = V4crosswalkPipeline(modelPath, label)

try:
    v4CrosswalkPipeline.run()
except:
    raise
