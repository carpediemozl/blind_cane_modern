# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 01:31:31 2021

@author: Wong Yi Jie
"""
from depthai_utils import V8crosswalkPipeline

modelPath = r'models/blob/text_detection_db_480x640_openvino_2021.4_6shave.blob'
label = ['crosswalk', 'guide_arrows']
crosswalkPipeline = V8crosswalkPipeline(modelPath, label)

try:
    crosswalkPipeline.run()
except:
    raise
