# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 01:31:31 2021

@author: Wong Yi Jie
"""
from depthai_utils import V5TrafficPipeline

modelPath = r'C:\Users\liuyang\Downloads\yolov5_openvino_2022.1_6shave (9).blob'
label = ['red', 'green']
trafficPipeline = V5TrafficPipeline(modelPath, label)

try:
    trafficPipeline.run()
except:
    raise
