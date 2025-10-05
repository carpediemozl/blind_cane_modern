from depthai_utils import TrafficPipeline

modelPath = r'models/blob/trafficlight_frozen_darknet_yolov4_model_openvino_2021.3_5shave.blob'
label = ['red', 'green']
trafficPipeline = TrafficPipeline(modelPath, label)

try:
    trafficPipeline.run()
except:
    raise
