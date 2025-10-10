import cv2
import depthai as dai
import numpy as np
import time

# --- 1. 核心参数 ---
MODEL_PATH = 'models/blob/trafficlight_frozen_darknet_yolov4_model_openvino_2021.3_5shave.blob'
LABEL_MAP = ['red', 'green']
NN_INPUT_SIZE = 416

def create_pipeline():

    # --- 2. 构建最终的、回归经典的 DepthAI Pipeline ---
    print("正在构建Pipeline (经典空间检测模式)...")
    pipeline = dai.Pipeline()

    # --- 定义节点 ---
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    spatial_nn = pipeline.create(dai.node.YoloSpatialDetectionNetwork)

    # 输出节点
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("detections")
    # 【新增】输出深度图，用于可视化
    #xout_depth = pipeline.create(dai.node.XLinkOut)
    #xout_depth.setStreamName("depth")

    # --- 配置节点 ---
    cam_rgb.setPreviewSize(NN_INPUT_SIZE, NN_INPUT_SIZE)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

    spatial_nn.setBlobPath(MODEL_PATH)
    spatial_nn.setConfidenceThreshold(0.5)
    spatial_nn.input.setBlocking(False)
    spatial_nn.setBoundingBoxScaleFactor(0.5)
    spatial_nn.setDepthLowerThreshold(100)
    spatial_nn.setDepthUpperThreshold(10000)
    # YOLOv4 参数
    spatial_nn.setNumClasses(2)
    spatial_nn.setCoordinateSize(4)
    spatial_nn.setAnchors(np.array([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]))
    spatial_nn.setAnchorMasks({"side52": [0, 1, 2], "side26": [3, 4, 5], "side13": [6, 7, 8]})
    spatial_nn.setIouThreshold(0.5)

    # --- 链接节点 ---
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    cam_rgb.preview.link(spatial_nn.input)
    #深度信息传递 不要深度图必须要这个信息传递
    stereo.depth.link(spatial_nn.inputDepth)

    spatial_nn.passthrough.link(xout_rgb.input)
    spatial_nn.out.link(xout_nn.input)
    # 【新增】将超级节点的passthroughDepth输出链接出来，用于显示
    #spatial_nn.passthroughDepth.link(xout_depth.input)

    print("Pipeline构建完成，正在连接设备...")

    return pipeline
