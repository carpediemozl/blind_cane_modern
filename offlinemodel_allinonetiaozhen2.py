import cv2
import depthai as dai
import numpy as np
import time

# --- 1. 核心参数 ---
MODEL_PATH = 'models/yolov8n_v6.blob'
LABEL_MAP = ['crosswalk', 'guide_arrows', 'blind_path', 'red_light', 'green_light']
NN_INPUT_SIZE = 640
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
NUM_CLASSES = 5
COORDINATE_SIZE = 4
ANCHORS = []
ANCHOR_MASKS = {}

def create_pipeline():

    # --- 2. 构建最终的 DepthAI Pipeline ---
    print("正在构建'大一统'模型Pipeline...")
    pipeline = dai.Pipeline()

    # --- 定义节点 ---
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    spatial_nn = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)

    # 【回归】我们只需要两个输出：一个用于AI结果，一个用于高速预览
    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("detections")
    xout_preview = pipeline.create(dai.node.XLinkOut)
    xout_preview.setStreamName("preview")

    # --- 配置节点 ---
    cam_rgb.setPreviewSize(NN_INPUT_SIZE, NN_INPUT_SIZE)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(30)

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

    spatial_nn.setBlobPath(MODEL_PATH)
    spatial_nn.setConfidenceThreshold(CONFIDENCE_THRESHOLD)
    spatial_nn.setNumClasses(NUM_CLASSES)
    spatial_nn.setCoordinateSize(COORDINATE_SIZE)
    spatial_nn.setAnchors(ANCHORS)
    spatial_nn.setAnchorMasks(ANCHOR_MASKS)
    spatial_nn.setIouThreshold(IOU_THRESHOLD)
    spatial_nn.input.setBlocking(False)
    spatial_nn.setBoundingBoxScaleFactor(0.5)
    spatial_nn.setDepthLowerThreshold(100)
    spatial_nn.setDepthUpperThreshold(10000)
    # 【关键优化】设置AI模型的输入队列大小和跳帧
    spatial_nn.input.setQueueSize(1)
    spatial_nn.input.setWaitForMessage(False)

    # --- 链接节点 ---
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    cam_rgb.preview.link(spatial_nn.input)
    stereo.depth.link(spatial_nn.inputDepth)
    # 【回归】直接从相机获取高速预览流
    cam_rgb.preview.link(xout_preview.input)
    spatial_nn.out.link(xout_nn.input)

    print("Pipeline构建完成，正在连接设备...")
    return pipeline
