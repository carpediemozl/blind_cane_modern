import cv2
import depthai as dai
import numpy as np
import time
import math#给birdeye部分使用
# 【新增】导入并定义触觉反馈所需的所有模块和配置
import RPi.GPIO as GPIO
from adafruit_pca9685 import PCA9685
from board import SCL, SDA
import busio

MOTOR_PINS = {
    0: {'in1': 5,  'in2': 6,  'pwm_channel': 2}, # 左马达
    1: {'in1': 13, 'in2': 19, 'pwm_channel': 3}, # 右马达
}

class HapticController:
    # ... (请将您 haptic_controller.py 中的 HapticController 类的完整代码复制到这里) ...
    def __init__(self, motor_config):
        self.config = motor_config
        self.motor_ids = list(self.config.keys())
        try:
            self.i2c = busio.I2C(SCL, SDA)
            self.pca = PCA9685(self.i2c)
            self.pca.frequency = 1000
            print("PCA9685 初始化成功。")
        except ValueError:
            print("错误: 无法找到I2C设备。请运行 'sudo i2cdetect -y 1' 检查硬件连接。")
            exit()
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for motor_id in self.motor_ids:
            pins = self.config[motor_id]
            GPIO.setup(pins['in1'], GPIO.OUT)
            GPIO.setup(pins['in2'], GPIO.OUT)
            GPIO.output(pins['in1'], GPIO.LOW)
            GPIO.output(pins['in2'], GPIO.LOW)
        print("GPIO 初始化成功。")

    def set_vibration(self, motor_id, intensity):
        if motor_id not in self.motor_ids: return
        intensity = max(0.0, min(1.0, intensity))
        pins = self.config[motor_id]
        if intensity > 0.01:
            GPIO.output(pins['in1'], GPIO.HIGH)
            GPIO.output(pins['in2'], GPIO.LOW)
            duty_cycle = int(intensity * 65535)
            self.pca.channels[pins['pwm_channel']].duty_cycle = duty_cycle
        else:
            GPIO.output(pins['in1'], GPIO.LOW)
            GPIO.output(pins['in2'], GPIO.LOW)
            self.pca.channels[pins['pwm_channel']].duty_cycle = 0

    def cleanup(self):
        print("\n正在停止所有马达并清理资源...")
        for motor_id in self.motor_ids:
            self.set_vibration(motor_id, 0)
        GPIO.cleanup()
        print("清理完成。")

class BirdEye:
    max_z = 3
    min_z = 0.3
    max_x = 1
    min_x = -1
    
    def __init__(self):
        self.fov = 180
        self.min_distance = 1
        self.shape = (320, 100, 3)
        pass
      
    def __make_bird_frame(self):
        frame = np.zeros(self.shape, np.uint8)
        min_y = int((1 - (self.min_distance - self.min_z) / (self.max_z - self.min_z)) * frame.shape[0])
        cv2.rectangle(frame, (0, min_y), (frame.shape[1], frame.shape[0]), (70, 70, 70), -1)
        alpha = (180 - self.fov) / 2
        center = int(frame.shape[1] / 2)
        max_p = frame.shape[0] - int(math.tan(math.radians(alpha)) * center)
        fov_cnt = np.array([
            (0, frame.shape[0]), (frame.shape[1], frame.shape[0]),
            (frame.shape[1], max_p), (center, frame.shape[0]),
            (0, max_p), (0, frame.shape[0]),
        ])
        cv2.fillPoly(frame, [fov_cnt], color=(70, 70, 70))
        return frame

    def __calc_x(self, val):
        norm = min(self.max_x, max(val, self.min_x))
        center = (norm - self.min_x) / (self.max_x - self.min_x) * self.shape[1]
        return int(max(center - 2, 0)), int(min(center + 2, self.shape[1]))

    def __calc_z(self, val):
        norm = min(self.max_z, max(val, self.min_z))
        center = (1 - (norm - self.min_z) / (self.max_z - self.min_z)) * self.shape[0]
        return int(max(center - 2, 0)), int(min(center + 2, self.shape[0]))
      
    def plotBirdEye(self, X, Z):
        frame = self.__make_bird_frame()
        for x, z in zip(X, Z):
            if z < 300: continue
            x, z = x/1000, z/1000
            left, right = self.__calc_x(x)
            top, bottom = self.__calc_z(z)
            frame = cv2.rectangle(frame, (left, top), (right, bottom), (100, 255, 0), 2)  
        width = int(frame.shape[1] * 300 / frame.shape[0])
        height = int(frame.shape[0] * 300 / frame.shape[0])
        dim = (width, height)
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        return frame

# --- 1. 核心参数 ---
MODEL_PATH = 'models/mobilenet-ssd_openvino_2021.2_6shave.blob'
LABEL_MAP = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor"
]
NN_INPUT_SIZE = 300
CONFIDENCE_THRESHOLD = 0.5
# 【新增】定义方位判断的阈值 (单位: 毫米)
DIRECTION_THRESHOLD_MM = 150 # 将阈值从200mm扩大到350mm 越小正前方视野越窄
# 【新增】设置马达的最大震动强度 (0.0 到 1.0)
MAX_VIBRATION_INTENSITY = 0.5 # 50%
# 【新增】心跳脉冲的参数
PULSE_INTERVAL_S = 0.4       # 心跳的间隔时间（秒）
PULSE_LOW_INTENSITY_FACTOR = 0.3 # 弱脉冲的强度是强脉冲的30%

def create_pipeline():

    # --- 2. 构建导航专用的 DepthAI Pipeline ---
    print("正在构建导航Pipeline...")
    pipeline = dai.Pipeline()

    # --- 定义节点 ---
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    # 【关键】使用 MobileNet 空间检测节点
    spatial_nn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
    # 【关键】创建一个 ImageManip 节点，用于将灰度图转换为模型需要的BGR格式
    image_manip = pipeline.create(dai.node.ImageManip)

    # 输出节点
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb") # 实际上是灰度图
    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("detections")
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")

    # --- 配置节点 ---
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    # 【注意】这里我们不需要对齐到RGB，因为我们根本没用RGB相机

    # 配置 ImageManip 节点
    image_manip.initialConfig.setResize(NN_INPUT_SIZE, NN_INPUT_SIZE)
    # MobileNet-SSD 模型需要BGR格式的输入，即使原始图像是灰度的
    image_manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)

    # 配置 MobileNet 空间检测节点
    spatial_nn.setBlobPath(MODEL_PATH)
    spatial_nn.setConfidenceThreshold(CONFIDENCE_THRESHOLD)
    spatial_nn.input.setBlocking(False)
    spatial_nn.setBoundingBoxScaleFactor(0.5)
    spatial_nn.setDepthLowerThreshold(100)
    spatial_nn.setDepthUpperThreshold(10000)

    # --- 链接节点 ---
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    # 【关键】将右侧单目摄像头的校正图像，作为AI模型的输入源
    stereo.rectifiedRight.link(image_manip.inputImage)
    image_manip.out.link(spatial_nn.input)
    stereo.depth.link(spatial_nn.inputDepth)

    spatial_nn.passthrough.link(xout_rgb.input)
    spatial_nn.out.link(xout_nn.input)
    spatial_nn.passthroughDepth.link(xout_depth.input)

    print("Pipeline构建完成，正在连接设备...")
    return pipeline

