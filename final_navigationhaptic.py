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

# --- 3. 主程序循环 ---
try:
    with dai.Device(pipeline) as device:
        print("设备连接成功！")
        
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        q_nn = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        
        # 【新增】初始化触觉反馈控制器
        haptics = HapticController(MOTOR_PINS)
        print("触觉反馈系统初始化成功。")

        frame = None
        detections = []
        
        # 创建 BirdEye 类的实例
        bird_eye = BirdEye()

        # 【新增】用于心跳脉冲的计时器和状态变量
        pulse_state_is_high = True
        last_pulse_time = time.monotonic()
        while True:
            in_rgb = q_rgb.get()
            in_nn = q_nn.get()
            in_depth = q_depth.get()

            frame = in_rgb.getCvFrame()
            detections = in_nn.detections
            depthFrame = in_depth.getFrame()
            
            # 将灰度图转换为BGR，以便绘制彩色框 本身就是灰度图 不需要再转化 会报错
            #frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

             # --- 1. 找到最近的障碍物 ---
            closest_detection = None
            min_distance = float('inf')

            for detection in detections:
                coords = detection.spatialCoordinates
                if 300 < coords.z < 3000:
                    if coords.z < min_distance:
                        min_distance = coords.z
                        closest_detection = detection
            
            # --- 2. 触觉反馈决策逻辑 & 状态变量定义 ---
            
            # 初始化状态变量
            left_motor_intensity = 0.0
            right_motor_intensity = 0.0
            status_message = "Status: Searching..."
            
            if closest_detection:
                coords = closest_detection.spatialCoordinates
                distance = coords.z
                horizontal_pos = coords.x

                # 计算基础震动强度
                raw_intensity = 1.0 - ((distance - 300) / (3000 - 300))
                intensity = raw_intensity * MAX_VIBRATION_INTENSITY
                intensity = max(0.0, min(1.0, intensity))

                # 根据水平位置决定哪个马达震动
                if horizontal_pos < -DIRECTION_THRESHOLD_MM: # 左侧
                    left_motor_intensity = intensity
                    status_message = f"Obstacle LEFT! Dist: {distance/10:.1f} cm"
                elif horizontal_pos > DIRECTION_THRESHOLD_MM: # 右侧
                    right_motor_intensity = intensity
                    status_message = f"Obstacle RIGHT! Dist: {distance/10:.1f} cm"
                else: # 正前方
                    status_message = f"Obstacle FRONT! Dist: {distance/10:.1f} cm"
                    # 【核心修改】实现“心跳”脉冲逻辑
                    current_time = time.monotonic()
                    if (current_time - last_pulse_time) > PULSE_INTERVAL_S:
                        pulse_state_is_high = not pulse_state_is_high # 反转状态
                        last_pulse_time = current_time
                    
                    if pulse_state_is_high:
                        # 强脉冲
                        left_motor_intensity = intensity
                        right_motor_intensity = intensity
                    else:
                        # 弱脉冲
                        left_motor_intensity = intensity * PULSE_LOW_INTENSITY_FACTOR
                        right_motor_intensity = intensity * PULSE_LOW_INTENSITY_FACTOR
            
            # 统一执行震动指令
            haptics.set_vibration(0, left_motor_intensity) # 马达0 = 左
            haptics.set_vibration(1, right_motor_intensity) # 马达1 = 右

            # --- 3. 绘图与鸟瞰图数据准备 ---
            
            # --- 【核心修改】绘制多行调试面板 ---
            # 背景框
            cv2.rectangle(frame, (5, 5), (280, 75), (50, 50, 50), -1)
            
            # 状态信息
            status_color = (0, 0, 255) if closest_detection else (255, 255, 255)
            cv2.putText(frame, status_message, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            
            # 左马达状态
            left_text = f"Left Motor (0): {left_motor_intensity*100:.0f}%"
            left_color = (0, 255, 0) if left_motor_intensity > 0 else (150, 150, 150)
            cv2.putText(frame, left_text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 1)
            
            # 右马达状态
            right_text = f"Right Motor (1): {right_motor_intensity*100:.0f}%"
            right_color = (0, 255, 0) if right_motor_intensity > 0 else (150, 150, 150)
            cv2.putText(frame, right_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 1)

            # --- 鸟瞰图和物体框绘制 (保持不变) ---
            spatial_coords_X = []
            spatial_coords_Z = []

            for detection in detections:
                # ... (您之前的绘图代码，用于显示所有检测框) ...
                coords = detection.spatialCoordinates
                if coords.z > 0:
                    spatial_coords_X.append(coords.x)
                    spatial_coords_Z.append(coords.z)
                x1 = int(detection.xmin * frame.shape[1]); x2 = int(detection.xmax * frame.shape[1])
                y1 = int(detection.ymin * frame.shape[0]); y2 = int(detection.ymax * frame.shape[0])
                try: label = LABEL_MAP[detection.label]
                except: label = "unknown"
                color = (0, 255, 0)
                if detection == closest_detection:
                    color = (0, 0, 255)
                cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(frame, f"Z: {int(coords.z / 10)} cm", (x1 + 10, y1 + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(frame, f"X: {int(coords.x)} mm", (x1 + 10, y1 + 60), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            bird_eye_frame = bird_eye.plotBirdEye(spatial_coords_X, spatial_coords_Z)
            cv2.imshow("Bird Eye View", bird_eye_frame)
            cv2.imshow("Navigation View", frame)

            if cv2.waitKey(1) == ord('q'):
                break

except Exception as e:
    print(f"\n程序在运行时出现异常: {e}")
    import traceback
    traceback.print_exc()

finally:
    # 【新增】确保在退出时，所有马达都安全停止
    if 'haptics' in locals():
        haptics.cleanup()
    cv2.destroyAllWindows()
    print("\n脚本执行结束。")