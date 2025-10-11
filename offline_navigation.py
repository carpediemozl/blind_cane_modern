import cv2
import depthai as dai
import numpy as np
import time
import math

# --- 1. 内置所有需要的类 ---

# 1.1 鸟瞰图生成器 (BirdEye Class)
class BirdEye:
    def __init__(self):
        self.max_z = 3.0
        self.min_z = 0.3
        self.max_x = 1.0
        self.min_x = -1.0
        self.fov = 180
        self.shape = (320, 100, 3)
      
    def __make_bird_frame(self):
        frame = np.zeros(self.shape, np.uint8)
        min_y = int((1 - (0 - self.min_z) / (self.max_z - self.min_z)) * self.shape[0])
        cv2.rectangle(frame, (0, min_y), (self.shape[1], self.shape[0]), (70, 70, 70), -1)
        alpha = (180 - self.fov) / 2
        center = int(self.shape[1] / 2)
        max_p = self.shape[0] - int(math.tan(math.radians(alpha)) * center)
        fov_cnt = np.array([
            (0, self.shape[0]), (self.shape[1], self.shape[0]),
            (self.shape[1], max_p), (center, self.shape[0]),
            (0, max_p), (0, self.shape[0]),
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
            if z < (self.min_z * 1000): continue
            x, z = x/1000, z/1000
            left, right = self.__calc_x(x)
            top, bottom = self.__calc_z(z)
            frame = cv2.rectangle(frame, (left, top), (right, bottom), (100, 255, 0), 2)  
        width = int(frame.shape[1] * 300 / frame.shape[0])
        height = int(frame.shape[0] * 300 / frame.shape[0])
        dim = (width, height)
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        return frame

# 1.2 【核心修改】模拟的触觉反馈控制器 (Mock Haptic Controller)
class MockHapticController:
    def __init__(self, motor_config):
        print("模拟触觉反馈系统已初始化。")
        self.motor_intensities = {key: 0.0 for key in motor_config.keys()}

    def set_vibration(self, motor_id, intensity):
        if motor_id in self.motor_intensities:
            self.motor_intensities[motor_id] = max(0.0, min(1.0, intensity))
        # 我们不再与硬件交互，只在内部记录状态

    def cleanup(self):
        print("\n模拟触觉反馈系统已清理。")

# --- 2. 核心参数 ---
VIDEO_PATH = "test_video.mp4" # 【在这里指定您的测试视频】
MODEL_PATH = 'models/mobilenet-ssd_openvino_2021.2_6shave.blob'
LABEL_MAP = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor"
]
NN_INPUT_SIZE = 300
CONFIDENCE_THRESHOLD = 0.5
DIRECTION_THRESHOLD_MM = 150
MAX_VIBRATION_INTENSITY = 0.7
PULSE_INTERVAL_S = 0.4
PULSE_LOW_INTENSITY_FACTOR = 0.3
MOTOR_PINS = {0: {}, 1: {}} # 模拟版本只需要key

# --- 3. 构建视频输入的Pipeline ---
def create_video_pipeline():
    print("正在构建导航Pipeline (视频输入模式)...")
    pipeline = dai.Pipeline()
    
    # 定义节点
    xinFrame = pipeline.create(dai.node.XLinkIn)
    xinFrame.setStreamName("inFrame")
    spatial_nn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
    # 【注意】离线模式下，我们不需要Mono相机和Stereo节点
    # 但Spatial NN节点需要一个假的depth输入，我们创建一个空的输入
    xinDepth = pipeline.create(dai.node.XLinkIn)
    xinDepth.setStreamName("inDepth")

    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("detections")

    # 配置节点
    spatial_nn.setBlobPath(MODEL_PATH)
    spatial_nn.setConfidenceThreshold(CONFIDENCE_THRESHOLD)
    spatial_nn.input.setBlocking(False)
    spatial_nn.setBoundingBoxScaleFactor(0.5)
    spatial_nn.setDepthLowerThreshold(100)
    spatial_nn.setDepthUpperThreshold(10000)
    
    # 链接节点
    xinFrame.out.link(spatial_nn.input)
    xinDepth.out.link(spatial_nn.inputDepth) # 链接假的depth输入
    spatial_nn.out.link(xout_nn.input)
    
    print("Pipeline构建完成。")
    return pipeline

# --- 4. 主程序循环 ---
try:
    pipeline = create_video_pipeline()
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): raise IOError(f"无法打开视频文件: {VIDEO_PATH}")

    with dai.Device(pipeline) as device:
        print("设备连接成功！")
        
        frameQueue = device.getInputQueue("inFrame")
        depthQueue = device.getInputQueue("inDepth") # 获取假的depth输入队列
        q_nn = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        
        haptics = MockHapticController(MOTOR_PINS)
        bird_eye = BirdEye()

        pulse_state_is_high = True
        last_pulse_time = time.monotonic()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("视频播放结束，正在从头开始...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # 发送视频帧
            img = dai.ImgFrame()
            resized_frame = cv2.resize(frame, (NN_INPUT_SIZE, NN_INPUT_SIZE))
            img.setData(resized_frame)
            img.setType(dai.ImgFrame.Type.BGR888p)
            img.setWidth(NN_INPUT_SIZE)
            img.setHeight(NN_INPUT_SIZE)
            frameQueue.send(img)

            # 发送一个空的深度帧，以满足节点需求
            depth_img = dai.ImgFrame()
            depth_img.setType(dai.ImgFrame.Type.RAW16)
            depth_img.setWidth(NN_INPUT_SIZE) # 尺寸需要匹配
            depth_img.setHeight(NN_INPUT_SIZE)
            depthQueue.send(depth_img)
            
            in_nn = q_nn.get()
            detections = in_nn.detections

            # --- 触觉反馈决策逻辑 ---
            # (这部分代码和您之前的版本完全一样)
            closest_detection = None
            min_distance = float('inf')
            for detection in detections:
                coords = detection.spatialCoordinates
                if 300 < coords.z < 3000:
                    if coords.z < min_distance:
                        min_distance = coords.z
                        closest_detection = detection
            left_motor_intensity = 0.0
            right_motor_intensity = 0.0
            status_message = "Status: Searching..."
            if closest_detection:
                coords = closest_detection.spatialCoordinates
                distance = coords.z
                horizontal_pos = coords.x
                raw_intensity = 1.0 - ((distance - 300) / (3000 - 300))
                intensity = raw_intensity * MAX_VIBRATION_INTENSITY
                intensity = max(0.0, min(1.0, intensity))
                if horizontal_pos < -DIRECTION_THRESHOLD_MM:
                    left_motor_intensity = intensity
                    status_message = f"Obstacle LEFT! Dist: {distance/10:.1f} cm"
                elif horizontal_pos > DIRECTION_THRESHOLD_MM:
                    right_motor_intensity = intensity
                    status_message = f"Obstacle RIGHT! Dist: {distance/10:.1f} cm"
                else:
                    status_message = f"Obstacle FRONT! Dist: {distance/10:.1f} cm"
                    current_time = time.monotonic()
                    if (current_time - last_pulse_time) > PULSE_INTERVAL_S:
                        pulse_state_is_high = not pulse_state_is_high
                        last_pulse_time = current_time
                    if pulse_state_is_high:
                        left_motor_intensity = intensity; right_motor_intensity = intensity
                    else:
                        left_motor_intensity = intensity * PULSE_LOW_INTENSITY_FACTOR
                        right_motor_intensity = intensity * PULSE_LOW_INTENSITY_FACTOR
            haptics.set_vibration(0, left_motor_intensity)
            haptics.set_vibration(1, right_motor_intensity)

            # --- 绘图逻辑 ---
            # (这部分代码也和您之前的版本完全一样)
            cv2.rectangle(frame, (5, 5), (280, 75), (50, 50, 50), -1)
            status_color = (0, 0, 255) if closest_detection else (255, 255, 255)
            cv2.putText(frame, status_message, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            left_text = f"Left Motor (0): {haptics.motor_intensities[0]*100:.0f}%"
            left_color = (0, 255, 0) if haptics.motor_intensities[0] > 0 else (150, 150, 150)
            cv2.putText(frame, left_text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 1)
            right_text = f"Right Motor (1): {haptics.motor_intensities[1]*100:.0f}%"
            right_color = (0, 255, 0) if haptics.motor_intensities[1] > 0 else (150, 150, 150)
            cv2.putText(frame, right_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 1)
            spatial_coords_X = []; spatial_coords_Z = []
            for detection in detections:
                coords = detection.spatialCoordinates
                if coords.z > 0:
                    spatial_coords_X.append(coords.x); spatial_coords_Z.append(coords.z)
                x1 = int(detection.xmin * frame.shape[1]); x2 = int(detection.xmax * frame.shape[1])
                y1 = int(detection.ymin * frame.shape[0]); y2 = int(detection.ymax * frame.shape[0])
                try: label = LABEL_MAP[detection.label]
                except: label = "unknown"
                color = (0, 255, 0)
                if detection == closest_detection: color = (0, 0, 255)
                cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(frame, f"Z (offline): {int(coords.z)} mm", (x1 + 10, y1 + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(frame, f"X (offline): {int(coords.x)} mm", (x1 + 10, y1 + 60), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            bird_eye_frame = bird_eye.plotBirdEye(spatial_coords_X, spatial_coords_Z)
            cv2.imshow("Bird Eye View (Offline)", bird_eye_frame)
            cv2.imshow("Navigation View (Offline)", frame)

            if cv2.waitKey(1) == ord('q'):
                break

except Exception as e:
    print(f"\n程序在运行时出现异常: {e}")
    import traceback
    traceback.print_exc()

finally:
    if 'haptics' in locals():
        haptics.cleanup()
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()
    print("\n脚本执行结束。")