import cv2
import depthai as dai
import numpy as np
import time
import math#给birdeye部分使用

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
        
        frame = None
        detections = []
        
        # 创建 BirdEye 类的实例
        bird_eye = BirdEye()

        while True:
            in_rgb = q_rgb.get()
            in_nn = q_nn.get()
            in_depth = q_depth.get()

            frame = in_rgb.getCvFrame()
            detections = in_nn.detections
            depthFrame = in_depth.getFrame()
            
            # 将灰度图转换为BGR，以便绘制彩色框 本身就是灰度图 不需要再转化 会报错
            #frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # 存储所有物体的X, Z坐标，用于生成鸟瞰图
            spatial_coords_X = []
            spatial_coords_Z = []

            for detection in detections:
                x1 = int(detection.xmin * frame.shape[1])
                x2 = int(detection.xmax * frame.shape[1])
                y1 = int(detection.ymin * frame.shape[0])
                y2 = int(detection.ymax * frame.shape[0])
                
                try: label = LABEL_MAP[detection.label]
                except: label = "unknown"
                
                coords = detection.spatialCoordinates
                
                # 只处理在有效距离内的物体
                if coords.z > 0:
                    spatial_coords_X.append(coords.x)
                    spatial_coords_Z.append(coords.z)

                    color = (0, 255, 0) # 默认绿色
                    if coords.z < 1500: # 如果距离小于1.5米
                        color = (0, 0, 255) # 变为红色警告
                        cv2.putText(frame, "Obstacle!", (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

                    cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"Z: {int(coords.z / 10)} cm", (x1 + 10, y1 + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 生成并显示鸟瞰图
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
    cv2.destroyAllWindows()
    print("\n脚本执行结束。")