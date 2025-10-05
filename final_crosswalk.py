import cv2
import depthai as dai
import numpy as np
import time

# --- 1. 核心参数 ---
MODEL_PATH = 'models/frozen_darknet_yolov4_model_openvino_2021.4_5shave.blob'
LABEL_MAP = ['crosswalk', 'guide_arrows']
NN_INPUT_SIZE = 416
#trafficlight为416 与之不同 ？ 640报错

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
#此处为置信度 可以调参
spatial_nn.setConfidenceThreshold(0.5)
spatial_nn.input.setBlocking(False)
spatial_nn.setBoundingBoxScaleFactor(0.5)
spatial_nn.setDepthLowerThreshold(100)
spatial_nn.setDepthUpperThreshold(10000)
# YOLOv4 参数
spatial_nn.setNumClasses(2)
spatial_nn.setCoordinateSize(4)
spatial_nn.setAnchors(np.array([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]))
#此处的AnchorMasks与trafficlight不同 不同报错 源代码写的有问题？
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

# --- 3. 主程序循环 ---
try:
    with dai.Device(pipeline) as device:
        print("设备连接成功！")
        
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        q_nn = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        # 【新增】获取深度图队列
        #q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        
        frame = None
        detections = []
        
        # 【新增】FPS计算器变量
        startTime = time.monotonic()
        counter = 0
        fps = 0

        while True:
            in_rgb = q_rgb.get()
            in_nn = q_nn.get()
            #不需要深度数据包
            #in_depth = q_depth.get()

            frame = in_rgb.getCvFrame()
            detections = in_nn.detections
            #不需要深度图帧
            #depthFrame = in_depth.getFrame()
            
            # 【新增】FPS计算逻辑
            counter += 1
            currentTime = time.monotonic()
            if (currentTime - startTime) > 1:
                fps = counter / (currentTime - startTime)
                counter = 0
                startTime = currentTime

            # 【新增】深度图可视化
            #不需要显示深度图
            '''
            depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
            cv2.imshow("depth", depthFrameColor)
            '''
            for detection in detections:
                x1 = int(detection.xmin * frame.shape[1])
                x2 = int(detection.xmax * frame.shape[1])
                y1 = int(detection.ymin * frame.shape[0])
                y2 = int(detection.ymax * frame.shape[0])
                
                try: label = LABEL_MAP[detection.label]
                except: label = "unknown"
                
                color = (0, 0, 255) if label == 'red' else (0, 255, 0)
                coords = detection.spatialCoordinates
                
                # 【增强】显示更完整的信息
                cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(frame, f"{detection.confidence * 100:.2f}%", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(frame, f"X: {int(coords.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(frame, f"Y: {int(coords.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(frame, f"Z: {int(coords.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 【新增】在主画面上显示FPS
            cv2.putText(frame, f"NN fps: {fps:.2f}", (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 255, 255))
            cv2.imshow("Final Detector", frame)

            if cv2.waitKey(1) == ord('q'):
                break

except Exception as e:
    print(f"\n程序在运行时出现异常: {e}")
    import traceback
    traceback.print_exc()

finally:
    cv2.destroyAllWindows()
    print("\n脚本执行结束。")