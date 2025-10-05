import cv2
import depthai as dai
import numpy as np
import time

# --- 1. 核心参数 ---
# (这部分保持不变)
MODEL_PATH = 'models/blob/trafficlight_frozen_darknet_yolov4_model_openvino_2021.3_5shave.blob'
LABEL_MAP = ['red', 'green']
NN_INPUT_SIZE = 416
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.1
NUM_CLASSES = 2
COORDINATE_SIZE = 4
ANCHORS = np.array([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])
ANCHOR_MASKS = {"side52": [0, 1, 2], "side26": [3, 4, 5], "side13": [6, 7, 8]}

# --- 2. 构建现代化的 DepthAI Pipeline ---
print("路标 1: 正在定义 Pipeline...")
pipeline = dai.Pipeline()

# ... (省略中间所有节点定义和配置代码，它们和您之前的版本完全一样) ...
cam_rgb = pipeline.create(dai.node.ColorCamera)
detection_nn = pipeline.create(dai.node.YoloDetectionNetwork)
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("detections")
cam_rgb.setPreviewSize(NN_INPUT_SIZE, NN_INPUT_SIZE)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam_rgb.setFps(40)
detection_nn.setConfidenceThreshold(CONFIDENCE_THRESHOLD)
detection_nn.setNumClasses(NUM_CLASSES)
detection_nn.setCoordinateSize(COORDINATE_SIZE)
detection_nn.setAnchors(ANCHORS)
detection_nn.setAnchorMasks(ANCHOR_MASKS)
detection_nn.setIouThreshold(IOU_THRESHOLD)
detection_nn.setBlobPath(MODEL_PATH)
detection_nn.input.setBlocking(False)
cam_rgb.preview.link(detection_nn.input)
detection_nn.out.link(xout_nn.input)
detection_nn.passthrough.link(xout_rgb.input)

print("路标 2: Pipeline 定义成功！")
print("         现在，将尝试连接到 OAK-D 设备...")

# --- 3. 主程序循环 ---
try:
    with dai.Device(pipeline) as device:
        # 【新增路标】
        print("路标 3: 设备连接成功！正在进入主循环...")

        # 获取输出队列
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        q_nn = device.getOutputQueue(name="detections", maxSize=4, blocking=False)

        frame = None
        detections = []
        startTime = time.monotonic()
        counter = 0
        fps = 0

        print("Modern Traffic Light Detector is running...")
        print("Press 'q' to quit.")

        while True:
            in_rgb = q_rgb.tryGet()
            in_nn = q_nn.tryGet()

            if in_rgb is not None:
                frame = in_rgb.getCvFrame()

            if in_nn is not None:
                detections = in_nn.detections
                counter += 1

            currentTime = time.monotonic()
            if (currentTime - startTime) > 1:
                fps = counter / (currentTime - startTime)
                counter = 0
                startTime = currentTime

            if frame is not None:
                for detection in detections:
                    x1 = int(detection.xmin * frame.shape[1])
                    x2 = int(detection.xmax * frame.shape[1])
                    y1 = int(detection.ymin * frame.shape[0])
                    y2 = int(detection.ymax * frame.shape[0])

                    try:
                        label = LABEL_MAP[detection.label]
                    except:
                        label = detection.label

                    color = (0, 0, 255) if label == 'red' else (0, 255, 0)
                    
                    cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"{detection.confidence * 100:.2f}%", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                cv2.putText(frame, f"NN fps: {fps:.2f}", (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 255, 255))
                cv2.imshow("Modern Traffic Light Detector", frame)

            if cv2.waitKey(1) == ord('q'):
                break

except Exception as e:
    print(f"\n程序出现异常: {e}")

finally:
    print("\n路标 4: 脚本执行结束。")