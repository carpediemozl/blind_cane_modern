import cv2
import depthai as dai
import numpy as np
import time

# --- 1. 核心参数 ---
VIDEO_PATH = "test_video.mp4" # 【在这里指定您的测试视频】
MODEL_PATH = 'models/frozen_darknet_yolov4_model_openvino_2021.4_5shave.blob'
LABEL_MAP = ['crosswalk', 'guide_arrows']
NN_INPUT_SIZE = 416 # 斑马线模型使用416x416的输入
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
NUM_CLASSES = 2
COORDINATE_SIZE = 4
ANCHORS = np.array([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])
# 【注意】这里的AnchorMasks与trafficlight版本不同
ANCHOR_MASKS = {"side52": [0, 1, 2], "side26": [3, 4, 5], "side13": [6, 7, 8]}

# --- 2. 构建视频输入的Pipeline ---
def create_video_pipeline():
    print("正在构建斑马线Pipeline (视频输入模式)...")
    pipeline = dai.Pipeline()
    
    # 定义节点
    xinFrame = pipeline.create(dai.node.XLinkIn)
    xinFrame.setStreamName("inFrame")
    spatial_nn = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
    xinDepth = pipeline.create(dai.node.XLinkIn)
    xinDepth.setStreamName("inDepth")
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("detections")

    # 配置节点
    spatial_nn.setBlobPath(MODEL_PATH)
    spatial_nn.setConfidenceThreshold(CONFIDENCE_THRESHOLD)
    spatial_nn.input.setBlocking(False)
    spatial_nn.setBoundingBoxScaleFactor(0.5)
    spatial_nn.setDepthLowerThreshold(100)
    spatial_nn.setDepthUpperThreshold(10000)
    spatial_nn.setNumClasses(NUM_CLASSES)
    spatial_nn.setCoordinateSize(COORDINATE_SIZE)
    spatial_nn.setAnchors(ANCHORS)
    spatial_nn.setAnchorMasks(ANCHOR_MASKS)
    spatial_nn.setIouThreshold(IOU_THRESHOLD)
    
    # 链接节点
    xinFrame.out.link(spatial_nn.input)
    xinDepth.out.link(spatial_nn.inputDepth)
    spatial_nn.passthrough.link(xout_rgb.input)
    spatial_nn.out.link(xout_nn.input)
    
    print("Pipeline构建完成。")
    return pipeline

# --- 3. 主程序循环 ---
try:
    pipeline = create_video_pipeline()
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): raise IOError(f"无法打开视频文件: {VIDEO_PATH}")

    with dai.Device(pipeline) as device:
        print("设备连接成功！")
        
        frameQueue = device.getInputQueue("inFrame")
        depthQueue = device.getInputQueue("inDepth")
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        q_nn = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        
        frame = None
        detections = []
        startTime = time.monotonic()
        counter = 0
        fps = 0

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

            # 发送一个空的深度帧
            depth_data = np.zeros((NN_INPUT_SIZE, NN_INPUT_SIZE), dtype=np.uint16)
            depth_img = dai.ImgFrame()
            depth_img.setData(depth_data)
            depth_img.setType(dai.ImgFrame.Type.RAW16)
            depth_img.setWidth(NN_INPUT_SIZE)
            depth_img.setHeight(NN_INPUT_SIZE)
            depthQueue.send(depth_img)
            
            in_rgb = q_rgb.get()
            in_nn = q_nn.get()
            
            display_frame = in_rgb.getCvFrame()
            detections = in_nn.detections

            counter += 1
            currentTime = time.monotonic()
            if (currentTime - startTime) > 1:
                fps = counter / (currentTime - startTime)
                counter = 0
                startTime = currentTime

            # 绘图逻辑
            for detection in detections:
                x1 = int(detection.xmin * display_frame.shape[1])
                x2 = int(detection.xmax * display_frame.shape[1])
                y1 = int(detection.ymin * display_frame.shape[0])
                y2 = int(detection.ymax * display_frame.shape[0])
                
                try: label = LABEL_MAP[detection.label]
                except: label = "unknown"
                
                color = (255, 255, 0) # 为斑马线设置黄色
                coords = detection.spatialCoordinates
                
                cv2.putText(display_frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(display_frame, f"{detection.confidence * 100:.2f}%", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(display_frame, f"Z (offline): {int(coords.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(display_frame, f"NN fps: {fps:.2f}", (2, display_frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 255, 255))
            cv2.imshow("Crosswalk Tester (Offline)", display_frame)

            if cv2.waitKey(1) == ord('q'):
                break

except Exception as e:
    print(f"\n程序在运行时出现异常: {e}")
    import traceback
    traceback.print_exc()

finally:
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()
    print("\n脚本执行结束。")