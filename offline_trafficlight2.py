import cv2
import depthai as dai
import numpy as np
import time

# --- 1. 核心参数 ---
VIDEO_PATH = "mangdaoxinhaodeng_416x416.mp4"
MODEL_PATH = 'models/blob/trafficlight_frozen_darknet_yolov4_model_openvino_2021.3_5shave.blob'
LABEL_MAP = ['red', 'green']
NN_INPUT_SIZE = 416
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
NUM_CLASSES = 2
COORDINATE_SIZE = 4
ANCHORS = np.array([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])
ANCHOR_MASKS = {"side52": [0, 1, 2], "side26": [3, 4, 5], "side13": [6, 7, 8]}

# --- 2. 构建视频输入的Pipeline ---
def create_video_pipeline():
    print("正在构建红绿灯Pipeline (视频输入模式)...")
    pipeline = dai.Pipeline()
    
    xinFrame = pipeline.create(dai.node.XLinkIn)
    xinFrame.setStreamName("inFrame")
    spatial_nn = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
    xinDepth = pipeline.create(dai.node.XLinkIn)
    xinDepth.setStreamName("inDepth")
    
    # 【注意】我们不再需要 passthrough (rgb) 输出了
    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("detections")

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
    
    xinFrame.out.link(spatial_nn.input)
    xinDepth.out.link(spatial_nn.inputDepth)
    spatial_nn.out.link(xout_nn.input)
    
    print("Pipeline构建完成。")
    return pipeline

# --- 3. 主程序循环 ---
try:
    # 【修改】使用我们新生成的、尺寸正确的视频
    pipeline = create_video_pipeline()
    cap = cv2.VideoCapture(VIDEO_PATH)

    pipeline = create_video_pipeline()
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): raise IOError(f"无法打开视频文件: {VIDEO_PATH}")

    with dai.Device(pipeline) as device:
        print("设备连接成功！")
        
        frameQueue = device.getInputQueue("inFrame")
        depthQueue = device.getInputQueue("inDepth")
        q_nn = device.getOutputQueue(name="detections", maxSize=1, blocking=True)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # 【简化】不再需要实时resize，直接发送
            img = dai.ImgFrame()
            img.setData(frame)
            img.setType(dai.ImgFrame.Type.BGR888p)
            img.setWidth(NN_INPUT_SIZE)
            img.setHeight(NN_INPUT_SIZE)
            frameQueue.send(img)

            # 发送空的深度帧
            depth_data = np.zeros((NN_INPUT_SIZE, NN_INPUT_SIZE), dtype=np.uint16)
            depth_img = dai.ImgFrame()
            depth_img.setData(depth_data); depth_img.setType(dai.ImgFrame.Type.RAW16)
            depth_img.setWidth(NN_INPUT_SIZE); depth_img.setHeight(NN_INPUT_SIZE)
            depthQueue.send(depth_img)
            
            in_nn = q_nn.get()
            detections = in_nn.detections
            
            # 【简化】绘图逻辑现在非常直接
            for detection in detections:
                # 坐标现在是相对于416x416的帧，无需换算
                x1 = int(detection.xmin * NN_INPUT_SIZE)
                x2 = int(detection.xmax * NN_INPUT_SIZE)
                y1 = int(detection.ymin * NN_INPUT_SIZE)
                y2 = int(detection.ymax * NN_INPUT_SIZE)
                
                try: label = LABEL_MAP[detection.label]
                except: label = "unknown"
                
                color = (0, 0, 255) if label == 'red' else (0, 255, 0)
                
                cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.imshow("Traffic Light Tester (Offline)", frame)

            # 我们不再需要智能延迟，waitKey(1)即可
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