import cv2
import depthai as dai
import numpy as np
import time

# --- 1. 核心参数 ---
VIDEO_PATH = "mangdaoxinhaodeng_416x416.mp4" # 确保这个416x416的视频文件存在
MODEL_PATH = 'models/blob/trafficlight_frozen_darknet_yolov4_model_openvino_2021.3_5shave.blob'
LABEL_MAP = ['red', 'green']
NN_INPUT_SIZE = 416
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
NUM_CLASSES = 2
COORDINATE_SIZE = 4
ANCHORS = np.array([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])
ANCHOR_MASKS = {"side52": [0, 1, 2], "side26": [3, 4, 5], "side13": [6, 7, 8]}

# --- 2. 构建最精简的 2D 检测 Pipeline ---
def create_2d_pipeline():
    print("正在构建纯2D检测Pipeline...")
    pipeline = dai.Pipeline()
    
    # --- 定义节点 (只有两个！) ---
    # 1. 输入节点：用于从树莓派/PC接收视频帧
    xinFrame = pipeline.create(dai.node.XLinkIn)
    xinFrame.setStreamName("inFrame")
    
    # 2. AI节点：使用纯2D的YoloDetectionNetwork
    detection_nn = pipeline.create(dai.node.YoloDetectionNetwork)

    # 输出节点
    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("detections")
    
    # --- 配置AI节点 ---
    detection_nn.setBlobPath(MODEL_PATH)
    detection_nn.setConfidenceThreshold(CONFIDENCE_THRESHOLD)
    detection_nn.setNumClasses(NUM_CLASSES)
    detection_nn.setCoordinateSize(COORDINATE_SIZE)
    detection_nn.setAnchors(ANCHORS)
    detection_nn.setAnchorMasks(ANCHOR_MASKS)
    detection_nn.setIouThreshold(IOU_THRESHOLD)
    detection_nn.input.setBlocking(False)
    
    # --- 链接节点 ---
    xinFrame.out.link(detection_nn.input)
    detection_nn.out.link(xout_nn.input)
    
    print("Pipeline构建完成。")
    return pipeline

# --- 3. 主程序循环 ---
try:
    pipeline = create_2d_pipeline()
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): raise IOError(f"无法打开视频文件: {VIDEO_PATH}")

    with dai.Device(pipeline) as device:
        print("设备连接成功！")
        
        frameQueue = device.getInputQueue("inFrame")
        q_nn = device.getOutputQueue(name="detections", maxSize=1, blocking=True)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("视频播放结束，正在从头开始...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # 发送视频帧给OAK-D
            img = dai.ImgFrame()
            img.setData(frame) # 视频已经是416x416，无需再resize
            img.setType(dai.ImgFrame.Type.BGR888p)
            img.setWidth(NN_INPUT_SIZE)
            img.setHeight(NN_INPUT_SIZE)
            frameQueue.send(img)
            
            # 等待AI处理结果
            in_nn = q_nn.get()
            detections = in_nn.detections
            
            # 在原始帧上进行绘图
            for detection in detections:
                x1 = int(detection.xmin * frame.shape[1])
                x2 = int(detection.xmax * frame.shape[1])
                y1 = int(detection.ymin * frame.shape[0])
                y2 = int(detection.ymax * frame.shape[0])
                
                try: label = LABEL_MAP[detection.label]
                except: label = "unknown"
                
                color = (0, 0, 255) if label == 'red' else (0, 255, 0)
                
                cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(frame, f"{detection.confidence * 100:.2f}%", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.imshow("Simple 2D Tester", frame)

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