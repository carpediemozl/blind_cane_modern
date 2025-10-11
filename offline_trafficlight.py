import cv2
import depthai as dai
import numpy as np
import time

# --- 1. 核心参数 ---
VIDEO_PATH = "mangdaoxinhaodeng.mp4" # 【在这里指定您的测试视频】
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
    
    # 定义节点
    xinFrame = pipeline.create(dai.node.XLinkIn)
    xinFrame.setStreamName("inFrame")
    spatial_nn = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
    # 离线模式下，我们不需要Mono相机和Stereo节点
    # 但Spatial NN节点需要一个假的depth输入，我们创建一个空的输入
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
    xinDepth.out.link(spatial_nn.inputDepth) # 链接假的depth输入
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
        depthQueue = device.getInputQueue("inDepth") # 获取假的depth输入队列
        q_nn = device.getOutputQueue(name="detections", maxSize=1, blocking=True) # 设置为阻塞模式 实现oakd数据同步
        # 【注意】我们不再需要 q_rgb 了，因为我们直接使用原始帧进行显示
        #q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        #q_nn = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        
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

            # 发送一个空的深度帧，以满足节点需求
            # 尺寸需要与 NN_INPUT_SIZE 匹配，因为设置了 setDepthAlign
            depth_data = np.zeros((NN_INPUT_SIZE, NN_INPUT_SIZE), dtype=np.uint16)
            depth_img = dai.ImgFrame()
            depth_img.setData(depth_data)
            depth_img.setType(dai.ImgFrame.Type.RAW16)
            depth_img.setWidth(NN_INPUT_SIZE)
            depth_img.setHeight(NN_INPUT_SIZE)
            depthQueue.send(depth_img)
            
            #等待这一帧处理结果 同步oakd处理速度 防止喂数据速度大于处理速度
            in_nn = q_nn.get()
            detections = in_nn.detections
            
            '''
            in_rgb = q_rgb.get()
            in_nn = q_nn.get()
            
            # 使用 passthrough 的帧作为显示基础，因为它与检测结果是同步的
            display_frame = in_rgb.getCvFrame()
            detections = in_nn.detections
            
             # 4. 只获取AI的检测结果
            in_nn = q_nn.get()
            detections = in_nn.detections
            '''

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
                
                color = (0, 0, 255) if label == 'red' else (0, 255, 0)
                coords = detection.spatialCoordinates
                
                cv2.putText(display_frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(display_frame, f"{detection.confidence * 100:.2f}%", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                # 显示一个标记，表明Z坐标在离线模式下是无效的
                cv2.putText(display_frame, f"Z (offline): {int(coords.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(frame, f"NN fps: {fps:.2f}", (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 255, 255))
            cv2.imshow("Traffic Light Tester (Offline)", frame)

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