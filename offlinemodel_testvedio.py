import cv2
import depthai as dai
import time

# --- 1. 测试配置 ---

# 【在这里选择您想测试的功能模块】
# 只需取消注释您想测试的那一行即可
# from final_traffic_light import create_pipeline, LABEL_MAP, NN_INPUT_SIZE
# from final_crosswalk import create_pipeline, LABEL_MAP, NN_INPUT_SIZE
from final_all_in_one import create_pipeline, LABEL_MAP, NN_INPUT_SIZE, COLORS

# 【在这里指定您的测试视频】
VIDEO_PATH = "test_video.mp4"

# --- 2. 构建特殊的“视频输入”Pipeline ---

# 我们从功能模块中导入它原始的Pipeline定义
pipeline = create_pipeline()

# 【偷梁换柱】
# 1. 找到原始的 ColorCamera 节点
cam_rgb_node = None
for node in pipeline.getAllNodes():
    if isinstance(node, dai.node.ColorCamera):
        cam_rgb_node = node
        break

if cam_rgb_node is None:
    raise RuntimeError("在Pipeline中找不到ColorCamera节点！")

# 2. 创建一个新的 XLinkIn 节点来替换它
xinFrame = pipeline.create(dai.node.XLinkIn)
xinFrame.setStreamName("inFrame")

# 3. 将原来链接到 ColorCamera 的所有链接，都改链接到我们的新输入节点上
for conn in pipeline.getConnections():
    if conn.outputNodeId == cam_rgb_node.id:
        # print(f"Redirecting link from ColorCamera to XLinkIn for input: {conn.inputName}")
        pipeline.removeConnection(conn)
        xinFrame.out.link(conn.input)

# 4. 删除无用的 ColorCamera 节点
pipeline.remove(cam_rgb_node)

print("Pipeline已修改为视频输入模式。")

# --- 3. 主程序循环 ---
try:
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {VIDEO_PATH}")

    with dai.Device(pipeline) as device:
        print("设备连接成功！")
        
        # 获取输入和输出队列
        frameQueue = device.getInputQueue("inFrame")
        q_nn = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("视频播放结束，正在从头开始...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # 将视频帧发送给OAK-D
            img = dai.ImgFrame()
            img.setData(cv2.resize(frame, (NN_INPUT_SIZE, NN_INPUT_SIZE)))
            img.setType(dai.ImgFrame.Type.BGR888p)
            img.setWidth(NN_INPUT_SIZE)
            img.setHeight(NN_INPUT_SIZE)
            frameQueue.send(img)
            
            # 获取AI检测结果
            in_nn = q_nn.get()
            detections = in_nn.detections

            # 绘图逻辑
            for detection in detections:
                x1 = int(detection.xmin * frame.shape[1])
                x2 = int(detection.xmax * frame.shape[1])
                y1 = int(detection.ymin * frame.shape[0])
                y2 = int(detection.ymax * frame.shape[0])
                
                try: label = LABEL_MAP[detection.label]
                except: label = "unknown"
                
                color = COLORS.get(label, (255, 255, 255))
                
                cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.imshow("Offline Tester", frame)

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