import cv2
import depthai as dai
import numpy as np
import time

# --- 1. 核心参数 ---
VIDEO_PATH = "mangdaoxinhaodeng416x416.mp4" # 【在这里指定您的测试视频】
MODEL_PATH = 'models/path_segmentation.blob'
NN_INPUT_SIZE = 256

# --- 2. 后处理辅助函数 (从您的脚本中直接复制) ---
def postprocess_mask(rgb_frame, nn_output):
    """
    对神经网络的原始输出进行后处理，生成可视化的分割掩码。
    """
    mask = np.array(nn_output).reshape((3, NN_INPUT_SIZE, NN_INPUT_SIZE))
    mask = np.moveaxis(mask, 0, -1)
    mask = ((mask + 1) / 2 * 255).astype(np.uint8)
    return cv2.addWeighted(mask, 0.8, rgb_frame, 0.5, 0)

# --- 3. 构建视频输入的Pipeline ---
def create_video_pipeline():
    print("正在构建图像分割Pipeline (视频输入模式)...")
    pipeline = dai.Pipeline()
    
    # 定义节点
    xinFrame = pipeline.create(dai.node.XLinkIn)
    xinFrame.setStreamName("inFrame")
    nn = pipeline.create(dai.node.NeuralNetwork)
    
    # 输出节点
    # 【注意】我们需要两个输出：一个用于AI结果，一个用于与结果同步的原始帧
    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("nn_out")
    xout_passthrough = pipeline.create(dai.node.XLinkOut)
    xout_passthrough.setStreamName("passthrough")

    # 配置节点
    nn.setBlobPath(MODEL_PATH)
    nn.input.setBlocking(False)
    
    # 链接节点
    xinFrame.out.link(nn.input)
    # 将输入帧直通一个输出，以保证与nn_out同步
    nn.passthrough.link(xout_passthrough.input)
    nn.out.link(xout_nn.input)
    
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
        q_passthrough = device.getOutputQueue(name="passthrough", maxSize=4, blocking=False)
        q_nn = device.getOutputQueue(name="nn_out", maxSize=4, blocking=False)
        
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
            
            # 获取与AI结果同步的原始帧和AI结果
            in_passthrough = q_passthrough.get()
            in_nn = q_nn.get()
            
            rgb_frame = in_passthrough.getCvFrame()
            nn_output = in_nn.getFirstLayerFp16()
            
            # 调用后处理函数
            segmentation_frame = postprocess_mask(rgb_frame, nn_output)
            
            # 显示原始视频（调整回原始尺寸以对比）和分割结果
            cv2.imshow("Original Video", frame)
            cv2.imshow("Segmentation Tester (Offline)", segmentation_frame)

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