import cv2
import depthai as dai
import numpy as np
import time

# --- 1. 核心参数 ---
MODEL_PATH = 'models/path_segmentation.blob'
NN_INPUT_SIZE = 256

# --- 2. 后处理辅助函数 (从旧代码中提取并优化) ---

def postprocess_mask(rgb_frame, nn_output):
    """
    对神经网络的原始输出进行后处理，生成可视化的分割掩码。
    :param rgb_frame: 原始的BGR图像帧
    :param nn_output: 神经网络输出的一维FP16数据
    :return: 原始图像与分割掩码混合后的图像
    """
    # 1. 将一维数据重塑为 (通道, 高, 宽)
    mask = np.array(nn_output).reshape((3, NN_INPUT_SIZE, NN_INPUT_SIZE))
    # 2. 将通道维度从第一个移动到最后一个 (高, 宽, 通道)，这是OpenCV需要的格式
    mask = np.moveaxis(mask, 0, -1)
    # 3. 将掩码的值从 [-1.0, 1.0] 的范围归一化到 [0, 255] 的范围
    mask = ((mask + 1) / 2 * 255).astype(np.uint8)
    # 4. 将掩码与原始图像进行加权混合，产生半透明效果
    return cv2.addWeighted(mask, 0.8, rgb_frame, 0.5, 0)

# --- 3. 构建图像分割专用的 DepthAI Pipeline ---
print("正在构建图像分割Pipeline...")
pipeline = dai.Pipeline()

# --- 定义节点 ---
cam_rgb = pipeline.create(dai.node.ColorCamera)
# 使用通用的神经网络节点
nn = pipeline.create(dai.node.NeuralNetwork)
# 输出节点
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn_out")

# --- 配置节点 ---
cam_rgb.setPreviewSize(NN_INPUT_SIZE, NN_INPUT_SIZE)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

nn.setBlobPath(MODEL_PATH)
nn.input.setBlocking(False)

# --- 链接节点 ---
cam_rgb.preview.link(nn.input)
# 将原始的、未经修改的预览帧发送到 'rgb' 输出
cam_rgb.preview.link(xout_rgb.input)
# 将神经网络的计算结果发送到 'nn_out' 输出
nn.out.link(xout_nn.input)

print("Pipeline构建完成，正在连接设备...")

# --- 4. 主程序循环 ---
try:
    with dai.Device(pipeline) as device:
        print("设备连接成功！")
        
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        q_nn = device.getOutputQueue(name="nn_out", maxSize=4, blocking=False)
        
        frame = None
        nn_output = None

        while True:
            in_rgb = q_rgb.tryGet()
            in_nn = q_nn.tryGet()

            if in_rgb is not None:
                frame = in_rgb.getCvFrame()

            if in_nn is not None:
                nn_output = in_nn.getFirstLayerFp16()

            if frame is not None and nn_output is not None:
                # 调用后处理函数，生成最终的分割图像
                segmentation_frame = postprocess_mask(frame, nn_output)
                
                # 显示原始图像和分割后的图像
                cv2.imshow("Original RGB", frame)
                cv2.imshow("Walkable Path Segmentation", segmentation_frame)

            if cv2.waitKey(1) == ord('q'):
                break

except Exception as e:
    print(f"\n程序在运行时出现异常: {e}")
    import traceback
    traceback.print_exc()

finally:
    cv2.destroyAllWindows()
    print("\n脚本执行结束。")