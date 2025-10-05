# coding=utf-8
from pathlib import Path
import cv2
import depthai as dai
import numpy as np
from time import monotonic
# coding=utf-8
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

# coding=utf-8
from time import monotonic

import cv2
import depthai as dai
import numpy as np


def frameNorm(frame, bbox):
    """
    nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height

    :param frame:
    :param bbox:
    :return:
    """
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


def drawText(frame, text, org, color=(255, 255, 255), thickness=1):
    cv2.putText(
        frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness + 3, cv2.LINE_AA
    )
    cv2.putText(
        frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness, cv2.LINE_AA
    )


def drawRect(frame, topLeft, bottomRight, color=(255, 255, 255), thickness=1):
    cv2.rectangle(frame, topLeft, bottomRight, (0, 0, 0), thickness + 3)
    cv2.rectangle(frame, topLeft, bottomRight, color, thickness)


def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.resize(arr, shape).transpose((2, 0, 1)).flatten()


def send_img(inFrameQueue, frame, W, H):
    img = dai.ImgFrame()
    img.setData(to_planar(frame, (W, H)))
    img.setTimestamp(dai.Clock.now())
    img.setWidth(W)
    img.setHeight(H)
    inFrameQueue.send(img)


def displayFrame(frame, detections, labelMap, bboxColors, depthFrameColor=None):
    for detection in detections:
        bbox = frameNorm(
            frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
        )
        drawText(
            frame,
            labelMap[detection.label],
            (bbox[0] + 10, bbox[1] + 20),
        )
        drawText(
            frame,
            f"{detection.confidence:.2%}",
            (bbox[0] + 10, bbox[1] + 35),
        )
        drawRect(
            frame,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            bboxColors[detection.label]
        )
        if hasattr(detection, "boundingBoxMapping") and depthFrameColor is not None:
            roi = detection.boundingBoxMapping.roi
            roi = roi.denormalize(
                depthFrameColor.shape[1], depthFrameColor.shape[0]
            )
            topLeft = roi.topLeft()
            bottomRight = roi.bottomRight()
            xmin = int(topLeft.x)
            ymin = int(topLeft.y)
            xmax = int(bottomRight.x)
            ymax = int(bottomRight.y)

            drawText(
                depthFrameColor,
                labelMap[detection.label],
                (xmin + 10, ymin + 20),
            )

            drawText(
                depthFrameColor,
                f"{detection.confidence:.2%}",
                (xmin + 10, ymin + 35),
            )

            drawText(
                depthFrameColor,
                f"X: {int(detection.spatialCoordinates.x)} mm",
                (xmin + 10, ymin + 50),
            )
            drawText(
                depthFrameColor,
                f"Y: {int(detection.spatialCoordinates.y)} mm",
                (xmin + 10, ymin + 65),
            )
            drawText(
                depthFrameColor,
                f"Z: {int(detection.spatialCoordinates.z)} mm",
                (xmin + 10, ymin + 80),
            )

            drawRect(
                depthFrameColor,
                (xmin, ymin),
                (xmax, ymax),
                bboxColors[detection.label],
            )

            drawText(
                frame,
                f"X: {int(detection.spatialCoordinates.x)} mm",
                (bbox[0] + 10, bbox[1] + 50),
            )
            drawText(
                frame,
                f"Y: {int(detection.spatialCoordinates.y)} mm",
                (bbox[0] + 10, bbox[1] + 65),
            )
            drawText(
                frame,
                f"Z: {int(detection.spatialCoordinates.z)} mm",
                (bbox[0] + 10, bbox[1] + 80),
            )


# coding=utf-8
from pathlib import Path
import cv2
import depthai as dai
import numpy as np
from time import monotonic

# 函数定义（frameNorm, drawText, drawRect, to_planar, send_img, displayFrame）保持不变

# 视频路径和模型路径
videoPath = r"C:\Users\liuyang\Documents\WeChat Files\wxid_zkasnjjznsyd22\FileStorage\Video\2024-03\693883f0dac3c8ba4309dd18efcd2ff1.mp4"
blobPath = r'C:\Users\liuyang\Downloads\FS-YOLOlight.blob'

# 标签和颜色
labelMap = ['red_light', 'green_light']
bboxColors = [[255, 191, 0], [0, 255, 0]]

# 创建pipeline
pipeline = dai.Pipeline()

# 定义pipeline源和输出
xinFrame = pipeline.create(dai.node.XLinkIn)
xinFrame.setStreamName("inFrame")

detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
# 设置模型
model = dai.OpenVINO.Blob(blobPath)
detectionNetwork.setBlob(model)
# 设置YOLO特定参数
detectionNetwork.setNumClasses(2)
detectionNetwork.setCoordinateSize(4)
detectionNetwork.setConfidenceThreshold(0.5)
detectionNetwork.setAnchors(np.array([10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]))
detectionNetwork.setAnchorMasks({"side52": [0, 1, 2], "side26": [3, 4, 5], "side13": [6, 7, 8]})
detectionNetwork.setIouThreshold(0.1)

xoutNN = pipeline.create(dai.node.XLinkOut)
xoutNN.setStreamName("detections")

# 链接
xinFrame.out.link(detectionNetwork.input)
detectionNetwork.out.link(xoutNN.input)

# 初始化视频编码器
cap = cv2.VideoCapture(videoPath)
dim = next(iter(model.networkInputs.values())).dims
W, H = dim[:2]

# out_video = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (416, 416))
frame_cnt = 0
frame_worded = 0
# 连接到设备并启动pipeline
with dai.Device(pipeline) as device:
    inFrameQueue = device.getInputQueue(name="inFrame")
    detectQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)

    while cap.isOpened():
        read_correctly, frame = cap.read()
        if not read_correctly:
            break
        frame_cnt += 1
        # 发送帧到设备
        send_img(inFrameQueue, frame, W, H)

        # 从设备获取检测结果
        detectQueueData = detectQueue.get()
        detections = detectQueueData.detections if detectQueueData is not None else []
        if len(detections) > 0:
            frame_worded += 1
        # 显示和保存处理后的帧
        if len(detections) > 0:

            displayFrame(frame, detections, labelMap, bboxColors)
            cv2.imshow("image", frame)
            # out_video.write(frame)  # 写入视频帧
            if frame_cnt % 1 == 0:
                img_name = f"frame_{frame_cnt}.jpg"  # 定义图片文件名
                cv2.imwrite(img_name, frame)  # 保存当前帧为图片
        if cv2.waitKey(1) == ord("q"):
            break

print('总帧数=', frame_cnt)
print('有效总帧数=', frame_worded)
# 释放资源
cap.release()
# out_video.release()
cv2.destroyAllWindows()
