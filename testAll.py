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


videoPath = r"C:\Users\liuyang\Desktop\test\8bf7cb29d1be8cf2b0326aa900cb6efe.mp4"

numClasses = 5

blob = Path(__file__).parent.joinpath(r'C:\Users\liuyang\Downloads\FasterNet-c2f-yolo.blob')
model = dai.OpenVINO.Blob(blob)
dim = next(iter(model.networkInputs.values())).dims
W, H = dim[:2]

output_name, output_tenser = next(iter(model.networkOutputs.items()))
# if "yolov6" in output_name:
#     numClasses = output_tenser.dims[2] - 5
# else:
#     numClasses = output_tenser.dims[2] // 3 - 5
# fmt: off
labelMap = [
    'crosswalk', 'guide_arrows', 'blind_path', 'red_light', 'green_light'
]
# fmt: on

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
xinFrame = pipeline.create(dai.node.XLinkIn)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
xoutNN = pipeline.create(dai.node.XLinkOut)

xinFrame.setStreamName("inFrame")
xoutNN.setStreamName("detections")

# Network specific settings
detectionNetwork.setBlob(model)
detectionNetwork.setConfidenceThreshold(0.5)

# Yolo specific parameters
detectionNetwork.setNumClasses(numClasses)
detectionNetwork.setCoordinateSize(4)
detectionNetwork.setAnchors(
            np.array([10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]))
detectionNetwork.setAnchorMasks(
    {"side52": [0, 1, 2], "side26": [3, 4, 5], "side13": [6, 7, 8]})
detectionNetwork.setIouThreshold(0.8)

# Linking
xinFrame.out.link(detectionNetwork.input)
detectionNetwork.out.link(xoutNN.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Input queue will be used to send video frames to the device.
    inFrameQueue = device.getInputQueue(name="inFrame")
    # Output queue will be used to get nn data from the video frames.
    detectQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)

    frame = None
    detections = []
    # Random Colors for bounding boxes
    bboxColors = np.random.randint(256, size=(numClasses, 3)).tolist()

    cap = cv2.VideoCapture(videoPath)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video = cv2.VideoWriter(videoPath, fourcc, 20.0, (W, H))
    while cap.isOpened():
        read_correctly, frame = cap.read()
        if not read_correctly:
            break

        send_img(inFrameQueue, frame, W, H)

        detectQueueData = detectQueue.get()

        if detectQueueData is not None:
            detections = detectQueueData.detections

        if frame is not None:
            displayFrame(frame, detections, labelMap, bboxColors)
            cv2.imshow("image", frame)

        if cv2.waitKey(1) == ord("q"):
            break
