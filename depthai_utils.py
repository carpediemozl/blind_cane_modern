import depthai as dai
import numpy as np
import time, sched, timeit
import cv2
import warnings
import pyttsx3

warnings.filterwarnings("ignore")
from task_utils import BirdEye


# -------------------------------------------------------------------------------
# Navigation Pipeline
# -------------------------------------------------------------------------------
class NavPipeline(BirdEye):
    def __init__(self, nnPath, labelMap,
                 syncNN=True, flipRectified=True,
                 erosionKernelSize=5, obstacleDistance=1000):
        # initialize BirdEye
        BirdEye.__init__(self)

        # declare variables
        self.__nnPath = nnPath
        self.__labelMap = labelMap
        self.__syncNN = syncNN
        self.__flipRectified = flipRectified

        # variables for obstacle avoidance
        assert erosionKernelSize in [2, 3, 4, 5, 6, 7], 'arg "erosionKernelSize" must be in [2, 3, 4, 5, 6, 7]'
        self.__erosionKernel = np.ones((erosionKernelSize, erosionKernelSize), np.uint8)
        self.__obstacleDistance = obstacleDistance

        # Create pipeline
        self.pipeline = dai.Pipeline()
        self.pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_2)

        # Define sources and outputs
        monoLeft = self.pipeline.createMonoCamera()
        monoRight = self.pipeline.createMonoCamera()
        stereo = self.pipeline.createStereoDepth()
        spatialDetectionNetwork = self.pipeline.createMobileNetSpatialDetectionNetwork()
        imageManip = self.pipeline.createImageManip()

        xoutManip = self.pipeline.createXLinkOut()
        nnOut = self.pipeline.createXLinkOut()
        depthRoiMap = self.pipeline.createXLinkOut()
        xoutDepth = self.pipeline.createXLinkOut()

        xoutManip.setStreamName("right")
        nnOut.setStreamName("detections")
        depthRoiMap.setStreamName("boundingBoxDepthMapping")
        xoutDepth.setStreamName("depth")

        # Properties
        imageManip.initialConfig.setResize(300, 300)
        # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
        imageManip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)

        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # StereoDepth
        stereo.setConfidenceThreshold(255)

        # Define a neural network that will make predictions based on the source frames
        spatialDetectionNetwork.setConfidenceThreshold(0.5)
        spatialDetectionNetwork.setBlobPath(self.__nnPath)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(65535)

        # Linking
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        imageManip.out.link(spatialDetectionNetwork.input)
        if self.__syncNN:
            spatialDetectionNetwork.passthrough.link(xoutManip.input)
        else:
            imageManip.out.link(xoutManip.input)

        spatialDetectionNetwork.out.link(nnOut.input)
        spatialDetectionNetwork.boundingBoxMapping.link(depthRoiMap.input)

        stereo.rectifiedRight.link(imageManip.inputImage)
        stereo.depth.link(spatialDetectionNetwork.inputDepth)
        spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

    def __obstacleAvoidance(self, crop=None, depthThreshold=600, sumThreshold=8000):
        '''
        purpose:
        - to guide user to avoid close obstalce on left or right side
        
        args
        1) crop
        - the cropped depth frame (ROI that we want)
        2) depthThreshold
        - the max distance to be considered CLOSE OBSTACLE (heurestically obtained)
        3) sumThreshold
        - the threshold for the sum of the left ROI and right ROI (heurestically obtained)
        '''
        if crop is not None:
            # create a frame
            obsFrame = np.full((30, 93, 3), 70, np.uint8)

            # mask the depth map
            masked = crop > depthThreshold
            masked = np.array(masked, dtype=np.float32)

            # erode the masked depth map
            # erosion = cv2.erode(masked, self.__erosionKernel, iterations = 1)
            erosion = cv2.morphologyEx(masked, cv2.MORPH_OPEN, self.__erosionKernel)

            # divide the eroded depth map to left and right
            left = erosion[:, :erosion.shape[1] // 2]
            right = erosion[:, erosion.shape[1] // 2:]

            # get the sum of the pixels in left and right
            leftSum = np.sum(1 - left)
            rightSum = np.sum(1 - right)

            # make decision
            if leftSum < sumThreshold:
                if rightSum < sumThreshold:
                    directionMessage = 'Clear'
                else:
                    directionMessage = 'Turn left to avoid Obstacle'
                    obsFrame = cv2.rectangle(obsFrame, (47, 0), (93, 30), (0, 0, 255), -1)
            elif rightSum < sumThreshold:
                directionMessage = 'Turn Right to avoid Obstacle'
                obsFrame = cv2.rectangle(obsFrame, (0, 0), (46, 30), (0, 0, 255), -1)
            else:
                directionMessage = 'Obstacle in Front!'
                obsFrame = cv2.rectangle(obsFrame, (0, 0), (93, 30), (0, 0, 255), -1)

                # show the images
            erosion = cv2.cvtColor(erosion, cv2.COLOR_GRAY2BGR)

            cv2.putText(erosion, directionMessage, (10, 14), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255))
            # cv2.imshow('Masked', masked)
            cv2.imshow('Crop', erosion)

            return obsFrame, erosion

    # new  
    def run(self, speech=None):
        # Connect to device and start pipeline
        with dai.Device(self.pipeline) as device:

            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            previewQueue = device.getOutputQueue(name="right", maxSize=4, blocking=False)
            detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
            depthRoiMapQueue = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4, blocking=False)
            depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

            rectifiedRight = None
            detections = []

            startTime = time.monotonic()
            counter = 0
            fps = 0
            color = (255, 255, 255)

            labelFrame = np.ones((330, 93, 3), np.uint8)
            labelFrame = np.full((330, 93, 3), 70, np.uint8)
            # labelFrame[:,0:2,:] = 255
            cv2.putText(labelFrame, 'far', (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (100, 255, 0))
            cv2.putText(labelFrame, 'obstacle', (10, 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (100, 255, 0))
            cv2.putText(labelFrame, 'close', (10, 305), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
            cv2.putText(labelFrame, 'obstacle', (10, 320), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))

            while True:
                inRectified = previewQueue.get()
                inDet = detectionNNQueue.get()
                inDepth = depthQueue.get()

                counter += 1
                currentTime = time.monotonic()
                if (currentTime - startTime) > 1:
                    fps = counter / (currentTime - startTime)
                    counter = 0
                    startTime = currentTime

                rectifiedRight = inRectified.getCvFrame()
                if self.__flipRectified:
                    rectifiedRight = cv2.flip(rectifiedRight, 1)

                depthFrame = inDepth.getFrame()
                if depthFrame is not None:
                    # not all region is useful for obstacle avoidance
                    # crop the ROI
                    depthCropped = depthFrame.copy()[200:, 120:520]
                    obsFrame, cropFrame = self.__obstacleAvoidance(depthCropped)

                depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depthFrameColor = cv2.equalizeHist(depthFrameColor)
                depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

                detections = inDet.detections
                if len(detections) != 0:
                    boundingBoxMapping = depthRoiMapQueue.get()
                    roiDatas = boundingBoxMapping.getConfigData()

                    for roiData in roiDatas:
                        roi = roiData.roi
                        roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                        topLeft = roi.topLeft()
                        bottomRight = roi.bottomRight()
                        xmin = int(topLeft.x)
                        ymin = int(topLeft.y)
                        xmax = int(bottomRight.x)
                        ymax = int(bottomRight.y)
                        cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color,
                                      cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

                # If the rectifiedRight is available, draw bounding boxes on it and show the rectifiedRight
                height = rectifiedRight.shape[0]
                width = rectifiedRight.shape[1]
                X, Z = [], []
                for detection in detections:
                    if self.__flipRectified:
                        swap = detection.xmin
                        detection.xmin = 1 - detection.xmax
                        detection.xmax = 1 - swap
                    # Denormalize bounding box
                    x1 = int(detection.xmin * width)
                    x2 = int(detection.xmax * width)
                    y1 = int(detection.ymin * height)
                    y2 = int(detection.ymax * height)

                    X.append(detection.spatialCoordinates.x)
                    Z.append(detection.spatialCoordinates.z)
                    try:
                        label = self.__labelMap[detection.label]
                    except:
                        label = detection.label

                    color = (100, 255, 0)
                    if detection.spatialCoordinates.z < self.__obstacleDistance:
                        color = (0, 0, 255)
                        if label == 'person':
                            msg = "Social Distance!"
                        else:
                            msg = 'Obstacle alert!'
                        cv2.putText(rectifiedRight, f"{msg}", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

                    cv2.putText(rectifiedRight, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(rectifiedRight, "{:.2f}%".format(detection.confidence * 100), (x1 + 10, y1 + 35),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    # cv2.putText(rectifiedRight, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    # cv2.putText(rectifiedRight, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    # cv2.putText(rectifiedRight, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(rectifiedRight, "Distance:", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(rectifiedRight, f"{int(detection.spatialCoordinates.z / 1000)} m", (x1 + 10, y1 + 65),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

                    cv2.rectangle(rectifiedRight, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                # get bird eye frame
                birdEyeFrame = self.plotBirdEye(X, Z)

                # combine the four frame
                frame = np.vstack((birdEyeFrame, obsFrame))
                frame = np.hstack((frame, labelFrame))
                '''
                cropFrame = cv2.resize(cropFrame, (300, 150), interpolation = cv2.INTER_AREA)
                rectifiedRight[150:,:] =rectifiedRight[150:,:] * cro pFrame
                '''
                rectifiedRight = cv2.resize(rectifiedRight, (330, 330), interpolation=cv2.INTER_AREA)
                rectifiedRight = np.hstack((rectifiedRight, frame))

                # add inference rate
                cv2.putText(rectifiedRight, "NN fps: {:.2f}".format(fps), (2, rectifiedRight.shape[0] - 4),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 255, 255))
                cv2.imshow("rectified right", rectifiedRight)

                if cv2.waitKey(1) == 13: #回车键
                    cv2.destroyAllWindows()
                    if speech is not None:
                        speech.task = None
                        speech.listen = True
                    break


# -------------------------------------------------------------------------------
# Road Segmentation Pipeline
# -------------------------------------------------------------------------------
class SegmentPipeline:
    def __init__(self, segModelPath):
        self.segModelPath = segModelPath

        # Create pipeline
        self.pipeline = dai.Pipeline()
        self.pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_2)

        # Create color cam node
        camRgb = self.pipeline.createColorCamera()
        camRgb.setPreviewSize(256, 256)
        camRgb.setInterleaved(False)
        camRgb.setFps(40)

        # create image manip node
        manip = self.pipeline.createImageManip()
        manip.initialConfig.setResize(256, 256)
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)

        # create nn node
        nn = self.pipeline.createNeuralNetwork()
        nn.setBlobPath(self.segModelPath)
        nn.setNumInferenceThreads(2)
        nn.setNumPoolFrames(4)
        nn.input.setBlocking(False)

        # link the nodes
        camRgb.preview.link(manip.inputImage)
        manip.out.link(nn.input)

        # output nodes
        xoutRgb_1 = self.pipeline.createXLinkOut()
        xoutRgb_1.setStreamName("rgb_1")
        xoutRgb_1.input.setBlocking(False)
        nn.passthrough.link(xoutRgb_1.input)

        nnOut = self.pipeline.createXLinkOut()
        nnOut.setStreamName("segmentation")
        nnOut.input.setBlocking(False)
        nn.out.link(nnOut.input)

    def run(self, speech=None):
        # Connect to device and start pipeline
        with dai.Device(self.pipeline) as device:

            # Output queues will be used to get the grayscale / depth frames and nn data from the outputs defined above
            qRgb = device.getOutputQueue(name="rgb_1", maxSize=4, blocking=False)
            qNet = device.getOutputQueue(name="segmentation", maxSize=4, blocking=False)

            def customReshape(x, target_shape):
                x = np.reshape(x, target_shape, order='F')
                for i in range(3):
                    x[:, :, i] = np.transpose(x[:, :, i])

                return x

            def show_deeplabv3p(output_colors, mask):
                mask = ((mask + 1) / 2 * 255).astype(np.uint8)
                return cv2.addWeighted(mask, 0.8, output_colors, 0.5, 0)

            startTime = 0
            counter = 0
            fps = 0

            # start looping
            while True:
                # Instead of get (blocking), we use tryGet (nonblocking) which will return the available data or None otherwise
                inRGB = qRgb.tryGet()
                inNet = qNet.tryGet()

                if inRGB is not None:
                    rgb = inRGB.getCvFrame()
                    cv2.imshow('rgb_1', rgb)

                counter += 1
                current_time = timeit.default_timer()
                if inRGB is not None and inNet is not None:
                    mask = inNet.getFirstLayerFp16()
                    mask = np.array(mask)
                    mask = customReshape(mask, (256, 256, 3))
                    if (current_time - startTime) > 1:
                        fps = counter / (current_time - startTime)
                        counter = 0
                        startTime = current_time

                    mask = show_deeplabv3p(rgb, mask)
                    cv2.putText(mask, "NN fps: {:.2f}".format(fps), (2, mask.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX,
                                0.4, (0, 255, 0))
                    cv2.imshow('mask', mask)

                # quit if user pressed 'q'
                if cv2.waitKey(1) == 13:
                    if speech is not None:
                        speech.task = None
                        speech.listen = True
                    cv2.destroyAllWindows()
                    break


# -------------------------------------------------------------------------------
# Pedestrian Traffic Light Detection Pipeline
# -------------------------------------------------------------------------------
class CrossWalkPipeline1:
    def __init__(self, modelPath, labelMap, syncNN=True):
        # define some variable
        self.modelPath = modelPath
        self.labelMap = labelMap
        self.syncNN = syncNN

        # Start defining a pipeline
        self.pipeline = dai.Pipeline()
        self.pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_4)

        # Define a source - color camera
        camRgb = self.pipeline.createColorCamera()
        camRgb.setPreviewSize(640, 640)
        camRgb.setInterleaved(False)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        # mono cam nodes
        monoLeft = self.pipeline.createMonoCamera()
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

        monoRight = self.pipeline.createMonoCamera()
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # stereo node
        stereo = self.pipeline.createStereoDepth()
        stereo.setConfidenceThreshold(255)
        # stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        # create yolo spatial network node
        spatialDetectionNetwork = self.pipeline.createYoloSpatialDetectionNetwork()
        spatialDetectionNetwork.setBlobPath(self.modelPath)
        spatialDetectionNetwork.setConfidenceThreshold(0.5)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(5000)
        # Yolo specific parameters
        spatialDetectionNetwork.setNumClasses(2)
        spatialDetectionNetwork.setCoordinateSize(4)
        spatialDetectionNetwork.setAnchors(
            np.array([10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]))
        spatialDetectionNetwork.setAnchorMasks({"side80": [0, 1, 2], "side40": [3, 4, 5], "side20": [6, 7, 8]})
        spatialDetectionNetwork.setIouThreshold(0.2)

        # link the nodes
        camRgb.preview.link(spatialDetectionNetwork.input)
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        # output node
        xoutRgb_2 = self.pipeline.createXLinkOut()
        xoutRgb_2.setStreamName("rgb")
        xoutNN = self.pipeline.createXLinkOut()
        xoutNN.setStreamName("detections")
        xoutBoundingBoxDepthMapping = self.pipeline.createXLinkOut()
        xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
        xoutDepth = self.pipeline.createXLinkOut()
        xoutDepth.setStreamName("depth")

        # connect to output node
        if self.syncNN:
            spatialDetectionNetwork.passthrough.link(xoutRgb_2.input)
        else:
            camRgb.preview.link(xoutRgb_2.input)

        spatialDetectionNetwork.out.link(xoutNN.input)
        spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)

        stereo.depth.link(spatialDetectionNetwork.inputDepth)
        spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

    def run(self):
        # Connect and start the pipeline
        cover = 0
        count = 0
        with dai.Device(self.pipeline) as device:

            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
            xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4,
                                                                blocking=False)
            depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

            frame = None
            detections = []

            startTime = time.monotonic()
            counter = 0
            fps = 0
            color = (255, 255, 255)
            red = (0, 0, 255)
            green = (0, 255, 0)

            # (new)
            s = sched.scheduler(time.time, time.sleep)
            start_time = time.time()

            while True:
                inPreview = previewQueue.get()
                inNN = detectionNNQueue.get()
                depth = depthQueue.get()
                num = 0
                count += 1
                # (new)
                current_time = time.time()
                elapsed_time = current_time - start_time
                notify_time = int(elapsed_time) % 5

                # (new)
                counter += 1
                current_time = time.monotonic()
                if (current_time - startTime) > 1:
                    fps = counter / (current_time - startTime)
                    counter = 0
                    startTime = current_time

                frame = inPreview.getCvFrame()
                depthFrame = depth.getFrame()

                depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depthFrameColor = cv2.equalizeHist(depthFrameColor)
                depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
                detections = inNN.detections
                if len(detections) != 0:
                    boundingBoxMapping = xoutBoundingBoxDepthMapping.get()
                    roiDatas = boundingBoxMapping.getConfigData()

                    for roiData in roiDatas:
                        roi = roiData.roi
                        roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                        topLeft = roi.topLeft()
                        bottomRight = roi.bottomRight()
                        xmin = int(topLeft.x)
                        ymin = int(topLeft.y)
                        xmax = int(bottomRight.x)
                        ymax = int(bottomRight.y)

                        cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color,
                                      cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

                # If the frame is available, draw bounding boxes on it and show the frame
                height = frame.shape[0]
                width = frame.shape[1]
                i = 0
                flag = 0
                bbox = np.zeros((10, 4), int)
                bbox2 = np.zeros((10, 4), int)
                for detection in detections:
                    # Denormalize bounding box
                    x1 = int(detection.xmin * width)
                    x2 = int(detection.xmax * width)
                    y1 = int(detection.ymin * height)
                    y2 = int(detection.ymax * height)

                    # 合并重合率大于0.2的标注框
                    # bbox[j, 0]--> x11 -->min  bbox[i, 0]--> x21 -->min
                    # bbox[j, 1]--> x12 -->max  bbox[i, 1]--> x22 -->min
                    # bbox[j, 2]--> y11 -->min  bbox[i, 2]--> y21 -->min
                    # bbox[j, 3]--> y12 -->max  bbox[i, 3]--> y22 -->min
                    bbox[i, :] = [x1, x2, y1, y2]
                    bbox2[i, :] = [x1, x2, y1, y2]
                    # flag = count
                    print('修改前bbox = \n', bbox)
                    # print('flag = \n', flag)
                    for j in range(0, i):
                        if bbox[i, 1] > bbox[j, 1] > bbox[i, 0] > bbox[j, 0]:
                            # 1-4
                            if bbox[j, 3] > bbox[i, 3] > bbox[j, 2] > bbox[i, 2]:
                                cover = (bbox[j, 1] - bbox[i, 0]) * (bbox[i, 3] - bbox[j, 2])
                            if bbox[i, 3] > bbox[j, 3] > bbox[j, 2] > bbox[i, 2]:
                                cover = (bbox[j, 1] - bbox[i, 0]) * (bbox[j, 3] - bbox[j, 2])
                            if bbox[j, 3] > bbox[i, 3] > bbox[i, 2] > bbox[j, 2]:
                                cover = (bbox[j, 1] - bbox[i, 0]) * (bbox[i, 3] - bbox[i, 2])
                            if bbox[i, 3] > bbox[j, 3] > bbox[i, 2] > bbox[j, 2]:
                                cover = (bbox[j, 1] - bbox[i, 0]) * (bbox[j, 3] - bbox[i, 2])
                        if bbox[j, 1] > bbox[i, 1] > bbox[j, 0] > bbox[i, 0]:
                            if bbox[j, 3] > bbox[i, 3] > bbox[j, 2] > bbox[i, 2]:
                                cover = (bbox[j, 1] - bbox[i, 0]) * (bbox[i, 3] - bbox[j, 2])
                            if bbox[i, 3] > bbox[j, 3] > bbox[j, 2] > bbox[i, 2]:
                                cover = (bbox[j, 1] - bbox[i, 0]) * (bbox[j, 3] - bbox[j, 2])
                            if bbox[j, 3] > bbox[i, 3] > bbox[i, 2] > bbox[j, 2]:
                                cover = (bbox[j, 1] - bbox[i, 0]) * (bbox[i, 3] - bbox[i, 2])
                            if bbox[i, 3] > bbox[j, 3] > bbox[i, 2] > bbox[j, 2]:
                                cover = (bbox[j, 1] - bbox[i, 0]) * (bbox[j, 3] - bbox[i, 2])
                        if bbox[j, 1] > bbox[i, 1] > bbox[i, 0] > bbox[j, 0]:
                            if bbox[j, 3] > bbox[i, 3] > bbox[j, 2] > bbox[i, 2]:
                                cover = (bbox[i, 1] - bbox[i, 0]) * (bbox[i, 3] - bbox[j, 2])
                            if bbox[i, 3] > bbox[j, 3] > bbox[j, 2] > bbox[i, 2]:
                                cover = (bbox[i, 1] - bbox[i, 0]) * (bbox[j, 3] - bbox[j, 2])
                            if bbox[j, 3] > bbox[i, 3] > bbox[i, 2] > bbox[j, 2]:
                                cover = (bbox[i, 1] - bbox[i, 0]) * (bbox[i, 3] - bbox[i, 2])
                            if bbox[i, 3] > bbox[j, 3] > bbox[i, 2] > bbox[j, 2]:
                                cover = (bbox[i, 1] - bbox[i, 0]) * (bbox[j, 3] - bbox[i, 2])
                        if bbox[i, 1] > bbox[j, 1] > bbox[j, 0] > bbox[i, 0]:
                            if bbox[j, 3] > bbox[i, 3] > bbox[j, 2] > bbox[i, 2]:
                                cover = (bbox[j, 1] - bbox[j, 0]) * (bbox[i, 3] - bbox[j, 2])
                            if bbox[j, 3] > bbox[i, 3] > bbox[i, 2] > bbox[j, 2]:
                                cover = (bbox[j, 1] - bbox[j, 0]) * (bbox[i, 3] - bbox[i, 2])
                            if bbox[i, 3] > bbox[j, 3] > bbox[i, 2] > bbox[j, 2]:
                                cover = (bbox[j, 1] - bbox[j, 0]) * (bbox[i, 3] - bbox[i, 2])
                            if bbox[i, 3] > bbox[j, 3] > bbox[j, 2] > bbox[i, 2]:
                                cover = (bbox[j, 1] - bbox[j, 0]) * (bbox[j, 3] - bbox[j, 2])

                        total = (bbox[j, 1] - bbox[j, 0]) * (bbox[j, 3] - bbox[j, 2]) + (
                                bbox[i, 1] - bbox[i, 0]) * (bbox[i, 3] - bbox[i, 2])
                        overlap = cover / (total - cover)
                        print('total = ', total)
                        print('cover = ', cover)
                        print('overlap = ', overlap)
                        if overlap >= 0.42 and np.any(bbox[j, :]):
                            x1 = max(bbox[j, 0], bbox[i, 0])
                            x2 = min(bbox[j, 1], bbox[i, 1])
                            y1 = max(bbox[j, 2], bbox[i, 2])
                            y2 = min(bbox[j, 3], bbox[i, 3])
                            bbox2[j, :] = [x1, x2, y1, y2]
                            bbox2[i, :] = 0
                            i -= 1
                            print('重绘')
                            print('bbox = \n', bbox)
                            print('bbox2 = \n', bbox2)
                            print('bbox行数 = \n', np.linalg.matrix_rank(bbox))
                            print('bbox2行数 = \n', np.linalg.matrix_rank(bbox2))
                            print(bbox[i + 1, :])
                            print(bbox[j, :])
                        # print('cover后x1 = ', x1)
                        # print('cover后x2 = ', x2)
                        # print('cover后y1 = ', y1)
                        # print('cover后y2 = ', y2)
                        else:
                            print('fuck')
                        j += 1
                    i = i + 1

                    try:
                        label = self.labelMap[detection.label]
                    except:
                        label = detection.label

                    if label == 'crosswalk':
                        color = green
                    else:
                        color = red
                    if np.linalg.matrix_rank(bbox) > np.linalg.matrix_rank(bbox2):
                        for n in range(i):
                            # cv2.putText(frame, str(label), (bbox2[n, 0] + 10, bbox2[n, 2] + 20),
                            #             cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                            # cv2.putText(frame, "{:.2f}".format(detection.confidence * 100),
                            #             (bbox2[n, 0] + 10, bbox2[n, 2] + 35),
                            #             cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                            # cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x) / 1000} m",
                            #             (bbox2[n, 0] + 10, bbox2[n, 2] + 50),
                            #             cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                            # cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y) / 1000} m",
                            #             (bbox2[n, 0] + 10, bbox2[n, 2] + 65),
                            #             cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                            # cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z) / 1000} m",
                            #             (bbox2[n, 0] + 10, bbox2[n, 2] + 80),
                            #             cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                            cv2.rectangle(frame, (bbox2[n, 0], bbox2[n, 2]), (bbox2[n, 1], bbox2[n, 3]), color,
                                          cv2.FONT_HERSHEY_SIMPLEX)
                            num += 1

                            print('画框次数', num)
                            print('坐标', bbox[n, :])
                            n += 1
                            print('i=', i)
                            # cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    # elif flag != count:
                    #     for n in range(i):
                    #         cv2.putText(frame, str(label), (bbox2[n, 0] + 10, bbox2[n, 2] + 20),
                    #                     cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    #         cv2.putText(frame, "{:.2f}".format(detection.confidence * 100),
                    #                     (bbox2[n, 0] + 10, bbox2[n, 2] + 35),
                    #                     cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    #         cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x) / 1000} m",
                    #                     (bbox2[n, 0] + 10, bbox2[n, 2] + 50),
                    #                     cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    #         cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y) / 1000} m",
                    #                     (bbox2[n, 0] + 10, bbox2[n, 2] + 65),
                    #                     cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    #         cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z) / 1000} m",
                    #                     (bbox2[n, 0] + 10, bbox2[n, 2] + 80),
                    #                     cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    #         cv2.rectangle(frame, (bbox2[n, 0], bbox2[n, 2]), (bbox2[n, 1], bbox2[n, 3]), color,
                    #                       cv2.FONT_HERSHEY_SIMPLEX)
                    #         num += 1
                    #
                    #         print('修改前画框次数', num)
                    #         print('坐标', bbox[n, :])
                    #         n += 1

                msg = None
                if len(detections) == 1:
                    if label == 'crosswalk':
                        msg = 'Red light! Dont Cross Yet'
                        color = green
                    cv2.putText(frame, msg, (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

                cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4,
                            color)
                # depthFrameColor = cv2.resize(depthFrameColor, None, fx=0.5, fy=0.5)
                cv2.imshow("depth", depthFrameColor)
                # frame = cv2.resize(frame, None, fx=10, fy=10)
                cv2.imshow("rgb", frame)

                print('count=', count)
                # (new)
                if cv2.waitKey(1) == 13:
                    cv2.destroyAllWindows()
                    # if speech is not None:
                    #     speech.task = None
                    #     speech.listen = True
                    break


class CrossWalkPipeline:
    def __init__(self, modelPath, labelMap, syncNN=True):
        # define some variable
        self.modelPath = modelPath
        self.labelMap = labelMap
        self.syncNN = syncNN
        engine = pyttsx3.init()
        # Start defining a pipeline
        self.pipeline = dai.Pipeline()
        self.pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_4)

        # Define a source - color camera
        camRgb = self.pipeline.createColorCamera()
        camRgb.setPreviewSize(640, 640)
        camRgb.setInterleaved(False)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        # mono cam nodes
        monoLeft = self.pipeline.createMonoCamera()
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

        monoRight = self.pipeline.createMonoCamera()
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # stereo node
        stereo = self.pipeline.createStereoDepth()
        stereo.setConfidenceThreshold(255)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        # create yolo spatial network node
        spatialDetectionNetwork = self.pipeline.createYoloSpatialDetectionNetwork()
        spatialDetectionNetwork.setBlobPath(self.modelPath)
        spatialDetectionNetwork.setConfidenceThreshold(0.6)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(65535)
        # Yolo specific parameters
        spatialDetectionNetwork.setNumClasses(2)
        spatialDetectionNetwork.setCoordinateSize(4)
        spatialDetectionNetwork.setAnchors(
            np.array([10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]))
        spatialDetectionNetwork.setAnchorMasks(
            {"side80": [0, 1, 2], "side40": [3, 4, 5], "side20": [6, 7, 8]})
        spatialDetectionNetwork.setIouThreshold(0.2)

        # link the nodes
        camRgb.preview.link(spatialDetectionNetwork.input)
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        # output node
        xoutRgb_2 = self.pipeline.createXLinkOut()
        xoutRgb_2.setStreamName("rgb")
        xoutNN = self.pipeline.createXLinkOut()
        xoutNN.setStreamName("detections")
        xoutBoundingBoxDepthMapping = self.pipeline.createXLinkOut()
        xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
        xoutDepth = self.pipeline.createXLinkOut()
        xoutDepth.setStreamName("depth")

        # connect to output node
        if self.syncNN:
            spatialDetectionNetwork.passthrough.link(xoutRgb_2.input)
        else:
            camRgb.preview.link(xoutRgb_2.input)

        spatialDetectionNetwork.out.link(xoutNN.input)
        spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)

        stereo.depth.link(spatialDetectionNetwork.inputDepth)
        spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

    def run(self):
        # Connect and start the pipeline
        with dai.Device(self.pipeline) as device:
            engine = pyttsx3.init()
            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            previewQueue = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
            detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
            xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=1,
                                                                blocking=False)
            depthQueue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)

            frame = None
            detections = []

            startTime = time.monotonic()
            counter = 0
            fps = 0
            color = (255, 255, 255)
            red = (0, 0, 255)
            green = (0, 255, 0)

            # (new)
            s = sched.scheduler(time.time, time.sleep)
            start_time = time.time()

            while True:
                inPreview = previewQueue.get()
                inNN = detectionNNQueue.get()
                depth = depthQueue.get()

                # (new)
                current_time = time.time()
                elapsed_time = current_time - start_time
                notify_time = int(elapsed_time) % 5

                # (new)
                counter += 1
                current_time = time.monotonic()
                if (current_time - startTime) > 1:
                    fps = counter / (current_time - startTime)
                    counter = 0
                    startTime = current_time

                frame = inPreview.getCvFrame()
                depthFrame = depth.getFrame()

                depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depthFrameColor = cv2.equalizeHist(depthFrameColor)
                depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
                detections = inNN.detections
                if len(detections) != 0:
                    boundingBoxMapping = xoutBoundingBoxDepthMapping.get()
                    roiDatas = boundingBoxMapping.getConfigData()

                    for roiData in roiDatas:
                        roi = roiData.roi
                        roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                        topLeft = roi.topLeft()
                        bottomRight = roi.bottomRight()
                        xmin = int(topLeft.x)
                        ymin = int(topLeft.y)
                        xmax = int(bottomRight.x)
                        ymax = int(bottomRight.y)

                        cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color,
                                      cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

                # If the frame is available, draw bounding boxes on it and show the frame
                height = frame.shape[0]
                width = frame.shape[1]
                for detection in detections:
                    # Denormalize bounding box
                    x1 = int(detection.xmin * width)
                    x2 = int(detection.xmax * width)
                    y1 = int(detection.ymin * height)
                    y2 = int(detection.ymax * height)
                    try:
                        label = self.labelMap[detection.label]
                    except:
                        label = detection.label

                    if label == 'crosswalk':
                        color = green
                    else:
                        color = red
                    cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, "{:.2f}".format(detection.confidence * 100), (x1 + 10, y1 + 35),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x) / 1000} m",
                                (x1 + 10, y1 + 50),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y) / 1000} m",
                                (x1 + 10, y1 + 65),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z) / 1000} m",
                                (x1 + 10, y1 + 80),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                msg = None
                if len(detections) == 1:
                    if label == 'crosswalk':
                        msg = 'Red light! Dont Cross Yet'
                        # tips = '前方有斑马线'
                        # engine.say(tips)
                        # engine.runAndWait()
                        # dist = '还有%d米' % int(detection.spatialCoordinates.z / 1000)
                        # engine.say(dist)
                        # engine.runAndWait()
                        color = green
                    cv2.putText(frame, msg, (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

                cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.4,
                            color)
                depthFrameColor = cv2.resize(depthFrameColor, None, fx=0.5, fy=0.5)
                cv2.imshow("depth", depthFrameColor)
                # frame = cv2.resize(frame, None, fx=10, fy=10)
                cv2.imshow("rgb", frame)

                # (new)
                
                if cv2.waitKey(1) == 13:
                    cv2.destroyAllWindows()
                    # if speech is not None:
                    #     speech.task = None
                    #     speech.listen = True
                    break


class V4crosswalkPipeline:
    def __init__(self, modelPath, labelMap, syncNN=True):
        # define some variable
        self.modelPath = modelPath
        self.labelMap = labelMap
        self.syncNN = syncNN

        # Start defining a pipeline
        self.pipeline = dai.Pipeline()
        self.pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_4)

        # Define a source - color camera
        camRgb = self.pipeline.createColorCamera()
        camRgb.setPreviewSize(640, 640)
        camRgb.setInterleaved(False)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        # mono cam nodes
        monoLeft = self.pipeline.createMonoCamera()
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

        monoRight = self.pipeline.createMonoCamera()
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # stereo node
        stereo = self.pipeline.createStereoDepth()
        stereo.setConfidenceThreshold(255)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        # create yolo spatial network node
        spatialDetectionNetwork = self.pipeline.createYoloSpatialDetectionNetwork()
        spatialDetectionNetwork.setBlobPath(self.modelPath)
        spatialDetectionNetwork.setConfidenceThreshold(0.5)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(65535)
        # Yolo specific parameters
        spatialDetectionNetwork.setNumClasses(2)
        spatialDetectionNetwork.setCoordinateSize(4)
        spatialDetectionNetwork.setAnchors(np.array([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]))

        spatialDetectionNetwork.setAnchorMasks({"side40": np.array([1, 2, 3]), "side20": np.array([3, 4, 5])})
        spatialDetectionNetwork.setIouThreshold(0.5)

        # link the nodes
        camRgb.preview.link(spatialDetectionNetwork.input)
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        # output node
        xoutRgb_2 = self.pipeline.createXLinkOut()
        xoutRgb_2.setStreamName("rgb")
        xoutNN = self.pipeline.createXLinkOut()
        xoutNN.setStreamName("detections")
        xoutBoundingBoxDepthMapping = self.pipeline.createXLinkOut()
        xoutBoundingBoxDepthMapping.setStreamName("boundingBoxMapping")
        xoutDepth = self.pipeline.createXLinkOut()
        xoutDepth.setStreamName("depth")

        # connect to output node
        if self.syncNN:
            spatialDetectionNetwork.passthrough.link(xoutRgb_2.input)
        else:
            camRgb.preview.link(xoutRgb_2.input)

        spatialDetectionNetwork.out.link(xoutNN.input)
        spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)

        stereo.depth.link(spatialDetectionNetwork.inputDepth)
        spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

    def run(self):
        # Connect and start the pipeline
        with dai.Device(self.pipeline) as device:
            engine = pyttsx3.init()
            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
            xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxMapping", maxSize=4,
                                                                blocking=False)
            depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

            frame = None
            detections = []

            startTime = time.monotonic()
            counter = 0
            fps = 0
            color = (255, 255, 255)
            red = (0, 0, 255)
            green = (0, 255, 0)

            # (new)
            s = sched.scheduler(time.time, time.sleep)
            start_time = time.time()

            while True:
                inPreview = previewQueue.get()
                inNN = detectionNNQueue.get()
                depth = depthQueue.get()

                # (new)
                current_time = time.time()
                elapsed_time = current_time - start_time
                notify_time = int(elapsed_time) % 5

                # (new)
                counter += 1
                current_time = time.monotonic()
                if (current_time - startTime) > 1:
                    fps = counter / (current_time - startTime)
                    counter = 0
                    startTime = current_time

                frame = inPreview.getCvFrame()
                depthFrame = depth.getFrame()

                depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depthFrameColor = cv2.equalizeHist(depthFrameColor)
                depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
                detections = inNN.detections
                if len(detections) != 0:
                    boundingBoxMapping = xoutBoundingBoxDepthMapping.get()
                    roiDatas = boundingBoxMapping.getConfigData()

                    for roiData in roiDatas:
                        roi = roiData.roi
                        roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                        topLeft = roi.topLeft()
                        bottomRight = roi.bottomRight()
                        xmin = int(topLeft.x)
                        ymin = int(topLeft.y)
                        xmax = int(bottomRight.x)
                        ymax = int(bottomRight.y)

                        cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color,
                                      cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

                # If the frame is available, draw bounding boxes on it and show the frame
                height = frame.shape[0]
                width = frame.shape[1]
                for detection in detections:
                    # Denormalize bounding box
                    x1 = int(detection.xmin * width)
                    x2 = int(detection.xmax * width)
                    y1 = int(detection.ymin * height)
                    y2 = int(detection.ymax * height)
                    try:
                        label = self.labelMap[detection.label]
                    except:
                        label = detection.label

                    if label == 'crosswalk':
                        color = green
                    else:
                        color = red
                    cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, "{:.2f}".format(detection.confidence * 100), (x1 + 10, y1 + 35),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x) / 10} cm", (x1 + 10, y1 + 50),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y) / 10} cm", (x1 + 10, y1 + 65),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z) / 10} cm", (x1 + 10, y1 + 80),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                msg = None
                if len(detections) == 1:
                    if label == 'crosswalk':
                        msg = 'Red light! Dont Cross Yet'
                        tips = '前方有斑马线'
                        engine.say(tips)
                        engine.runAndWait()
                        dist = '还有%d米' % int(detection.spatialCoordinates.z / 1000)
                        engine.say(dist)
                        engine.runAndWait()
                        color = green
                    cv2.putText(frame, msg, (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

                cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4,
                            color)
                # depthFrameColor = cv2.resize(depthFrameColor, None, fx=0.5, fy=0.5)
                cv2.imshow("depth", depthFrameColor)
                # frame = cv2.resize(frame, None, fx=10, fy=10)
                cv2.imshow("rgb", frame)

                # (new)
                key = cv2.waitKey(1)
                if key == 13:
                    cv2.destroyAllWindows()
                    # if speech is not None:
                    #     speech.task = None
                    #     speech.listen = True
                    break


class V8crosswalkPipeline:
    def __init__(self, modelPath, labelMap, syncNN=True):
        # define some variable
        self.modelPath = modelPath
        self.labelMap = labelMap
        self.syncNN = syncNN

        # Start defining a pipeline
        self.pipeline = dai.Pipeline()
        self.pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2022_1)

        # Define a source - color camera
        camRgb = self.pipeline.createColorCamera()
        camRgb.setPreviewSize(640, 640)
        camRgb.setInterleaved(False)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        # mono cam nodes
        monoLeft = self.pipeline.createMonoCamera()
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

        monoRight = self.pipeline.createMonoCamera()
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # stereo node
        stereo = self.pipeline.createStereoDepth()
        stereo.setConfidenceThreshold(255)
        # stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        # create yolo spatial network node
        spatialDetectionNetwork = self.pipeline.createYoloSpatialDetectionNetwork()
        spatialDetectionNetwork.setBlobPath(self.modelPath)
        spatialDetectionNetwork.setConfidenceThreshold(0.5)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(65535)
        # Yolo specific parameters
        spatialDetectionNetwork.setNumClasses(2)
        spatialDetectionNetwork.setCoordinateSize(4)
        spatialDetectionNetwork.setAnchors(
            [])
        spatialDetectionNetwork.setAnchorMasks(
            {})
        # spatialDetectionNetwork.setAnchors(
        #     np.array([5,6, 8,14, 15,11,10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]))
        # spatialDetectionNetwork.setAnchorMasks(
        #     {"side104": [0, 1, 2], "side52": [3, 4, 5], "side26": [6, 7, 8], "side13": [9, 10, 11]})
        # spatialDetectionNetwork.setAnchors(
        #     np.array([5,6, 8,14, 15,11,10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]))
        # spatialDetectionNetwork.setAnchorMasks(
        #     {"side160": [0, 1, 2],"side80": [3, 4, 5], "side40": [6, 7, 8], "side20": [9, 10, 11]})
        spatialDetectionNetwork.setIouThreshold(0.5)

        # link the nodes
        camRgb.preview.link(spatialDetectionNetwork.input)
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        # output node
        xoutRgb_2 = self.pipeline.createXLinkOut()
        xoutRgb_2.setStreamName("rgb")
        xoutNN = self.pipeline.createXLinkOut()
        xoutNN.setStreamName("detections")
        xoutBoundingBoxDepthMapping = self.pipeline.createXLinkOut()
        xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
        xoutDepth = self.pipeline.createXLinkOut()
        xoutDepth.setStreamName("depth")

        # connect to output node
        if self.syncNN:
            spatialDetectionNetwork.passthrough.link(xoutRgb_2.input)
        else:
            camRgb.preview.link(xoutRgb_2.input)

        spatialDetectionNetwork.out.link(xoutNN.input)
        spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)

        stereo.depth.link(spatialDetectionNetwork.inputDepth)
        spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

    def run(self):
        # Connect and start the pipeline
        with dai.Device(self.pipeline) as device:
            engine = pyttsx3.init()
            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            previewQueue = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
            detectionNNQueue = device.getOutputQueue(name="detections", maxSize=1, blocking=False)
            xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=1,
                                                                blocking=False)
            depthQueue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)

            frame = None
            detections = []

            startTime = time.monotonic()
            counter = 0
            fps = 0
            color = (255, 255, 255)
            red = (0, 0, 255)
            green = (0, 255, 0)

            # (new)
            s = sched.scheduler(time.time, time.sleep)
            start_time = time.time()

            while True:
                inPreview = previewQueue.get()
                inNN = detectionNNQueue.get()
                depth = depthQueue.get()

                # (new)
                current_time = time.time()
                elapsed_time = current_time - start_time
                notify_time = int(elapsed_time) % 5

                # (new)
                counter += 1
                current_time = time.monotonic()
                if (current_time - startTime) > 1:
                    fps = counter / (current_time - startTime)
                    counter = 0
                    startTime = current_time

                frame = inPreview.getCvFrame()
                depthFrame = depth.getFrame()

                depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depthFrameColor = cv2.equalizeHist(depthFrameColor)
                depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
                detections = inNN.detections
                if len(detections) != 0:
                    boundingBoxMapping = xoutBoundingBoxDepthMapping.get()
                    roiDatas = boundingBoxMapping.getConfigData()

                    for roiData in roiDatas:
                        roi = roiData.roi
                        roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                        topLeft = roi.topLeft()
                        bottomRight = roi.bottomRight()
                        xmin = int(topLeft.x)
                        ymin = int(topLeft.y)
                        xmax = int(bottomRight.x)
                        ymax = int(bottomRight.y)

                        cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color,
                                      cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

                # If the frame is available, draw bounding boxes on it and show the frame
                height = frame.shape[0]
                width = frame.shape[1]
                for detection in detections:
                    # Denormalize bounding box
                    x1 = int(detection.xmin * width)
                    x2 = int(detection.xmax * width)
                    y1 = int(detection.ymin * height)
                    y2 = int(detection.ymax * height)
                    try:
                        label = self.labelMap[detection.label]
                    except:
                        label = detection.label

                    if label == 'red':
                        color = red
                    else:
                        color = green
                    cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, "{:.2f}".format(detection.confidence * 100), (x1 + 10, y1 + 35),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x) / 10} cm", (x1 + 10, y1 + 50),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y) / 10} cm", (x1 + 10, y1 + 65),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z) / 10} cm", (x1 + 10, y1 + 80),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                msg = None
                if len(detections) == 1:
                    if label == 'crosswalk':
                        msg = 'Red light! Dont Cross Yet'
                        # tips = '前方有斑马线'
                        # engine.say(tips)
                        # engine.runAndWait()
                        # dist = '还有%d米' % int(detection.spatialCoordinates.z / 1000)
                        # engine.say(dist)
                        # engine.runAndWait()
                        color = green
                    cv2.putText(frame, msg, (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

                cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.4,
                            color)
                depthFrameColor = cv2.resize(depthFrameColor, None, fx=0.5, fy=0.5)
                cv2.imshow("depth", depthFrameColor)
                # frame = cv2.resize(frame, None, fx=10, fy=10)
                cv2.imshow("rgb", frame)

                # (new)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    # if speech is not None:
                    #     speech.task = None
                    #     speech.listen = True
                    break


class TrafficPipeline:
    def __init__(self, modelPath, labelMap, syncNN=True):
        # define some variable
        self.modelPath = modelPath
        self.labelMap = labelMap
        self.syncNN = syncNN

        # Start defining a pipeline
        self.pipeline = dai.Pipeline()
        self.pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2022_1)

        # Define a source - color camera
        camRgb = self.pipeline.createColorCamera()
        camRgb.setPreviewSize(416, 416)
        camRgb.setInterleaved(False)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        # mono cam nodes
        monoLeft = self.pipeline.createMonoCamera()
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

        monoRight = self.pipeline.createMonoCamera()
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # stereo node
        stereo = self.pipeline.createStereoDepth()
        stereo.setConfidenceThreshold(255)
        #将深度图的输出视角，对齐到彩色摄像头的视角
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        #create yolo spatial network node
        spatialDetectionNetwork = self.pipeline.createYoloSpatialDetectionNetwork()
        spatialDetectionNetwork.setBlobPath(self.modelPath)
        spatialDetectionNetwork.setConfidenceThreshold(0.5)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(65535)
        # Yolo specific parameters
        spatialDetectionNetwork.setNumClasses(2)
        spatialDetectionNetwork.setCoordinateSize(4)
        spatialDetectionNetwork.setAnchors(np.array([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]))

        # spatialDetectionNetwork.setAnchorMasks({"side28": np.array([1, 2, 3]), "side14": np.array([3, 4, 5])})
        spatialDetectionNetwork.setAnchorMasks(
            {"side52": [0, 1, 2], "side26": [3, 4, 5], "side13": [6, 7, 8]})
        spatialDetectionNetwork.setIouThreshold(0.1)

        # link the nodes
        camRgb.preview.link(spatialDetectionNetwork.input)
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        # output node
        xoutRgb_2 = self.pipeline.createXLinkOut()
        xoutRgb_2.setStreamName("rgb")
        xoutNN = self.pipeline.createXLinkOut()
        xoutNN.setStreamName("detections")
        xoutBoundingBoxDepthMapping = self.pipeline.createXLinkOut()
        xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
        xoutDepth = self.pipeline.createXLinkOut()
        xoutDepth.setStreamName("depth")

        # connect to output node
        if self.syncNN:
            spatialDetectionNetwork.passthrough.link(xoutRgb_2.input)
        else:
            camRgb.preview.link(xoutRgb_2.input)

        spatialDetectionNetwork.out.link(xoutNN.input)
        spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)

        stereo.depth.link(spatialDetectionNetwork.inputDepth)
        spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

    def run(self):
        # Connect and start the pipeline
        with dai.Device(self.pipeline) as device:
            engine = pyttsx3.init()
            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
            xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4,
                                                                blocking=False)
            depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

            frame = None
            detections = []

            startTime = time.monotonic()
            counter = 0
            fps = 0
            color = (255, 255, 255)
            red = (0, 0, 255)
            green = (0, 255, 0)

            # (new)
            s = sched.scheduler(time.time, time.sleep)
            start_time = time.time()

            while True:
                inPreview = previewQueue.get()
                inNN = detectionNNQueue.get()
                depth = depthQueue.get()

                # (new)
                current_time = time.time()
                elapsed_time = current_time - start_time
                notify_time = int(elapsed_time) % 5

                # (new)
                counter += 1
                current_time = time.monotonic()
                if (current_time - startTime) > 1:
                    fps = counter / (current_time - startTime)
                    counter = 0
                    startTime = current_time

                frame = inPreview.getCvFrame()
                depthFrame = depth.getFrame()

                depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depthFrameColor = cv2.equalizeHist(depthFrameColor)
                depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
                detections = inNN.detections
                if len(detections) != 0:
                    boundingBoxMapping = xoutBoundingBoxDepthMapping.get()
                    roiDatas = boundingBoxMapping.getConfigData()

                    for roiData in roiDatas:
                        roi = roiData.roi
                        roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                        topLeft = roi.topLeft()
                        bottomRight = roi.bottomRight()
                        xmin = int(topLeft.x)
                        ymin = int(topLeft.y)
                        xmax = int(bottomRight.x)
                        ymax = int(bottomRight.y)

                        cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color,
                                      cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

                # If the frame is available, draw bounding boxes on it and show the frame
                height = frame.shape[0]
                width = frame.shape[1]
                for detection in detections:
                    # Denormalize bounding box
                    x1 = int(detection.xmin * width)
                    x2 = int(detection.xmax * width)
                    y1 = int(detection.ymin * height)
                    y2 = int(detection.ymax * height)
                    try:
                        label = self.labelMap[detection.label]
                    except:
                        label = detection.label

                    if label == 'red':
                        color = red
                    else:
                        color = green
                    cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, "{:.2f}".format(detection.confidence * 100), (x1 + 10, y1 + 35),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x) / 10} cm", (x1 + 10, y1 + 50),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y) / 10} cm", (x1 + 10, y1 + 65),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z) / 1000} m", (x1 + 10, y1 + 80),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                msg = None
                if len(detections) == 1:
                    if label == 'red':
                        msg = 'Red light'
                        # tips = '前方红灯'
                        # engine.say(tips)
                        # engine.runAndWait()
                        color = red
                    else:
                        msg = 'Green light,remaining %d m' % int(detection.spatialCoordinates.z/1000)
                        # tips = '前方绿灯'
                        # engine.say(tips)
                        # engine.runAndWait()
                        # dist = '还有%d米' % int(detection.spatialCoordinates.z / 1000)
                        # engine.say(dist)
                        # engine.runAndWait()
                        color = green
                    cv2.putText(frame, msg, (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

                cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4,
                            color)
                # depthFrameColor = cv2.resize(depthFrameColor, None, fx=0.5, fy=0.5)
                cv2.imshow("depth", depthFrameColor)
                # frame = cv2.resize(frame, None, fx=10, fy=10)
                cv2.imshow("rgb", frame)

                # (new)
                key = cv2.waitKey(1)
                if key == 13:
                    cv2.destroyAllWindows()
                    # if speech is not None:
                    #     speech.task = None
                    #     speech.listen = True
                    break


class V5TrafficPipeline:
    def __init__(self, modelPath, labelMap, syncNN=True):
        # define some variable
        self.modelPath = modelPath
        self.labelMap = labelMap
        self.syncNN = syncNN

        # Start defining a pipeline
        self.pipeline = dai.Pipeline()
        self.pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_4)

        # Define a source - color camera
        camRgb = self.pipeline.createColorCamera()
        camRgb.setPreviewSize(640, 640)
        camRgb.setInterleaved(False)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        # mono cam nodes
        monoLeft = self.pipeline.createMonoCamera()
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

        monoRight = self.pipeline.createMonoCamera()
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # stereo node
        stereo = self.pipeline.createStereoDepth()
        stereo.setConfidenceThreshold(255)
        # stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        # create yolo spatial network node
        spatialDetectionNetwork = self.pipeline.createYoloSpatialDetectionNetwork()
        spatialDetectionNetwork.setBlobPath(self.modelPath)
        spatialDetectionNetwork.setConfidenceThreshold(0.5)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(65535)
        # Yolo specific parameters
        spatialDetectionNetwork.setNumClasses(2)
        spatialDetectionNetwork.setCoordinateSize(4)
        # spatialDetectionNetwork.setAnchors(
        #     np.array([10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]))
        # spatialDetectionNetwork.setAnchorMasks(
        #     {"side80": [0, 1, 2], "side40": [3, 4, 5], "side20": [6, 7, 8]})
        # spatialDetectionNetwork.setAnchors(
        #     np.array([5,6, 8,14, 15,11,10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]))
        # spatialDetectionNetwork.setAnchorMasks(
        #     {"side104": [0, 1, 2], "side52": [3, 4, 5], "side26": [6, 7, 8], "side13": [9, 10, 11]})
        spatialDetectionNetwork.setAnchors(
            np.array([10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]))
        spatialDetectionNetwork.setAnchorMasks(
            {"side80": [0, 1, 2], "side40": [3, 4, 5], "side20": [6, 7, 8]})
        # spatialDetectionNetwork.setAnchors(
        #     np.array([5,6, 8,14, 15,11,10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]))
        # spatialDetectionNetwork.setAnchorMasks(
        #     {"side160": [0, 1, 2],"side80": [3, 4, 5], "side40": [6, 7, 8], "side20": [9, 10, 11]})
        spatialDetectionNetwork.setIouThreshold(0.5)

        # link the nodes
        camRgb.preview.link(spatialDetectionNetwork.input)
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        # output node
        xoutRgb_2 = self.pipeline.createXLinkOut()
        xoutRgb_2.setStreamName("rgb")
        xoutNN = self.pipeline.createXLinkOut()
        xoutNN.setStreamName("detections")
        xoutBoundingBoxDepthMapping = self.pipeline.createXLinkOut()
        xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
        xoutDepth = self.pipeline.createXLinkOut()
        xoutDepth.setStreamName("depth")

        # connect to output node
        if self.syncNN:
            spatialDetectionNetwork.passthrough.link(xoutRgb_2.input)
        else:
            camRgb.preview.link(xoutRgb_2.input)

        spatialDetectionNetwork.out.link(xoutNN.input)
        spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)

        stereo.depth.link(spatialDetectionNetwork.inputDepth)
        spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

    def run(self):
        # Connect and start the pipeline
        with dai.Device(self.pipeline) as device:
            engine = pyttsx3.init()
            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            previewQueue = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
            detectionNNQueue = device.getOutputQueue(name="detections", maxSize=1, blocking=False)
            xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=1,
                                                                blocking=False)
            depthQueue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)

            frame = None
            detections = []

            startTime = time.monotonic()
            counter = 0
            fps = 0
            color = (255, 255, 255)
            red = (0, 0, 255)
            green = (0, 255, 0)

            # (new)
            s = sched.scheduler(time.time, time.sleep)
            start_time = time.time()

            while True:
                inPreview = previewQueue.get()
                inNN = detectionNNQueue.get()
                depth = depthQueue.get()

                # (new)
                current_time = time.time()
                elapsed_time = current_time - start_time
                notify_time = int(elapsed_time) % 5

                # (new)
                counter += 1
                current_time = time.monotonic()
                if (current_time - startTime) > 1:
                    fps = counter / (current_time - startTime)
                    counter = 0
                    startTime = current_time

                frame = inPreview.getCvFrame()
                depthFrame = depth.getFrame()

                depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depthFrameColor = cv2.equalizeHist(depthFrameColor)
                depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
                detections = inNN.detections
                if len(detections) != 0:
                    boundingBoxMapping = xoutBoundingBoxDepthMapping.get()
                    roiDatas = boundingBoxMapping.getConfigData()

                    for roiData in roiDatas:
                        roi = roiData.roi
                        roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                        topLeft = roi.topLeft()
                        bottomRight = roi.bottomRight()
                        xmin = int(topLeft.x)
                        ymin = int(topLeft.y)
                        xmax = int(bottomRight.x)
                        ymax = int(bottomRight.y)

                        cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color,
                                      cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

                # If the frame is available, draw bounding boxes on it and show the frame
                height = frame.shape[0]
                width = frame.shape[1]
                for detection in detections:
                    # Denormalize bounding box
                    x1 = int(detection.xmin * width)
                    x2 = int(detection.xmax * width)
                    y1 = int(detection.ymin * height)
                    y2 = int(detection.ymax * height)
                    try:
                        label = self.labelMap[detection.label]
                    except:
                        label = detection.label

                    if label == 'red':
                        color = red
                    else:
                        color = green
                    cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, "{:.2f}".format(detection.confidence * 100), (x1 + 10, y1 + 35),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x) / 10} cm", (x1 + 10, y1 + 50),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y) / 10} cm", (x1 + 10, y1 + 65),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z) / 10} cm", (x1 + 10, y1 + 80),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                msg = None
                if len(detections) == 1:
                    if label == 'red':
                        msg = 'Red light! Dont Cross Yet'
                        tips = '前方红灯'
                        engine.say(tips)
                        engine.runAndWait()
                        color = red
                    else:
                        msg = 'Green light! Cross Now'
                        tips = '前方绿灯'
                        engine.say(tips)
                        engine.runAndWait()
                        dist = '还有%d米' % int(detection.spatialCoordinates.z / 1000)
                        engine.say(dist)
                        engine.runAndWait()
                        color = green
                    cv2.putText(frame, msg, (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

                cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4,
                            color)
                # depthFrameColor = cv2.resize(depthFrameColor, None, fx=0.5, fy=0.5)
                cv2.imshow("depth", depthFrameColor)
                # frame = cv2.resize(frame, None, fx=10, fy=10)
                cv2.imshow("rgb", frame)

                # (new)
                key = cv2.waitKey(1)
                if key == 13:
                    cv2.destroyAllWindows()
                    # if speech is not None:
                    #     speech.task = None
                    #     speech.listen = True
                    break


class V8TrafficPipeline:
    def __init__(self, modelPath, labelMap, syncNN=True):
        # define some variable
        self.modelPath = modelPath
        self.labelMap = labelMap
        self.syncNN = syncNN

        # Start defining a pipeline
        self.pipeline = dai.Pipeline()
        self.pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2022_1)

        # Define a source - color camera
        camRgb = self.pipeline.createColorCamera()
        camRgb.setPreviewSize(640, 640)
        camRgb.setInterleaved(False)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        # mono cam nodes
        monoLeft = self.pipeline.createMonoCamera()
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

        monoRight = self.pipeline.createMonoCamera()
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # stereo node
        stereo = self.pipeline.createStereoDepth()
        stereo.setConfidenceThreshold(255)
        # stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        # create yolo spatial network node
        spatialDetectionNetwork = self.pipeline.createYoloSpatialDetectionNetwork()
        spatialDetectionNetwork.setBlobPath(self.modelPath)
        spatialDetectionNetwork.setConfidenceThreshold(0.5)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(65535)
        # Yolo specific parameters
        spatialDetectionNetwork.setNumClasses(2)
        spatialDetectionNetwork.setCoordinateSize(4)
        spatialDetectionNetwork.setAnchors(
            [])
        spatialDetectionNetwork.setAnchorMasks(
            {})
        # spatialDetectionNetwork.setAnchors(
        #     np.array([5,6, 8,14, 15,11,10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]))
        # spatialDetectionNetwork.setAnchorMasks(
        #     {"side104": [0, 1, 2], "side52": [3, 4, 5], "side26": [6, 7, 8], "side13": [9, 10, 11]})
        # spatialDetectionNetwork.setAnchors(
        #     np.array([5,6, 8,14, 15,11,10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]))
        # spatialDetectionNetwork.setAnchorMasks(
        #     {"side160": [0, 1, 2],"side80": [3, 4, 5], "side40": [6, 7, 8], "side20": [9, 10, 11]})
        spatialDetectionNetwork.setIouThreshold(0.5)

        # link the nodes
        camRgb.preview.link(spatialDetectionNetwork.input)
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        # output node
        xoutRgb_2 = self.pipeline.createXLinkOut()
        xoutRgb_2.setStreamName("rgb")
        xoutNN = self.pipeline.createXLinkOut()
        xoutNN.setStreamName("detections")
        xoutBoundingBoxDepthMapping = self.pipeline.createXLinkOut()
        xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
        xoutDepth = self.pipeline.createXLinkOut()
        xoutDepth.setStreamName("depth")

        # connect to output node
        if self.syncNN:
            spatialDetectionNetwork.passthrough.link(xoutRgb_2.input)
        else:
            camRgb.preview.link(xoutRgb_2.input)

        spatialDetectionNetwork.out.link(xoutNN.input)
        spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)

        stereo.depth.link(spatialDetectionNetwork.inputDepth)
        spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

    def run(self):
        # Connect and start the pipeline
        with dai.Device(self.pipeline) as device:
            engine = pyttsx3.init()
            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            previewQueue = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
            detectionNNQueue = device.getOutputQueue(name="detections", maxSize=1, blocking=False)
            xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=1,
                                                                blocking=False)
            depthQueue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)

            frame = None
            detections = []

            startTime = time.monotonic()
            counter = 0
            fps = 0
            color = (255, 255, 255)
            red = (0, 0, 255)
            green = (0, 255, 0)

            # (new)
            s = sched.scheduler(time.time, time.sleep)
            start_time = time.time()

            while True:
                inPreview = previewQueue.get()
                inNN = detectionNNQueue.get()
                depth = depthQueue.get()

                # (new)
                current_time = time.time()
                elapsed_time = current_time - start_time
                notify_time = int(elapsed_time) % 5

                # (new)
                counter += 1
                current_time = time.monotonic()
                if (current_time - startTime) > 1:
                    fps = counter / (current_time - startTime)
                    counter = 0
                    startTime = current_time

                frame = inPreview.getCvFrame()
                depthFrame = depth.getFrame()

                depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depthFrameColor = cv2.equalizeHist(depthFrameColor)
                depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
                detections = inNN.detections
                if len(detections) != 0:
                    boundingBoxMapping = xoutBoundingBoxDepthMapping.get()
                    roiDatas = boundingBoxMapping.getConfigData()

                    for roiData in roiDatas:
                        roi = roiData.roi
                        roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                        topLeft = roi.topLeft()
                        bottomRight = roi.bottomRight()
                        xmin = int(topLeft.x)
                        ymin = int(topLeft.y)
                        xmax = int(bottomRight.x)
                        ymax = int(bottomRight.y)

                        cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color,
                                      cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

                # If the frame is available, draw bounding boxes on it and show the frame
                height = frame.shape[0]
                width = frame.shape[1]
                for detection in detections:
                    # Denormalize bounding box
                    x1 = int(detection.xmin * width)
                    x2 = int(detection.xmax * width)
                    y1 = int(detection.ymin * height)
                    y2 = int(detection.ymax * height)
                    try:
                        label = self.labelMap[detection.label]
                    except:
                        label = detection.label

                    if label == 'red':
                        color = red
                    else:
                        color = green
                    cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, "{:.2f}".format(detection.confidence * 100), (x1 + 10, y1 + 35),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x) / 10} cm", (x1 + 10, y1 + 50),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y) / 10} cm", (x1 + 10, y1 + 65),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z) / 10} cm", (x1 + 10, y1 + 80),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                msg = None
                if len(detections) == 1:
                    if label == 'red':
                        msg = 'Red light! Dont Cross Yet'
                        tips = '前方红灯'
                        engine.say(tips)
                        engine.runAndWait()
                        color = red
                    else:
                        msg = 'Green light! Cross Now'
                        tips = '前方绿灯'
                        engine.say(tips)
                        engine.runAndWait()
                        dist = '还有%d米' % int(detection.spatialCoordinates.z / 1000)
                        engine.say(dist)
                        engine.runAndWait()
                        color = green
                    cv2.putText(frame, msg, (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

                cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4,
                            color)
                # depthFrameColor = cv2.resize(depthFrameColor, None, fx=0.5, fy=0.5)
                cv2.imshow("depth", depthFrameColor)
                # frame = cv2.resize(frame, None, fx=10, fy=10)
                cv2.imshow("rgb", frame)

                # (new)
                key = cv2.waitKey(1)
                if key == 13:
                    cv2.destroyAllWindows()
                    # if speech is not None:
                    #     speech.task = None
                    #     speech.listen = True
                    break


# -------------------------------------------------------------------------------
# PPE (Protective Equipment) Detection Pipeline
# -------------------------------------------------------------------------------
class PpesPipeline:
    def __init__(self, modelPath, labelMap, syncNN=True):
        # define some variable
        self.modelPath = modelPath
        self.labelMap = labelMap
        self.syncNN = syncNN

        # Start defining a pipeline
        self.pipeline = dai.Pipeline()
        self.pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_2)

        # Define a source - color camera
        camRgb = self.pipeline.createColorCamera()
        camRgb.setPreviewSize(416, 416)
        camRgb.setInterleaved(False)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        # mono cam nodes
        monoLeft = self.pipeline.createMonoCamera()
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

        monoRight = self.pipeline.createMonoCamera()
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # stereo node
        stereo = self.pipeline.createStereoDepth()
        stereo.setConfidenceThreshold(255)

        # create yolo spatial network node
        spatialDetectionNetwork = self.pipeline.createYoloSpatialDetectionNetwork()
        spatialDetectionNetwork.setBlobPath(self.modelPath)
        spatialDetectionNetwork.setConfidenceThreshold(0.5)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(5000)
        # Yolo specific parameters
        spatialDetectionNetwork.setNumClasses(3)
        spatialDetectionNetwork.setCoordinateSize(4)
        spatialDetectionNetwork.setAnchors(np.array([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]))
        spatialDetectionNetwork.setAnchorMasks({"side26": np.array([1, 2, 3]), "side13": np.array([3, 4, 5])})
        spatialDetectionNetwork.setIouThreshold(0.5)

        # link the nodes
        camRgb.preview.link(spatialDetectionNetwork.input)
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        # output node
        xoutRgb_2 = self.pipeline.createXLinkOut()
        xoutRgb_2.setStreamName("rgb")
        xoutNN = self.pipeline.createXLinkOut()
        xoutNN.setStreamName("detections")
        xoutBoundingBoxDepthMapping = self.pipeline.createXLinkOut()
        xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
        xoutDepth = self.pipeline.createXLinkOut()
        xoutDepth.setStreamName("depth")

        # connect to output node
        if self.syncNN:
            spatialDetectionNetwork.passthrough.link(xoutRgb_2.input)
        else:
            camRgb.preview.link(xoutRgb_2.input)

        spatialDetectionNetwork.out.link(xoutNN.input)
        spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)

        stereo.depth.link(spatialDetectionNetwork.inputDepth)
        spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

    def run(self, speech):
        # if speech exist
        if speech is not None:
            speech.listen = True

        # Connect and start the pipeline
        with dai.Device(self.pipeline) as device:

            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
            xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4,
                                                                blocking=False)
            depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

            frame = None
            detections = []

            # default is set to sanitizer
            speech.item = 'sanitizer'

            startTime = time.monotonic()
            counter = 0
            fps = 0
            color = (255, 255, 255)

            # (new)
            s = sched.scheduler(time.time, time.sleep)
            start_time = time.time()

            # (new)
            def direction():
                # initialize variable
                x_command, y_command, z_command = None, None, None

                # X-direction
                if detection.spatialCoordinates.x >= 400:
                    x_command = "Right"
                if 400 > detection.spatialCoordinates.x >= 100:
                    x_command = "Slight Right"

                if detection.spatialCoordinates.x <= -400:
                    x_command = "Left"
                if -400 < detection.spatialCoordinates.x <= -100:
                    x_command = "Slight Left"

                # Y-direction
                if detection.spatialCoordinates.y >= 400:
                    y_command = "Up"
                if 400 > detection.spatialCoordinates.y >= 100:
                    y_command = "Slight Up"

                if detection.spatialCoordinates.y <= -400:
                    y_command = "Down"
                if -400 < detection.spatialCoordinates.y <= -100:
                    y_command = "Slight Down"

                # Z-direction
                if detection.spatialCoordinates.z >= 1000:
                    z_command = "Move forward 1 step"
                if 1000 > detection.spatialCoordinates.z >= 500:
                    z_command = "Move slightly forward"
                if 500 > detection.spatialCoordinates.z >= 0:
                    z_command = "The item is 50cm infront of you"

                return x_command, y_command, z_command

            while True:
                inPreview = previewQueue.get()
                inNN = detectionNNQueue.get()
                depth = depthQueue.get()

                # (new)
                current_time = time.time()
                elapsed_time = current_time - start_time
                notify_time = int(elapsed_time) % 5
                '''
                # (new)
                while speech.item is None:
                    pass
                '''
                # (new)
                counter += 1
                current_time = time.monotonic()
                if (current_time - startTime) > 1:
                    fps = counter / (current_time - startTime)
                    counter = 0
                    startTime = current_time

                frame = inPreview.getCvFrame()
                depthFrame = depth.getFrame()

                depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depthFrameColor = cv2.equalizeHist(depthFrameColor)
                depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
                detections = inNN.detections
                if len(detections) != 0:
                    boundingBoxMapping = xoutBoundingBoxDepthMapping.get()
                    roiDatas = boundingBoxMapping.getConfigData()

                    for roiData in roiDatas:
                        roi = roiData.roi
                        roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                        topLeft = roi.topLeft()
                        bottomRight = roi.bottomRight()
                        xmin = int(topLeft.x)
                        ymin = int(topLeft.y)
                        xmax = int(bottomRight.x)
                        ymax = int(bottomRight.y)

                        cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color,
                                      cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

                # If the frame is available, draw bounding boxes on it and show the frame
                height = frame.shape[0]
                width = frame.shape[1]
                for detection in detections:
                    # Denormalize bounding box
                    x1 = int(detection.xmin * width)
                    x2 = int(detection.xmax * width)
                    y1 = int(detection.ymin * height)
                    y2 = int(detection.ymax * height)
                    try:
                        label = self.labelMap[detection.label]
                    except:
                        label = detection.label

                    # (new)
                    if label == speech.item:
                        cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                        cv2.putText(frame, "{:.2f}".format(detection.confidence * 100), (x1 + 10, y1 + 35),
                                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                        cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x) / 10} cm", (x1 + 10, y1 + 50),
                                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                        cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y) / 10} cm", (x1 + 10, y1 + 65),
                                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                        cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z) / 10} cm", (x1 + 10, y1 + 80),
                                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                        x_command, y_command, z_command = direction()
                        cv2.putText(frame, x_command, (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                        cv2.putText(frame, y_command, (10, 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                        cv2.putText(frame, z_command, (10, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

                cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4,
                            color)
                cv2.imshow("depth", depthFrameColor)
                cv2.imshow("rgb", frame)

                # (new)
                key = cv2.waitKey(1)
                if key == ord('r'):
                    speech.item = None

                if key == 13:
                    cv2.destroyAllWindows()
                    if speech is not None:
                        speech.task = None
                        speech.listen = True
                    break

    def run_without_audio(self):
        # Connect and start the pipeline
        with dai.Device(self.pipeline) as device:

            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
            xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4,
                                                                blocking=False)
            depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

            frame = None
            detections = []

            # default set to sanitizer
            item = 'sanitizer'

            startTime = time.monotonic()
            counter = 0
            fps = 0
            color = (255, 255, 255)

            # (new)
            s = sched.scheduler(time.time, time.sleep)
            start_time = time.time()

            # (new)
            def direction():
                # initialize variable
                x_command, y_command, z_command = None, None, None

                # X-direction
                if detection.spatialCoordinates.x >= 400:
                    x_command = "Right"
                if 400 > detection.spatialCoordinates.x >= 100:
                    x_command = "Slight Right"

                if detection.spatialCoordinates.x <= -400:
                    x_command = "Left"
                if -400 < detection.spatialCoordinates.x <= -100:
                    x_command = "Slight Left"

                # Y-direction
                if detection.spatialCoordinates.y >= 400:
                    y_command = "Up"
                if 400 > detection.spatialCoordinates.y >= 100:
                    y_command = "Slight Up"

                if detection.spatialCoordinates.y <= -400:
                    y_command = "Down"
                if -400 < detection.spatialCoordinates.y <= -100:
                    y_command = "Slight Down"

                # Z-direction
                if detection.spatialCoordinates.z >= 1000:
                    z_command = "Move forward 1 step"
                if 1000 > detection.spatialCoordinates.z >= 500:
                    z_command = "Move slightly forward"
                if 500 > detection.spatialCoordinates.z >= 0:
                    z_command = "The item is 50cm infront of you"

                return x_command, y_command, z_command

            print('Instruction:')
            print('(i) To switch item to be found, press "r"')
            print('(ii) To quit, press "q"\n')
            print('If the button not working:')
            print('- click the any of the image frames first')
            print('- then, try clicking the button again\n')
            while True:
                inPreview = previewQueue.get()
                inNN = detectionNNQueue.get()
                depth = depthQueue.get()

                # (new)
                current_time = time.time()
                elapsed_time = current_time - start_time
                notify_time = int(elapsed_time) % 5

                # (new)
                if item is None:
                    cv2.destroyAllWindows()
                    while True:
                        item = input("Items: sanitizer, facemask, thermometer\nEnter object to find: ")
                        if item in self.labelMap:
                            break
                        print('Invalid Item!')
                    print("Turn around slowly to search for " + item)

                # (new)
                counter += 1
                current_time = time.monotonic()
                if (current_time - startTime) > 1:
                    fps = counter / (current_time - startTime)
                    counter = 0
                    startTime = current_time

                frame = inPreview.getCvFrame()
                depthFrame = depth.getFrame()

                depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depthFrameColor = cv2.equalizeHist(depthFrameColor)
                depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
                detections = inNN.detections
                if len(detections) != 0:
                    boundingBoxMapping = xoutBoundingBoxDepthMapping.get()
                    roiDatas = boundingBoxMapping.getConfigData()

                    for roiData in roiDatas:
                        roi = roiData.roi
                        roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                        topLeft = roi.topLeft()
                        bottomRight = roi.bottomRight()
                        xmin = int(topLeft.x)
                        ymin = int(topLeft.y)
                        xmax = int(bottomRight.x)
                        ymax = int(bottomRight.y)

                        cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color,
                                      cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

                # If the frame is available, draw bounding boxes on it and show the frame
                height = frame.shape[0]
                width = frame.shape[1]
                for detection in detections:
                    # Denormalize bounding box
                    x1 = int(detection.xmin * width)
                    x2 = int(detection.xmax * width)
                    y1 = int(detection.ymin * height)
                    y2 = int(detection.ymax * height)
                    try:
                        label = self.labelMap[detection.label]
                    except:
                        label = detection.label

                    # (new)
                    if label == item:
                        cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                        cv2.putText(frame, "{:.2f}".format(detection.confidence * 100), (x1 + 10, y1 + 35),
                                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                        cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x) / 10} cm", (x1 + 10, y1 + 50),
                                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                        cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y) / 10} cm", (x1 + 10, y1 + 65),
                                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                        cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z) / 10} cm", (x1 + 10, y1 + 80),
                                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                        x_command, y_command, z_command = direction()
                        cv2.putText(frame, x_command, (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                        cv2.putText(frame, y_command, (10, 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                        cv2.putText(frame, z_command, (10, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

                cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4,
                            color)
                cv2.imshow("depth", depthFrameColor)
                cv2.imshow("rgb", frame)

                # (new)
                key = cv2.waitKey(1)
                if key == 13:
                    item = None

                if key == ord('q'):
                    cv2.destroyAllWindows()
                    break


class V8allPipeline:
    def __init__(self, modelPath, labelMap, syncNN=True):
        # define some variable
        self.modelPath = modelPath
        self.labelMap = labelMap
        self.syncNN = syncNN

        # Start defining a pipeline
        self.pipeline = dai.Pipeline()
        self.pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2022_1)

        # Define a source - color camera
        camRgb = self.pipeline.createColorCamera()
        camRgb.setPreviewSize(416, 416)
        camRgb.setInterleaved(False)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        # mono cam nodes
        monoLeft = self.pipeline.createMonoCamera()
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

        monoRight = self.pipeline.createMonoCamera()
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # stereo node
        stereo = self.pipeline.createStereoDepth()
        stereo.setConfidenceThreshold(255)
        # stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        # create yolo spatial network node
        spatialDetectionNetwork = self.pipeline.createYoloSpatialDetectionNetwork()
        spatialDetectionNetwork.setBlobPath(self.modelPath)
        spatialDetectionNetwork.setConfidenceThreshold(0.5)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(65535)
        # Yolo specific parameters
        spatialDetectionNetwork.setNumClasses(5)
        spatialDetectionNetwork.setCoordinateSize(4)
        spatialDetectionNetwork.setAnchors(
            [])
        spatialDetectionNetwork.setAnchorMasks(
            {})
        # spatialDetectionNetwork.setAnchors(
        #     np.array([5,6, 8,14, 15,11,10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]))
        # spatialDetectionNetwork.setAnchorMasks(
        #     {"side104": [0, 1, 2], "side52": [3, 4, 5], "side26": [6, 7, 8], "side13": [9, 10, 11]})
        # spatialDetectionNetwork.setAnchors(
        #     np.array([5,6, 8,14, 15,11,10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]))
        # spatialDetectionNetwork.setAnchorMasks(
        #     {"side160": [0, 1, 2],"side80": [3, 4, 5], "side40": [6, 7, 8], "side20": [9, 10, 11]})
        spatialDetectionNetwork.setIouThreshold(0.5)

        # link the nodes
        camRgb.preview.link(spatialDetectionNetwork.input)
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        # output node
        xoutRgb_2 = self.pipeline.createXLinkOut()
        xoutRgb_2.setStreamName("rgb")
        xoutNN = self.pipeline.createXLinkOut()
        xoutNN.setStreamName("detections")
        xoutBoundingBoxDepthMapping = self.pipeline.createXLinkOut()
        xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
        xoutDepth = self.pipeline.createXLinkOut()
        xoutDepth.setStreamName("depth")

        # connect to output node
        if self.syncNN:
            spatialDetectionNetwork.passthrough.link(xoutRgb_2.input)
        else:
            camRgb.preview.link(xoutRgb_2.input)

        spatialDetectionNetwork.out.link(xoutNN.input)
        spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)

        stereo.depth.link(spatialDetectionNetwork.inputDepth)
        spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

    def run(self):
        # Connect and start the pipeline
        with dai.Device(self.pipeline) as device:
            engine = pyttsx3.init()
            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            previewQueue = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
            detectionNNQueue = device.getOutputQueue(name="detections", maxSize=1, blocking=False)
            xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=1,
                                                                blocking=False)
            depthQueue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)

            frame = None
            detections = []

            startTime = time.monotonic()
            counter = 0
            fps = 0
            color = (255, 255, 255)
            red = (0, 0, 255)
            green = (0, 255, 0)

            # (new)
            s = sched.scheduler(time.time, time.sleep)
            start_time = time.time()

            while True:
                inPreview = previewQueue.get()
                inNN = detectionNNQueue.get()
                depth = depthQueue.get()

                # (new)
                current_time = time.time()
                elapsed_time = current_time - start_time
                notify_time = int(elapsed_time) % 5

                # (new)
                counter += 1
                current_time = time.monotonic()
                if (current_time - startTime) > 1:
                    fps = counter / (current_time - startTime)
                    counter = 0
                    startTime = current_time

                frame = inPreview.getCvFrame()
                depthFrame = depth.getFrame()

                depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depthFrameColor = cv2.equalizeHist(depthFrameColor)
                depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
                detections = inNN.detections
                if len(detections) != 0:
                    boundingBoxMapping = xoutBoundingBoxDepthMapping.get()
                    roiDatas = boundingBoxMapping.getConfigData()

                    for roiData in roiDatas:
                        roi = roiData.roi
                        roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                        topLeft = roi.topLeft()
                        bottomRight = roi.bottomRight()
                        xmin = int(topLeft.x)
                        ymin = int(topLeft.y)
                        xmax = int(bottomRight.x)
                        ymax = int(bottomRight.y)

                        cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color,
                                      cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

                # If the frame is available, draw bounding boxes on it and show the frame
                height = frame.shape[0]
                width = frame.shape[1]
                for detection in detections:
                    # Denormalize bounding box
                    x1 = int(detection.xmin * width)
                    x2 = int(detection.xmax * width)
                    y1 = int(detection.ymin * height)
                    y2 = int(detection.ymax * height)
                    try:
                        label = self.labelMap[detection.label]
                    except:
                        label = detection.label

                    if label == 'red':
                        color = red
                    else:
                        color = green
                    cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, "{:.2f}".format(detection.confidence * 100), (x1 + 10, y1 + 35),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x) / 10} cm", (x1 + 10, y1 + 50),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y) / 10} cm", (x1 + 10, y1 + 65),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z) / 10} cm", (x1 + 10, y1 + 80),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                msg = None
                if len(detections) == 1:
                    if label == 'crosswalk':
                        msg = 'Red light! Dont Cross Yet'
                        tips = '前方有斑马线'
                        engine.say(tips)
                        engine.runAndWait()
                        dist = '还有%d米' % int(detection.spatialCoordinates.z / 1000)
                        engine.say(dist)
                        engine.runAndWait()
                        color = green
                    if label == 'blind_path':
                        msg = 'blind_path! go stright'
                        tips = '前方盲道'
                        engine.say(tips)
                        engine.runAndWait()
                        # if
                        # engine.say(dist)
                        # engine.runAndWait()
                        color = green
                    if label == 'red_light':
                        msg = 'Red light! Dont Cross Yet'
                        tips = '前方红灯'
                        engine.say(tips)
                        engine.runAndWait()
                        color = red
                    if label == 'green_light':
                        msg = 'Green light! Cross Now'
                        tips = '前方绿灯'
                        engine.say(tips)
                        engine.runAndWait()
                        dist = '还有%d米' % int(detection.spatialCoordinates.z / 1000)
                        engine.say(dist)
                        engine.runAndWait()
                        color = green
                    cv2.putText(frame, msg, (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

                cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.4,
                            color)
                depthFrameColor = cv2.resize(depthFrameColor, None, fx=0.5, fy=0.5)
                cv2.imshow("depth", depthFrameColor)
                # frame = cv2.resize(frame, None, fx=10, fy=10)
                cv2.imshow("rgb", frame)

                # (new)
                key = cv2.waitKey(1)
                if key == 13:
                    cv2.destroyAllWindows()
                    # if speech is not None:
                    #     speech.task = None
                    #     speech.listen = True
                    break


class V5allPipeline:
    def __init__(self, modelPath, labelMap, syncNN=True):
        # define some variable
        self.modelPath = modelPath
        self.labelMap = labelMap
        self.syncNN = syncNN

        # Start defining a pipeline
        self.pipeline = dai.Pipeline()
        self.pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2022_1)

        # Define a source - color camera
        camRgb = self.pipeline.createColorCamera()
        camRgb.setPreviewSize(416, 416)
        camRgb.setInterleaved(False)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        # mono cam nodes
        monoLeft = self.pipeline.createMonoCamera()
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

        monoRight = self.pipeline.createMonoCamera()
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # stereo node
        stereo = self.pipeline.createStereoDepth()
        stereo.setConfidenceThreshold(255)
        # stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        # create yolo spatial network node
        spatialDetectionNetwork = self.pipeline.createYoloSpatialDetectionNetwork()
        spatialDetectionNetwork.setBlobPath(self.modelPath)
        spatialDetectionNetwork.setConfidenceThreshold(0.75)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(65535)
        # Yolo specific parameters
        spatialDetectionNetwork.setNumClasses(5)
        spatialDetectionNetwork.setCoordinateSize(4)
        # spatialDetectionNetwork.setAnchors(
        #     [])
        # spatialDetectionNetwork.setAnchorMasks(
        #     {})
        spatialDetectionNetwork.setAnchors(
            np.array([10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]))
        spatialDetectionNetwork.setAnchorMasks(
            {"side52": [0, 1, 2], "side26": [3, 4, 5], "side13": [6, 7, 8]})
        # spatialDetectionNetwork.setAnchors(
        #     np.array([5,6, 8,14, 15,11,10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]))
        # spatialDetectionNetwork.setAnchorMasks(
        #     {"side104": [0, 1, 2], "side52": [3, 4, 5], "side26": [6, 7, 8], "side13": [9, 10, 11]})
        # spatialDetectionNetwork.setAnchors(
        #     np.array([5,6, 8,14, 15,11,10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]))
        # spatialDetectionNetwork.setAnchorMasks(
        #     {"side160": [0, 1, 2],"side80": [3, 4, 5], "side40": [6, 7, 8], "side20": [9, 10, 11]})
        spatialDetectionNetwork.setIouThreshold(0.1)

        # link the nodes
        camRgb.preview.link(spatialDetectionNetwork.input)
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        # output node
        xoutRgb_2 = self.pipeline.createXLinkOut()
        xoutRgb_2.setStreamName("rgb")
        xoutNN = self.pipeline.createXLinkOut()
        xoutNN.setStreamName("detections")
        xoutBoundingBoxDepthMapping = self.pipeline.createXLinkOut()
        xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
        xoutDepth = self.pipeline.createXLinkOut()
        xoutDepth.setStreamName("depth")

        # connect to output node
        if self.syncNN:
            spatialDetectionNetwork.passthrough.link(xoutRgb_2.input)
        else:
            camRgb.preview.link(xoutRgb_2.input)

        spatialDetectionNetwork.out.link(xoutNN.input)
        spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)

        stereo.depth.link(spatialDetectionNetwork.inputDepth)
        spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

    def run(self):
        # Connect and start the pipeline
        with dai.Device(self.pipeline) as device:
            engine = pyttsx3.init()
            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            previewQueue = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
            detectionNNQueue = device.getOutputQueue(name="detections", maxSize=1, blocking=False)
            xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=1,
                                                                blocking=False)
            depthQueue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)

            frame = None
            detections = []

            startTime = time.monotonic()
            counter = 0
            fps = 0
            color = (255, 255, 255)
            red = (0, 0, 255)
            green = (0, 255, 0)

            # (new)
            s = sched.scheduler(time.time, time.sleep)
            start_time = time.time()

            while True:
                inPreview = previewQueue.get()
                inNN = detectionNNQueue.get()
                depth = depthQueue.get()

                # (new)
                current_time = time.time()
                elapsed_time = current_time - start_time
                notify_time = int(elapsed_time) % 5

                # (new)
                counter += 1
                current_time = time.monotonic()
                if (current_time - startTime) > 1:
                    fps = counter / (current_time - startTime)
                    counter = 0
                    startTime = current_time

                frame = inPreview.getCvFrame()
                depthFrame = depth.getFrame()

                depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depthFrameColor = cv2.equalizeHist(depthFrameColor)
                depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
                detections = inNN.detections
                if len(detections) != 0:
                    boundingBoxMapping = xoutBoundingBoxDepthMapping.get()
                    roiDatas = boundingBoxMapping.getConfigData()

                    for roiData in roiDatas:
                        roi = roiData.roi
                        roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                        topLeft = roi.topLeft()
                        bottomRight = roi.bottomRight()
                        xmin = int(topLeft.x)
                        ymin = int(topLeft.y)
                        xmax = int(bottomRight.x)
                        ymax = int(bottomRight.y)

                        cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color,
                                      cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

                # If the frame is available, draw bounding boxes on it and show the frame
                height = frame.shape[0]
                width = frame.shape[1]
                for detection in detections:
                    # Denormalize bounding box
                    x1 = int(detection.xmin * width)
                    x2 = int(detection.xmax * width)
                    y1 = int(detection.ymin * height)
                    y2 = int(detection.ymax * height)
                    try:
                        label = self.labelMap[detection.label]
                    except:
                        label = detection.label

                    if label == 'red':
                        color = red
                    else:
                        color = green
                    cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, "{:.2f}".format(detection.confidence * 100), (x1 + 10, y1 + 35),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x) / 10} cm", (x1 + 10, y1 + 50),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y) / 10} cm", (x1 + 10, y1 + 65),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z) / 1000} m", (x1 + 10, y1 + 80),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                msg = None
                if len(detections) == 1:
                    if label == 'crosswalk':
                        msg = 'crosswalk turn %s and remaining %d m' % (s, int(detection.spatialCoordinates.z/1000))
                        if int(detection.spatialCoordinates.x) < -150:
                            s = 'left'
                            # tips = '左前方有斑马线'
                            # engine.say(tips)
                            # engine.runAndWait()
                        elif int(detection.spatialCoordinates.x) > 150:
                            s = 'right'
                            # tips = '右前方有斑马线'
                            # engine.say(tips)
                            # engine.runAndWait()
                        else:
                            s = 'mid'
                            # tips = '前方有斑马线'
                            # engine.say(tips)
                            # engine.runAndWait()
                        # dist = '还有%d米' % int(detection.spatialCoordinates.z / 1000)
                        # engine.say(dist)
                        # engine.runAndWait()
                        color = green
                    if label == 'blind_path':
                        msg = 'blind_path turn %s and remaining %d m' % (s, int(detection.spatialCoordinates.z/1000))
                        # tips = '前方盲道'
                        # engine.say(tips)
                        # engine.runAndWait()
                        if int(detection.spatialCoordinates.x) > 150:
                            s = 'right'
                            # dist = '右侧%d米' % int(detection.spatialCoordinates.x / 1000)
                        elif int(detection.spatialCoordinates.x) < -150:
                            s = 'left'
                            # dist = '左侧%d米' % int(detection.spatialCoordinates.x / 1000)
                        else:
                            s = 'mid'
                            # dist = '还有%d米' % int(detection.spatialCoordinates.x / 1000)
                        # engine.say(dist)
                        # engine.runAndWait()
                        color = green
                    if label == 'red_light':
                        msg = 'Red light'
                        # tips = '前方红灯'
                        # engine.say(tips)
                        # engine.runAndWait()
                        color = red
                    if label == 'green_light':
                        msg = 'Green light，remaining %d m' % int(detection.spatialCoordinates.z/1000)
                        # tips = '前方绿灯'
                        # engine.say(tips)
                        # engine.runAndWait()
                        # dist = '还有%d米' % int(detection.spatialCoordinates.z / 1000)
                        # engine.say(dist)
                        # engine.runAndWait()
                        color = green
                    cv2.putText(frame, msg, (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

                cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.4,
                            color)
                depthFrameColor = cv2.resize(depthFrameColor, None, fx=0.5, fy=0.5)
                cv2.imshow("depth", depthFrameColor)
                # frame = cv2.resize(frame, None, fx=10, fy=10)
                cv2.imshow("rgb", frame)

                # (new)
                key = cv2.waitKey(1)
                if key == 13:
                    cv2.destroyAllWindows()
                    # if speech is not None:
                    #     speech.task = None
                    #     speech.listen = True
                    break


class V5blindPipeline:
    def __init__(self, modelPath, labelMap, syncNN=True):
        # define some variable
        self.modelPath = modelPath
        self.labelMap = labelMap
        self.syncNN = syncNN

        # Start defining a pipeline
        self.pipeline = dai.Pipeline()
        self.pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2022_1)

        # Define a source - color camera
        camRgb = self.pipeline.createColorCamera()
        camRgb.setPreviewSize(416, 416)
        camRgb.setInterleaved(False)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        # mono cam nodes
        monoLeft = self.pipeline.createMonoCamera()
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

        monoRight = self.pipeline.createMonoCamera()
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # stereo node
        stereo = self.pipeline.createStereoDepth()
        stereo.setConfidenceThreshold(255)
        # stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        # create yolo spatial network node
        spatialDetectionNetwork = self.pipeline.createYoloSpatialDetectionNetwork()
        spatialDetectionNetwork.setBlobPath(self.modelPath)
        spatialDetectionNetwork.setConfidenceThreshold(0.75)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(65535)
        # Yolo specific parameters
        spatialDetectionNetwork.setNumClasses(2)
        spatialDetectionNetwork.setCoordinateSize(4)
        # spatialDetectionNetwork.setAnchors(
        #     [])
        # spatialDetectionNetwork.setAnchorMasks(
        #     {})
        spatialDetectionNetwork.setAnchors(
            np.array([10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]))
        spatialDetectionNetwork.setAnchorMasks(
            {"side52": [0, 1, 2], "side26": [3, 4, 5], "side13": [6, 7, 8]})
        # spatialDetectionNetwork.setAnchors(
        #     np.array([5,6, 8,14, 15,11,10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]))
        # spatialDetectionNetwork.setAnchorMasks(
        #     {"side104": [0, 1, 2], "side52": [3, 4, 5], "side26": [6, 7, 8], "side13": [9, 10, 11]})
        # spatialDetectionNetwork.setAnchors(
        #     np.array([5,6, 8,14, 15,11,10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]))
        # spatialDetectionNetwork.setAnchorMasks(
        #     {"side160": [0, 1, 2],"side80": [3, 4, 5], "side40": [6, 7, 8], "side20": [9, 10, 11]})
        spatialDetectionNetwork.setIouThreshold(0.1)

        # link the nodes
        camRgb.preview.link(spatialDetectionNetwork.input)
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        # output node
        xoutRgb_2 = self.pipeline.createXLinkOut()
        xoutRgb_2.setStreamName("rgb")
        xoutNN = self.pipeline.createXLinkOut()
        xoutNN.setStreamName("detections")
        xoutBoundingBoxDepthMapping = self.pipeline.createXLinkOut()
        xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
        xoutDepth = self.pipeline.createXLinkOut()
        xoutDepth.setStreamName("depth")

        # connect to output node
        if self.syncNN:
            spatialDetectionNetwork.passthrough.link(xoutRgb_2.input)
        else:
            camRgb.preview.link(xoutRgb_2.input)

        spatialDetectionNetwork.out.link(xoutNN.input)
        spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)

        stereo.depth.link(spatialDetectionNetwork.inputDepth)
        spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

    def run(self):
        # Connect and start the pipeline
        with dai.Device(self.pipeline) as device:
            engine = pyttsx3.init()
            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            previewQueue = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
            detectionNNQueue = device.getOutputQueue(name="detections", maxSize=1, blocking=False)
            xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=1,
                                                                blocking=False)
            depthQueue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)

            frame = None
            detections = []

            startTime = time.monotonic()
            counter = 0
            fps = 0
            color = (255, 255, 255)
            red = (0, 0, 255)
            green = (0, 255, 0)

            # (new)
            s = sched.scheduler(time.time, time.sleep)
            start_time = time.time()

            while True:
                inPreview = previewQueue.get()
                inNN = detectionNNQueue.get()
                depth = depthQueue.get()

                # (new)
                current_time = time.time()
                elapsed_time = current_time - start_time
                notify_time = int(elapsed_time) % 5

                # (new)
                counter += 1
                current_time = time.monotonic()
                if (current_time - startTime) > 1:
                    fps = counter / (current_time - startTime)
                    counter = 0
                    startTime = current_time

                frame = inPreview.getCvFrame()
                depthFrame = depth.getFrame()

                depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depthFrameColor = cv2.equalizeHist(depthFrameColor)
                depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
                detections = inNN.detections
                if len(detections) != 0:
                    boundingBoxMapping = xoutBoundingBoxDepthMapping.get()
                    roiDatas = boundingBoxMapping.getConfigData()

                    for roiData in roiDatas:
                        roi = roiData.roi
                        roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                        topLeft = roi.topLeft()
                        bottomRight = roi.bottomRight()
                        xmin = int(topLeft.x)
                        ymin = int(topLeft.y)
                        xmax = int(bottomRight.x)
                        ymax = int(bottomRight.y)

                        cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color,
                                      cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

                # If the frame is available, draw bounding boxes on it and show the frame
                height = frame.shape[0]
                width = frame.shape[1]
                for detection in detections:
                    # Denormalize bounding box
                    x1 = int(detection.xmin * width)
                    x2 = int(detection.xmax * width)
                    y1 = int(detection.ymin * height)
                    y2 = int(detection.ymax * height)
                    try:
                        label = self.labelMap[detection.label]
                    except:
                        label = detection.label

                    if label == 'red':
                        color = red
                    else:
                        color = green
                    cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, "{:.2f}".format(detection.confidence * 100), (x1 + 10, y1 + 35),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x) / 10} cm", (x1 + 10, y1 + 50),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y) / 10} cm", (x1 + 10, y1 + 65),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z) / 1000} m", (x1 + 10, y1 + 80),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                msg = None
                if len(detections) == 1:
                    if label == 'crosswalk':
                        msg = 'crosswalk turn %s and remaining %d m' % (s, int(detection.spatialCoordinates.z/1000))
                        if int(detection.spatialCoordinates.x) < -150:
                            s = 'left'
                            # tips = '左前方有斑马线'
                            # engine.say(tips)
                            # engine.runAndWait()
                        elif int(detection.spatialCoordinates.x) > 150:
                            s = 'right'
                            # tips = '右前方有斑马线'
                            # engine.say(tips)
                            # engine.runAndWait()
                        else:
                            s = 'mid'
                            # tips = '前方有斑马线'
                            # engine.say(tips)
                            # engine.runAndWait()
                        # dist = '还有%d米' % int(detection.spatialCoordinates.z / 1000)
                        # engine.say(dist)
                        # engine.runAndWait()
                        color = green
                    else:
                        msg = 'blind_path turn %s and remaining %d m' % (s, int(detection.spatialCoordinates.z/1000))
                        # tips = '前方盲道'
                        # engine.say(tips)
                        # engine.runAndWait()
                        if int(detection.spatialCoordinates.x) > 150:
                            s = 'right'
                            # dist = '右侧%d米' % int(detection.spatialCoordinates.x / 1000)
                        elif int(detection.spatialCoordinates.x) < -150:
                            s = 'left'
                            # dist = '左侧%d米' % int(detection.spatialCoordinates.x / 1000)
                        else:
                            s = 'mid'
                            # dist = '还有%d米' % int(detection.spatialCoordinates.x / 1000)
                        # engine.say(dist)
                        # engine.runAndWait()
                        color = green
                    cv2.putText(frame, msg, (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

                cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.4,
                            color)
                depthFrameColor = cv2.resize(depthFrameColor, None, fx=0.5, fy=0.5)
                cv2.imshow("depth", depthFrameColor)
                # frame = cv2.resize(frame, None, fx=10, fy=10)
                cv2.imshow("rgb", frame)

                # (new)
                key = cv2.waitKey(1)
                if key == 13:
                    cv2.destroyAllWindows()
                    # if speech is not None:
                    #     speech.task = None
                    #     speech.listen = True
                    break


import cv2
import base64
import keyboard
from volcenginesdkarkruntime import Ark

import depthai as dai

from depthai import Pipeline, Device


import pyaudio
import wave
import sys

class AIPipeline:
    def __init__(self):
        self.client = Ark(api_key="f317e062-879b-40af-b520-0f235cf2bf94")
        self.pipeline = dai.Pipeline()
        self._configure_pipeline() 

    def _configure_pipeline(self):
        """配置相机管道"""
        cam = self.pipeline.createColorCamera()
        cam.setPreviewSize(640, 480)
        cam.setInterleaved(False)
        
        xout = self.pipeline.createXLinkOut()
        xout.setStreamName("rgb")
        cam.preview.link(xout.input)

    def _get_frame(self, device):
        """从设备获取帧"""
        q_rgb = device.getOutputQueue("rgb", maxSize=1, blocking=True)
        return q_rgb.get().getCvFrame()

    def _analyze_frame(self, frame):
        """分析图像内容"""
        _, buffer = cv2.imencode('.jpg', frame)
        base64_img = base64.b64encode(buffer).decode()
        
        response = self.client.chat.completions.create(
            model="doubao-1-5-vision-pro-32k-250115",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_img}"},
                    {"type": "text", "text": "描述场景中的主要物体及其空间分布"}
                ]
            }]
        )
        return response.choices[0].message.content

    def run(self):
        """主运行循环"""
        try:
            with dai.Device(self.pipeline) as device:
                print("相机已启动 (按S截图/ESC退出)")
                
                while True:
                    frame = self._get_frame(device)
                    cv2.imshow("OAK-D View", frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('s'):
                        analysis = self._analyze_frame(frame)
                        print(f"\n场景分析: {analysis}")
                        multiruntest(1, analysis)
                        play_pcm_file("/home/pi/Desktop/wtcane/tests/yuyin.pcm",16600,1,2)
                        #cv2.imwrite(f'snapshot_{len(glob.glob("snapshot_*.jpg"))+1}.jpg', frame)
                    elif key == 27:  # ESC键
                        break
                    
                        
        except Exception as e:
            print(f"设备错误: {str(e)}")
        finally:
            cv2.destroyAllWindows()
            
import time
import threading
import sys

#import nls

URL="wss://nls-gateway-cn-shanghai.aliyuncs.com/ws/v1"
TOKEN="a5b00368b204495cbbc428caabf87391"  #参考https://help.aliyun.com/document_detail/450255.html获取token
APPKEY="eoSYciBNQx6aiELj"       #获取Appkey请前往控制台：https://nls-portal.console.aliyun.com/applist

class TestTts:
    def __init__(self, tid, test_file):
        self.__th = threading.Thread(target=self.__test_run)
        self.__id = tid
        self.__test_file = test_file
   
    def start(self, text):
        self.__text = text
        self.__f = open(self.__test_file, "wb")
        self.__th.start()
    
    def test_on_metainfo(self, message, *args):
        print("on_metainfo message=>{}".format(message))  

    def test_on_error(self, message, *args):
        print("on_error args=>{}".format(args))

    def test_on_close(self, *args):
        print("on_close: args=>{}".format(args))
        try:
            self.__f.close()
        except Exception as e:
            print("close file failed since:", e)

    def test_on_data(self, data, *args):
        try:
            self.__f.write(data)
        except Exception as e:
            print("write data failed:", e)

    def test_on_completed(self, message, *args):
        print("on_completed:args=>{} message=>{}".format(args, message))


    def __test_run(self):
      	print("thread:{} start..".format(self.__id))
      	tts = nls.NlsSpeechSynthesizer(url=URL,
      	      	      	      	       token=TOKEN,
      	      	      	      	       appkey=APPKEY,
      	      	      	      	       on_metainfo=self.test_on_metainfo,
      	      	      	      	       on_data=self.test_on_data,
      	      	      	      	       on_completed=self.test_on_completed,
      	      	      	      	       on_error=self.test_on_error,
      	      	      	      	       on_close=self.test_on_close,
      	      	      	      	       callback_args=[self.__id])
      	print("{}: session start".format(self.__id))
      	r = tts.start(self.__text, voice="ailun")
      	print("{}: tts done with result:{}".format(self.__id, r))

def multiruntest(num=500, text=""):
    for i in range(0, num):
        name = "thread" + str(i)
        t = TestTts(name, "tests/yuyin.pcm")
        t.start(text)


def play_pcm_file(pcm_path, sample_rate=44100, channels=1, sample_width=2):
    """播放PCM音频文件
    :param pcm_path: PCM文件路径
    :param sample_rate: 采样率(默认44100Hz)
    :param channels: 声道数(默认1)
    :param sample_width: 采样位宽(默认2字节/16bit)
    """
    p = pyaudio.PyAudio()
    
    try:
        # 打开PCM文件
        with open(pcm_path, 'rb') as f:
            # 创建音频流
            stream = p.open(
                format=p.get_format_from_width(sample_width),
                channels=channels,
                rate=sample_rate,
                output=True
            )
            
            print(f"正在播放 {pcm_path} (按Ctrl+C停止)...")
            
            # 流式播放(每次读取1024个样本)
            data = f.read(1024 * sample_width * channels)
            while data:
                stream.write(data)
                data = f.read(1024 * sample_width * channels)
                
    except KeyboardInterrupt:
        print("\n播放已停止")
    except Exception as e:
        print(f"播放出错: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
