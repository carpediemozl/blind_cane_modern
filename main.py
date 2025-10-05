from depthai_utils import NavPipeline, SegmentPipeline, TrafficPipeline, PpesPipeline, V4crosswalkPipeline
from task_utils import SpeechRecognizer
import cv2
from depthai_utils import CrossWalkPipeline
import keyboard
import pyttsx3

# crosswalk model path
crosswalkmodelPath = r'models/test2.blob'
crosswalklabelMap = ['crosswalk', 'guide_arrows']
crosswalkPipeilne = CrossWalkPipeline(modelPath=crosswalkmodelPath,
                                      labelMap=crosswalklabelMap)
v4crosswalkmodelPath = r'models/frozen_darknet_yolov4_model_openvino_2021.4_5shave.blob'
v4crosswalklabelMap = ['crosswalk', 'guide_arrows']
v4crosswalkPipeilne = V4crosswalkPipeline(modelPath=v4crosswalkmodelPath,
                                        labelMap=v4crosswalklabelMap)
# -------------------------------------------------------------------------------
# define the Pipelines
# -------------------------------------------------------------------------------
# navigation (obstacle avoidance + social distancing)
commonObjectPath = r'models/mobilenet-ssd_openvino_2021.2_6shave.blob'
commonObjectLabelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
                        "cow",
                        "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
                        "tvmonitor"]
navPipeline = NavPipeline(nnPath=commonObjectPath,
                          labelMap=commonObjectLabelMap)

# walkable path segmentation
segModelPath = r'models\path_segmentation.blob'
segmentPipeline = SegmentPipeline(segModelPath=segModelPath)

# model to detect pedestrian traffic light
trafficModelPath = r'models/trafficlight_frozen_darknet_yolov4_model_openvino_2021.3_5shave.blob'
trafficLabelMap = ['red', 'green']
trafficPipeline = TrafficPipeline(modelPath=trafficModelPath,
                                  labelMap=trafficLabelMap)

# model to detect protective equipment (ppe)
ppesModelPath = r'models/peps_model.blob'
ppesLabelMap = ["sanitizer", "facemask", "thermometer"]
syncNN = True
ppesPipeline = PpesPipeline(modelPath=ppesModelPath,
                            labelMap=ppesLabelMap,
                            syncNN=syncNN)
engine = pyttsx3.init()
# run the speech recognizer
# speech = SpeechRecognizer()
# speech.start()

try:
    # trafficPipeline.run(speech)
    # while True:
    #     if speech.task == 'navigation':
    #         navPipeline.run(speech)
    #     if speech.task == 'segmentation':
    #         segmentPipeline.run(speech)
    #     if speech.task == 'traffic light':
    #         crosswalkPipeilne.run(speech)
    #     if speech.task == 'search':
    #         ppesPipeline.run(speech)
    while True:
        if keyboard.is_pressed('w'):
            engine.say("开始红绿灯识别")
            engine.runAndWait()
            print('开始红绿灯识别')
            trafficPipeline.run()
        elif keyboard.is_pressed('s'):
            engine.say("开始斑马线识别")
            engine.runAndWait()
            print('开始斑马线识别')
            crosswalkPipeilne.run()
        elif keyboard.is_pressed('p'):
            engine.say("开始斑马线识别")
            engine.runAndWait()
            print('开始斑马线识别')
            v4crosswalkPipeilne.run()
        elif keyboard.is_pressed('a'):
            engine.say("开始道路分割")
            engine.runAndWait()
            print('开始道路分割')
            segmentPipeline.run()
        elif keyboard.is_pressed('d'):
            engine.say("开始避障")
            engine.runAndWait()
            print('开始避障')
            navPipeline.run()
        elif keyboard.is_pressed('x'):  # if key 'enter' is pressed
            engine.say("进入目标检测，按r切换")
            engine.runAndWait()
            print('进入目标检测，按r切换')
            ppesPipeline.run_without_audio()
        elif keyboard.is_pressed('q'):
            engine.say("退出")
            engine.runAndWait()
            print('Quit!')
            break
except:
    cv2.destroyAllWindows()
    # speech.end = True
    # speech.join()
    raise
