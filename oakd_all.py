from depthai_utils import V5allPipeline

modelPath = r'models/yolov8n_v6.blob'
label = ['crosswalk', 'guide_arrows', 'blind_path', 'red_light', 'green_light']
crosswalkPipeline = V5allPipeline(modelPath, label)

from pynput import keyboard
from ai_readoakpic import OakdVisionTTS
ailine = OakdVisionTTS()


from depthai_utils import NavPipeline

model_path = r'models/mobilenet-ssd_openvino_2021.2_6shave.blob'

# -------------------------------------------------------------------------------
# MobilenetSSD label nnLabels
# -------------------------------------------------------------------------------
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

navPipeline = NavPipeline(nnPath=model_path, labelMap=labelMap)

from depthai_utils import SegmentPipeline

segmentationModelPath = r'models/path_segmentation.blob'
segmentPipeline = SegmentPipeline(segmentationModelPath)

from depthai_utils import TrafficPipeline

trafficModelPath = r'models/blob/trafficlight_frozen_darknet_yolov4_model_openvino_2021.3_5shave.blob'
label = ['red', 'green']
trafficPipeline = TrafficPipeline(trafficModelPath, label)


def on_press(key):
    try:
        if key == keyboard.Key.space:  # 例如，按下空格键
            print("空格键被按下")
            # 在这里添加你的切换指令或操作
            # 例如，切换到另一个程序或窗口
        elif key == keyboard.KeyCode.from_char('1'):  # 例如，按下'a'键
            print("1被按下")
            try:
                ailine.run()
            except:
                raise
                
        #elif key == keyboard.KeyCode.from_char('2'):  # 例如，按下'a'键
            #print("2被按下")
            #try:
                #crosswalkPipeline.run()
            #except:
                #raise
                
        elif key == keyboard.KeyCode.from_char('3'):  # 例如，按下'a'键
            print("3被按下")
            try:
                navPipeline.run()
            except:
                raise
            
        elif key == keyboard.KeyCode.from_char('4'):  # 例如，按下'a'键
            print("4被按下")
            try:
                segmentPipeline.run()
            except:
                raise
        
        elif key == keyboard.KeyCode.from_char('7'):  # 例如，按下'a'键
            print("7被按下")
            try:
                trafficPipeline.run()
            except:
                raise
                
            # 执行其他操作
    except AttributeError:
        pass  # 忽略非按键的异常，例如按键释放事件
 
def on_release(key):
    print('{0} released'.format(key))
    if key == keyboard.KeyCode.from_char('9'):  # 停止监听，按下ESC退出
        return False
 
# 收集事件，这里会一直运行直到返回False
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
