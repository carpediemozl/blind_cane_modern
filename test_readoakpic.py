
import cv2
import base64
import keyboard
from volcenginesdkarkruntime import Ark

import depthai as dai

from depthai import Pipeline, Device

import pyaudio
import wave
import sys

from aliyuyin import TestTts

import threading
import time

class OakdSnapshot:
    
    
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
                        text_audio(analysis)
                        play_pcm_file("/home/pi/Desktop/wtcane/tests/0924.wav",16600,1,2)
                        #cv2.imwrite(f'snapshot_{len(glob.glob("snapshot_*.jpg"))+1}.jpg', frame)
                    elif key == 27:  # ESC键
                        break
                        
        except Exception as e:
            print(f"设备错误: {str(e)}")
        finally:
            cv2.destroyAllWindows()
            



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

def text_audio(TEXT,num=500):
    for i in range(0, num):
        name = "thread" + str(i)
        t = TestTts(name, "tests/0924.wav")
        t.start(TEXT)

def worker(thread_id):
    print(f"线程 {thread_id} 开始工作")
    time.sleep(2)  # 模拟耗时操作
    print(f"线程 {thread_id} 工作完成")

if __name__ == "__main__":
    threads = []
    for i in range(3):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    print("所有线程执行完毕")


if __name__ == "__main__":
    camera =OakdSnapshot()
    camera.run()
