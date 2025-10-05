#self.TOKEN = "edc00fccc66d436a92d00fda4dbb0e05"
#  self.APPKEY = "eoSYciBNQx6aiELj"

import cv2
import base64
import threading
import pyaudio
import wave
import nls
from volcenginesdkarkruntime import Ark
import depthai as dai

class OakdVisionTTS:
    def __init__(self):
        # 视觉分析配置
        self.vision_client = Ark(api_key="f317e062-879b-40af-b520-0f235cf2bf94")
        
        # 语音合成配置
        self.TOKEN = "4173a0c15b3146be83dccb5ab85460ee"
        self.APPKEY = "eoSYciBNQx6aiELj"
        self.URL = "wss://nls-gateway-cn-shanghai.aliyuncs.com/ws/v1"
        self.voice = "xiaoyun"
        
        # 音频播放配置
        self.audio = pyaudio.PyAudio()
        self.playback_thread = None
        
        # 相机初始化
        self.pipeline = dai.Pipeline()
        self._configure_pipeline()
        self.analysis_thread = None
        self.tts_thread = None
        self.latest_frame = None

    def _configure_pipeline(self):
        cam = self.pipeline.createColorCamera()
        cam.setPreviewSize(640, 480)
        cam.setInterleaved(False)
        xout = self.pipeline.createXLinkOut()
        xout.setStreamName("rgb")
        cam.preview.link(xout.input)

    def _get_frame(self, device):
        q_rgb = device.getOutputQueue("rgb", maxSize=1, blocking=True)
        return q_rgb.get().getCvFrame()

    def _play_audio(self, file_path, sample_rate=16800, channels=1, sample_width=2):
        """音频播放线程"""
        def playback():
            
            """播放PCM音频文件
            :param pcm_path: PCM文件路径
            :param sample_rate: 采样率(默认44100Hz)
            :param channels: 声道数(默认1)
            :param sample_width: 采样位宽(默认2字节/16bit)
            """
            p = pyaudio.PyAudio()
            
            try:
                # 打开PCM文件
                with open(file_path, 'rb') as f:
                    # 创建音频流
                    stream = p.open(
                        format=p.get_format_from_width(sample_width),
                        channels=channels,
                        rate=sample_rate,
                        output=True
                    )
                    
                    print(f"正在播放 {file_path} (按Ctrl+C停止)...")
                    
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
        
        self.playback_thread = threading.Thread(target=playback)
        self.playback_thread.start()

    class TTSProcessor(threading.Thread):
        def __init__(self, text, url, token, appkey, voice, callback):
            super().__init__()
            self.text = text
            self.url = url
            self.token = token
            self.appkey = appkey
            self.voice = voice
            self.callback = callback
            self.output_file = "output.wav"

        def run(self):
            tts = nls.NlsSpeechSynthesizer(
                url=self.url,
                token=self.token,
                appkey=self.appkey,
                on_data=self._on_data,
                on_completed=self._on_completed,
                on_error=self._on_error
            )
            try:
                with open(self.output_file, 'wb') as f:
                    tts.start(self.text, voice=self.voice)
                    self.callback(self.output_file)
            except Exception as e:
                print(f"语音合成错误: {e}")

        def _on_data(self, data):
            try:
                with open(self.output_file, 'ab') as f:
                    f.write(data)
            except Exception as e:
                print(f"写入数据失败: {e}")

        def _on_completed(self, message):
            print(f"合成完成: {message}")

        def _on_error(self, message):
            print(f"合成错误: {message}")

    def _analyze_frame(self, frame):
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            base64_img = base64.b64encode(buffer).decode()
            
            response = self.vision_client.chat.completions.create(
                model="doubao-1-5-vision-pro-32k-250115",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_img}"},
                        {"type": "text", "text": "描述场景中的主要物体及其空间分布"}
                    ]
                }]
            )
            
            analysis = response.choices[0].message.content
            print("\n分析结果:", analysis)
            
            self.tts_thread = self.TTSProcessor(
                analysis, 
                self.URL, 
                self.TOKEN, 
                self.APPKEY, 
                self.voice,
                self._play_audio
            )
            self.tts_thread.start()
            
        except Exception as e:
            print(f"处理错误: {e}")

    def run(self):
        try:
            with dai.Device(self.pipeline) as device:
                print("相机已启动 (按0截图/enter退出)")
                
                while True:
                    frame = self._get_frame(device)
                    cv2.imshow("OAK-D View", frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('0'):
                        if frame is not None:
                            print("开始分析当前画面...")
                            if self.analysis_thread and self.analysis_thread.is_alive():
                                print("等待上次分析完成...")
                                self.analysis_thread.join()
                            
                            self.analysis_thread = threading.Thread(
                                target=self._analyze_frame,
                                args=(frame.copy(),)
                            )
                            self.analysis_thread.start()
                    elif key == 13:
                        break

        finally:
            cv2.destroyAllWindows()
            if self.analysis_thread:
                self.analysis_thread.join()
            if self.tts_thread:
                self.tts_thread.join()
            if self.playback_thread:
                self.playback_thread.join()
            self.audio.terminate()

if __name__ == "__main__":
    system = OakdVisionTTS()
    system.run()
