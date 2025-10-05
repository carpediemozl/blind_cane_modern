
import pyaudio
import wave
import sys

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

if __name__ == "__main__":
    
    play_pcm_file("/home/pi/Desktop/wtcane/tests/yuyin.pcm",16600,1,2)
