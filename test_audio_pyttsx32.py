# 保存为 tts_file.py
import subprocess
import os

def speak_via_file(text, lang='en'):
    """生成 WAV 文件并通过 aplay 播放"""
    wav_file = '/tmp/speech.wav'
    
    # 生成 WAV
    if lang == 'zh':
        subprocess.run(['espeak', '-v', 'zh', '-w', wav_file, text])
    else:
        subprocess.run(['espeak', '-w', wav_file, text])
    
    # 播放（会自动使用默认音频设备）
    subprocess.run(['aplay', wav_file])
    
    # 清理
    os.remove(wav_file)

# 测试
speak_via_file("This is a test message")
speak_via_file("这是中文测试", lang='zh')