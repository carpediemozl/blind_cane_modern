import sounddevice as sd
import soundfile as sf

duration = 5 #录音时长 秒
fs =44100 #采样率 Hz
channels = 2 #声道数

def record_audio(duration, fs, channels):
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=channels)
    sd.wait()
    return myrecording
    
    
audio_data = record_audio(duration, fs, channels)

filename = 'output.wav'
sf.write(filename, audio_data, fs)
print(f"已保存到{filename}")

