from gtts import gTTS
import os

# 你想要播报的文字
text_to_say = "你好，恭喜你，蓝牙耳机连接成功！现在你听到的声音，来自你的树莓派。"

# 设置语言为中文
language = 'zh-cn'

# 创建 gTTS 对象
speech = gTTS(text=text_to_say, lang=language, slow=False)

# 将语音保存为 mp3 文件
speech.save("test.mp3")

# 使用 mpg123 播放这个 mp3 文件
print("正在播放语音...")
os.system("mpg123 test.mp3")
print("播放完毕。")