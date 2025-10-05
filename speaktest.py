import pyttsx3

engine = pyttsx3.init()
x = ['你好吗？']
engine.say(x)
engine.runAndWait()
a = 500
y = '向天再借%d年' % a
engine.say(y)
engine.runAndWait()
