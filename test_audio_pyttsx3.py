import subprocess

def speak(text, language='zh'):
    """
    使用 subprocess 直接调用 espeak-ng 命令进行语音播报。
    这是最稳定、最透明、保证可行的方案。
    """
    try:
        print(f"准备使用 espeak-ng (语言: {language}) 播报...")
        command = ['espeak-ng', '-v', language, text]
        
        # 执行命令，等待其完成
        subprocess.run(command, check=True, capture_output=True, text=True)
        
        print("播报成功。")
    except FileNotFoundError:
        print("错误: 'espeak-ng' 命令未找到。请运行 'sudo apt-get install espeak-ng'。")
    except subprocess.CalledProcessError as e:
        # 如果 espeak-ng 出错，打印它的错误信息
        print(f"espeak-ng 执行失败。错误信息:\n{e.stderr}")

# --- 要播报的中文内容 ---
text_to_say = "恭喜你，这才是真正的最终解决方案。我们放弃了有问题的库，直接调用了可靠的系统命令。"

# --- 执行播бо ---
speak(text_to_say)