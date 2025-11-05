import depthai as dai
import cv2
import base64
import requests
import os
import time
import threading
import subprocess
from datetime import datetime

# 配置
DOUBAO_API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
BEARER_TOKEN = "3330a52a-b7b4-45ce-8486-49369bc4982e"
PROMPT_TEXT = "这张图片中的红绿灯是什么颜色？请只回答颜色，例如'红色'、'绿色'或'黄色'。如果看不清或没有红绿灯，请回答'未知'。"
IMAGE_FILENAME = "capture.jpg"

# 状态控制
is_recognizing = False
exit_requested = False

# 预览分辨率
PREVIEW_W = 300
PREVIEW_H = 300

# 历史截图保存
SAVE_HISTORY = True
HISTORY_DIR = "captures"

# 音频进程管理
current_audio_process = None
audio_lock = threading.Lock()

# ========== 工具函数 ==========
def log(msg):
    """带时间戳的日志"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def save_history(frame):
    """保存历史截图"""
    if SAVE_HISTORY:
        if not os.path.exists(HISTORY_DIR):
            os.makedirs(HISTORY_DIR)
        filename = os.path.join(HISTORY_DIR, f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(filename, frame)
        log(f"历史截图: {filename}")
        return filename
    return None

def encode_image_to_base64(image_path):
    log("编码图片为 Base64...")
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
    log(f"编码完成，长度: {len(encoded)} 字符")
    return encoded

def analyze_image_with_doubao(base64_image):
    log("调用豆包 API...")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {BEARER_TOKEN}"
    }
    payload = {
        "model": "doubao-seed-1-6-vision-250815",
        "messages": [{"role": "user","content": [{"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},{"type": "text","text": PROMPT_TEXT}]}]
    }
    try:
        start = time.time()
        resp = requests.post(DOUBAO_API_URL, headers=headers, json=payload, timeout=20)
        elapsed = time.time() - start
        
        resp.raise_for_status()
        res = resp.json()
        content = res['choices'][0]['message']['content'].strip()
        
        log(f"API 成功，耗时 {elapsed:.2f}秒")
        log(f"识别结果: {content}")
        return content
    except Exception as e:
        log(f"API 错误: {e}")
        return "API调用错误"

def check_traffic_light(result):
    """检查是否识别到有效的红绿灯"""
    result_lower = result.lower()
    valid = ['红色', 'red', '绿色', 'green', '黄色', 'yellow']
    for color in valid:
        if color in result_lower:
            return True
    return False

def format_speech(result):
    """格式化播报文本：红色->现在是红灯"""
    result_lower = result.lower()
    
    if '红色' in result_lower or 'red' in result_lower:
        return "现在是红灯"
    elif '绿色' in result_lower or 'green' in result_lower:
        return "现在是绿灯"
    elif '黄色' in result_lower or 'yellow' in result_lower:
        return "现在是黄灯"
    else:
        return result

def speak_result_espeak(text):
    """使用 espeak 本地 TTS（不需要网络）"""
    global current_audio_process
    
    log(f"播报: {text}")
    
    with audio_lock:
        try:
            # 停止之前的音频
            if current_audio_process is not None:
                try:
                    current_audio_process.terminate()
                    current_audio_process.wait(timeout=1)
                except:
                    pass
            
            # 使用 espeak 直接播报（本地，不需要网络）
            current_audio_process = subprocess.Popen(
                ['espeak', '-v', 'zh', '-s', '150', text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            log(f"播报已启动 (PID: {current_audio_process.pid})")
            
            # 等待播放完成
            current_audio_process.wait()
            log("播报完成")
            
        except FileNotFoundError:
            log("错误: 未找到 espeak，请安装: sudo apt install espeak")
        except Exception as e:
            log(f"TTS 错误: {e}")
            import traceback
            traceback.print_exc()

def speak_result_gtts_fallback(text):
    """使用 gTTS（需要网络，作为备用方案）"""
    global current_audio_process
    
    log(f"播报: {text}")
    
    with audio_lock:
        try:
            # 停止之前的音频
            if current_audio_process is not None:
                try:
                    current_audio_process.terminate()
                    current_audio_process.wait(timeout=1)
                except:
                    pass
            
            from gtts import gTTS
            
            # 生成 TTS
            tts = gTTS(text=text, lang='zh-cn')
            tts.save("result.mp3")
            log("TTS 生成完成")
            
            # 播放
            current_audio_process = subprocess.Popen(
                ['mpg123', '-q', 'result.mp3'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            log(f"播报已启动 (PID: {current_audio_process.pid})")
            
            # 等待播放完成
            current_audio_process.wait()
            log("播报完成")
            
        except Exception as e:
            log(f"gTTS 错误: {e}")
            # 网络失败时回退到 espeak
            log("回退到 espeak...")
            speak_result_espeak(text)

# 选择使用 espeak（推荐，不需要网络）
speak_result = speak_result_espeak

# 如果你想用 gTTS（需要网络），取消下面这行的注释
# speak_result = speak_result_gtts_fallback

def recognition_thread_target(frame):
    global is_recognizing
    
    log("=" * 60)
    log("开始识别")
    log("=" * 60)
    
    try:
        # 步骤 1: 保存截图
        log("步骤 1/4: 保存截图")
        cv2.imwrite(IMAGE_FILENAME, frame)
        log(f"已保存: {IMAGE_FILENAME}")
        
        # 步骤 2: Base64 编码
        log("步骤 2/4: Base64 编码")
        b64 = encode_image_to_base64(IMAGE_FILENAME)
        
        # 步骤 3: 调用 API
        log("步骤 3/4: 调用豆包 API")
        result = analyze_image_with_doubao(b64)
        
        # 步骤 4: 播报结果
        log("步骤 4/4: 播报结果")
        
        if check_traffic_light(result):
            speech = format_speech(result)
            speak_result(speech)
            log("识别成功")
        else:
            log("未检测到红绿灯")
            speak_result("未检测到红绿灯，请重新对准红绿灯后再按空格键")
        
    except Exception as e:
        log(f"识别异常: {e}")
        import traceback
        traceback.print_exc()
        try:
            speak_result("识别流程出错")
        except:
            pass
    finally:
        is_recognizing = False
        log("=" * 60)
        log("识别结束")
        log("=" * 60)
        log("")

def build_pipeline():
    log("构建 Pipeline...")
    p = dai.Pipeline()
    cam = p.create(dai.node.ColorCamera)
    cam.setPreviewSize(PREVIEW_W, PREVIEW_H)
    cam.setInterleaved(False)
    xout = p.create(dai.node.XLinkOut)
    xout.setStreamName("rgb")
    cam.preview.link(xout.input)
    log("Pipeline 构建完成")
    return p

def main():
    global is_recognizing, exit_requested, current_audio_process

    print("\n" + "=" * 70)
    print("红绿灯识别系统")
    print("=" * 70)
    print(f"历史截图: {'开启' if SAVE_HISTORY else '关闭'} -> {HISTORY_DIR}")
    print(f"TTS 引擎: espeak (本地)")
    print("=" * 70)
    print("\n操作说明:")
    print("   [空格键] - 截图并识别")
    print("   [Q 键]   - 退出程序")
    print("=" * 70 + "\n")

    win_name = "Preview - 按 空格 识别, q 退出"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, PREVIEW_W, PREVIEW_H)

    reconnect_delay = 2.0
    max_retries = 10
    retries = 0

    while not exit_requested:
        pipeline = build_pipeline()
        try:
            log("连接设备...")
            with dai.Device(pipeline) as device:
                log(f"设备已连接: {device.getDeviceName()}")
                log(f"设备 ID: {device.getMxId()}")
                log("")
                
                q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
                frame = None
                retries = 0

                while True:
                    in_rgb = q_rgb.tryGet()
                    if in_rgb is not None:
                        frame = in_rgb.getCvFrame()

                    if frame is None:
                        key = cv2.waitKey(50) & 0xFF
                        if key == ord('q'):
                            exit_requested = True
                            break
                        continue

                    display = frame.copy()
                    
                    # 显示状态
                    if is_recognizing:
                        cv2.putText(display, "RECOGNIZING...", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        cv2.putText(display, "Ready [SPACE]", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # 显示时间
                    time_str = datetime.now().strftime("%H:%M:%S")
                    cv2.putText(display, time_str, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    cv2.imshow(win_name, display)

                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        log("用户退出")
                        exit_requested = True
                        break
                        
                    elif key == ord(' '):
                        if not is_recognizing:
                            log("空格键 - 开始识别")
                            is_recognizing = True
                            
                            # 保存历史
                            save_history(frame.copy())
                            
                            # 启动识别线程
                            t = threading.Thread(target=recognition_thread_target, args=(frame.copy(),), daemon=True)
                            t.start()
                        else:
                            log("识别进行中，请等待...")

                if exit_requested:
                    break

        except RuntimeError as e:
            log(f"设备通信异常: {e}")
            retries += 1
            if retries > max_retries:
                log(f"超过最大重试次数 ({max_retries})，退出")
                break
            log(f"等待 {reconnect_delay}秒 后重连 ({retries}/{max_retries})...")
            time.sleep(reconnect_delay)
            continue
            
        except Exception as e:
            log(f"未处理异常: {e}")
            import traceback
            traceback.print_exc()
            break

    # 退出时清理音频进程
    if current_audio_process is not None:
        try:
            current_audio_process.terminate()
        except:
            pass
    
    cv2.destroyAllWindows()
    print("\n" + "=" * 70)
    print("程序已退出")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()