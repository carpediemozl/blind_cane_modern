import depthai as dai
import cv2
import base64
import requests
import os
from gtts import gTTS
import time
import threading

# 配置
DOUBAO_API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
BEARER_TOKEN = "3330a52a-b7b4-45ce-8486-49369bc4982e"
PROMPT_TEXT = "这张图片中的红绿灯是什么颜色？请只回答颜色，例如‘红色’、‘绿色’或‘黄色’。如果看不清或没有红绿灯，请回答‘未知’。"
IMAGE_FILENAME = "capture.jpg"

# 状态控制
is_recognizing = False
exit_requested = False

# 安全的 preview 分辨率（DepthAI 常用且稳定）
PREVIEW_W = 300
PREVIEW_H = 300

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def analyze_image_with_doubao(base64_image):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {BEARER_TOKEN}"
    }
    payload = {
        "model": "doubao-seed-1-6-vision-250815",
        "messages": [{"role": "user","content": [{"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},{"type": "text","text": PROMPT_TEXT}]}]
    }
    try:
        resp = requests.post(DOUBAO_API_URL, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        res = resp.json()
        content = res['choices'][0]['message']['content']
        return content
    except Exception as e:
        print(f"API 错误: {e}")
        return "API调用错误"

def speak_result(text, lang='zh-cn'):
    try:
        tts = gTTS(text=text, lang=lang)
        tts.save("result.mp3")
        # 使用 mpg123 播放（确保已安装 mpg123），如果没有可替换为 aplay/espeak 方案
        os.system("mpg123 -q result.mp3")
    except Exception as e:
        print(f"TTS 错误: {e}")

def recognition_thread_target(frame):
    global is_recognizing
    try:
        cv2.imwrite(IMAGE_FILENAME, frame)
        b64 = encode_image_to_base64(IMAGE_FILENAME)
        result = analyze_image_with_doubao(b64)
        speak_result(f"当前识别到的红绿灯颜色为: {result}")
    except Exception as e:
        print(f"识别线程异常: {e}")
        try:
            speak_result("识别流程出错")
        except:
            pass
    finally:
        is_recognizing = False

def build_pipeline():
    p = dai.Pipeline()
    cam = p.create(dai.node.ColorCamera)
    cam.setPreviewSize(PREVIEW_W, PREVIEW_H)
    cam.setInterleaved(False)
    # 不强制设置 sensor 分辨率，避免与 preview 冲突
    # cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)  # 有时会和 preview 冲突，除非确知兼容
    xout = p.create(dai.node.XLinkOut)
    xout.setStreamName("rgb")
    cam.preview.link(xout.input)
    return p

def main():
    global is_recognizing, exit_requested

    win_name = "Preview - 按 空格 识别, q 退出"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, PREVIEW_W, PREVIEW_H)

    reconnect_delay = 2.0
    max_retries = 10
    retries = 0

    while not exit_requested:
        pipeline = build_pipeline()
        try:
            print("尝试连接设备...")
            with dai.Device(pipeline) as device:
                print("设备连接成功")
                q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
                frame = None
                retries = 0  # 成功连接后重置重试计数

                while True:
                    # 优先尝试获取最新帧（非阻塞）
                    in_rgb = q_rgb.tryGet()
                    if in_rgb is not None:
                        frame = in_rgb.getCvFrame()

                    # 如果没有帧，短暂等待以处理 GUI 事件
                    if frame is None:
                        key = cv2.waitKey(50) & 0xFF
                        if key == ord('q'):
                            exit_requested = True
                            break
                        continue

                    display = frame.copy()
                    if is_recognizing:
                        cv2.putText(display, "RECOGNIZING...", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    cv2.imshow(win_name, display)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        exit_requested = True
                        break
                    elif key == ord(' '):
                        if not is_recognizing:
                            is_recognizing = True
                            t = threading.Thread(target=recognition_thread_target, args=(frame.copy(),), daemon=True)
                            t.start()
                        else:
                            print("识别正在进行中，请稍候...")

                # 退出内层循环后，context manager 会退出并 cleanup 设备
                if exit_requested:
                    break

        except RuntimeError as e:
            # 通常为 X_LINK_ERROR 或 设备通信异常
            print(f"设备通信异常: {e}")
            retries += 1
            if retries > max_retries:
                print("超过最大重试次数，退出。")
                break
            print(f"等待 {reconnect_delay} 秒后重连（{retries}/{max_retries}）...")
            time.sleep(reconnect_delay)
            continue
        except Exception as e:
            print(f"未处理的异常: {e}")
            import traceback
            traceback.print_exc()
            break

    cv2.destroyAllWindows()
    print("程序退出。")

if __name__ == "__main__":
    main()