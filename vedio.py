import cv2

# 视频文件路径
video_path = 'OpenCV-AI-Competition-UTAR4Vision-main/output_video.avi'

# 使用cv2.VideoCapture打开视频文件
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 逐帧读取视频
while True:
    # 读取一帧
    ret, frame = cap.read()

    # 如果正确读取帧，ret为True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # 显示帧
    cv2.imshow('frame', frame)

    # 按 'q' 键退出循环
    if cv2.waitKey(1) == ord('q'):
        break

# 完成所有操作后，释放捕获器
cap.release()
# 关闭所有OpenCV窗口
cv2.destroyAllWindows()