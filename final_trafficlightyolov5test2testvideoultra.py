"""
OAK-D 视频检测工具 - 使用视频文件进行检测
使用视频文件的画面通过 OAK-D 进行推理检测
"""

import cv2
import depthai as dai
import numpy as np
import time
from pathlib import Path
from collections import deque, Counter
import sys

# --- 配置 ---
MODEL_PATH = 'models/knottedblob/best10301nohue_openvino_2022.1_6shave.blob'
LABEL_MAP = ['Green', 'Red', 'Yellow']
NN_INPUT_SIZE = 416

CLASS_THRESHOLDS = {
    'Green': 0.55,
    'Red': 0.50,
    'Yellow': 0.45
}

# 配置类
class Config:
    def __init__(self):
        self.use_temporal_smoothing = True
        self.buffer_size = 7
        self.min_votes = 4
        self.show_stats = True
        self.show_confidence = True
        self.save_output = False
        self.output_path = 'output.mp4'

config = Config()


# --- 时序平滑类 ---
class TemporalSmoother:
    def __init__(self, buffer_size=7, min_votes=4):
        self.buffer_size = buffer_size
        self.min_votes = min_votes
        self.label_buffer = deque(maxlen=buffer_size)
        self.conf_buffer = deque(maxlen=buffer_size)
        self.bbox_buffer = deque(maxlen=buffer_size)
        
    def update(self, detection):
        if detection is not None:
            self.label_buffer.append(detection['label'])
            self.conf_buffer.append(detection['confidence'])
            self.bbox_buffer.append(detection['bbox'])
        else:
            self.label_buffer.append(None)
            self.conf_buffer.append(0)
            self.bbox_buffer.append(None)
    
    def get_smoothed_detection(self):
        if len(self.label_buffer) < self.min_votes:
            return None
        
        valid_labels = [l for l in self.label_buffer if l is not None]
        if len(valid_labels) < self.min_votes:
            return None
        
        label_counts = Counter(valid_labels)
        most_common_label, vote_count = label_counts.most_common(1)[0]
        
        if vote_count < self.min_votes:
            return None
        
        label_confs = [self.conf_buffer[i] for i in range(len(self.label_buffer)) 
                       if self.label_buffer[i] == most_common_label]
        avg_confidence = np.mean(label_confs) if label_confs else 0
        
        label_bboxes = [self.bbox_buffer[i] for i in range(len(self.label_buffer))
                        if self.label_buffer[i] == most_common_label and self.bbox_buffer[i] is not None]
        
        if label_bboxes:
            avg_bbox = np.mean(label_bboxes, axis=0)
        else:
            avg_bbox = None
        
        return {
            'label': most_common_label,
            'confidence': avg_confidence,
            'bbox': avg_bbox,
            'votes': vote_count,
            'total_frames': len(self.label_buffer)
        }
    
    def reset(self):
        self.label_buffer.clear()
        self.conf_buffer.clear()
        self.bbox_buffer.clear()


# --- 统计类 ---
class DetectionStats:
    def __init__(self):
        self.history = {label: deque(maxlen=300) for label in LABEL_MAP}
        self.total_counts = {label: 0 for label in LABEL_MAP}
        self.confidence_history = deque(maxlen=100)
        
    def update(self, label, confidence):
        for l in LABEL_MAP:
            self.history[l].append(1 if l == label else 0)
        self.total_counts[label] += 1
        self.confidence_history.append(confidence)
    
    def get_recent_distribution(self, frames=30):
        counts = {}
        for label in LABEL_MAP:
            if len(self.history[label]) >= frames:
                counts[label] = sum(list(self.history[label])[-frames:])
            else:
                counts[label] = sum(self.history[label])
        return counts
    
    def draw_stats(self, frame, x=10, y=100):
        if not config.show_stats:
            return frame
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x-5, y-25), (x+300, y+150), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        cv2.putText(frame, "Stats (Last 30 frames)", 
                   (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        recent = self.get_recent_distribution(30)
        y_offset = y + 25
        
        for label in LABEL_MAP:
            count = recent[label]
            total = self.total_counts[label]
            
            if label == 'Green':
                color = (0, 255, 0)
            elif label == 'Red':
                color = (0, 0, 255)
            else:
                color = (0, 255, 255)
            
            text = f"{label}: {count}/30 (Total: {total})"
            cv2.putText(frame, text, (x, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            bar_width = int(count / 30 * 200)
            cv2.rectangle(frame, (x, y_offset+5), (x+200, y_offset+15), (50, 50, 50), -1)
            cv2.rectangle(frame, (x, y_offset+5), (x+bar_width, y_offset+15), color, -1)
            
            y_offset += 30
        
        return frame


def create_pipeline():
    """创建用于视频推理的 Pipeline"""
    pipeline = dai.Pipeline()
    
    # 创建 YOLO 检测网络节点
    detection_nn = pipeline.create(dai.node.YoloDetectionNetwork)
    detection_nn.setBlobPath(MODEL_PATH)
    detection_nn.setConfidenceThreshold(0.3)
    detection_nn.setNumClasses(len(LABEL_MAP))
    detection_nn.setCoordinateSize(4)
    detection_nn.setAnchors([10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326])
    detection_nn.setAnchorMasks({"side52": [0, 1, 2], "side26": [3, 4, 5], "side13": [6, 7, 8]})
    detection_nn.setIouThreshold(0.5)
    detection_nn.input.setBlocking(False)
    
    # 输入节点（用于接收视频帧）
    xin = pipeline.create(dai.node.XLinkIn)
    xin.setStreamName("in")
    
    # 输出节点
    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("detections")
    
    # 连接
    xin.out.link(detection_nn.input)
    detection_nn.out.link(xout_nn.input)
    
    return pipeline


def preprocess_frame(frame, size=NN_INPUT_SIZE):
    """预处理视频帧用于推理"""
    # 调整大小
    resized = cv2.resize(frame, (size, size))
    return resized


def create_dai_frame(frame):
    """将 OpenCV 帧转换为 DepthAI ImgFrame"""
    img = dai.ImgFrame()
    img.setType(dai.ImgFrame.Type.BGR888p)
    img.setWidth(frame.shape[1])
    img.setHeight(frame.shape[0])
    # 转换为平面格式 (CHW)
    img.setData(frame.transpose(2, 0, 1).flatten())
    return img


# --- 主函数 ---
def main(video_path):
    print("=" * 70)
    print("OAK-D 视频检测工具")
    print("=" * 70)
    print(f"视频源: {video_path}")
    print(f"模型: {MODEL_PATH}")
    print(f"阈值: {CLASS_THRESHOLDS}")
    print("=" * 70)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频信息:")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {fps:.2f} FPS")
    print(f"  总帧数: {total_frames}")
    
    print("\n按键说明:")
    print("  SPACE - 暂停/继续")
    print("  s     - 保存当前帧")
    print("  r     - 重置统计")
    print("  t     - 切换时序平滑")
    print("  d     - 切换统计显示")
    print("  q     - 退出")
    print("=" * 70)
    
    # 创建 Pipeline
    pipeline = create_pipeline()
    
    # 启动设备
    try:
        with dai.Device(pipeline) as device:
            print("\n✓ OAK-D 设备连接成功\n")
            
            # 获取队列
            q_in = device.getInputQueue("in")
            q_nn = device.getOutputQueue("detections", 4, False)
            
            # 初始化
            smoother = TemporalSmoother(config.buffer_size, config.min_votes)
            stats = DetectionStats()
            
            frame_count = 0
            paused = False
            start_time = time.time()
            
            cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
            
            while True:
                if not paused:
                    # 读取视频帧
                    ret, video_frame = cap.read()
                    if not ret:
                        print("\n✓ 视频播放完毕")
                        break
                    
                    frame_count += 1
                    
                    # 预处理帧
                    processed_frame = preprocess_frame(video_frame, NN_INPUT_SIZE)
                    
                    # 转换为 DepthAI 格式并发送
                    dai_frame = create_dai_frame(processed_frame)
                    q_in.send(dai_frame)
                    
                    # 获取检测结果
                    in_nn = q_nn.tryGet()
                    
                    if in_nn is not None:
                        detections = in_nn.detections
                        
                        # 过滤检测结果
                        filtered = []
                        for det in detections:
                            try:
                                if det.label < len(LABEL_MAP):
                                    label = LABEL_MAP[det.label]
                                    if det.confidence >= CLASS_THRESHOLDS[label]:
                                        filtered.append({
                                            'label': label,
                                            'confidence': det.confidence,
                                            'bbox': [det.xmin, det.ymin, det.xmax, det.ymax]
                                        })
                            except Exception as e:
                                print(f"检测处理错误: {e}")
                                continue
                        
                        # 选择最佳检测
                        best_det = max(filtered, key=lambda x: x['confidence']) if filtered else None
                        
                        # 时序平滑
                        if config.use_temporal_smoothing:
                            smoother.update(best_det)
                            smoothed = smoother.get_smoothed_detection()
                        else:
                            smoothed = best_det
                    else:
                        smoothed = None
                    
                    # 显示检测结果
                    display = video_frame.copy()
                    
                    # 绘制检测框
                    if smoothed and smoothed['bbox'] is not None:
                        label = smoothed['label']
                        bbox = smoothed['bbox']
                        confidence = smoothed['confidence']
                        
                        # 转换坐标到原始帧
                        x1 = int(bbox[0] * width)
                        y1 = int(bbox[1] * height)
                        x2 = int(bbox[2] * width)
                        y2 = int(bbox[3] * height)
                        
                        # 颜色
                        if label == 'Green':
                            color = (0, 255, 0)
                        elif label == 'Red':
                            color = (0, 0, 255)
                        else:
                            color = (0, 255, 255)
                        
                        # 绘制
                        cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)
                        
                        # 标签文本
                        text = f"{label} {confidence*100:.1f}%"
                        if 'votes' in smoothed:
                            vote = f"({smoothed['votes']}/{smoothed['total_frames']})"
                        else:
                            vote = ""
                        
                        cv2.putText(display, text, (x1, y1-30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        if vote:
                            cv2.putText(display, vote, (x1, y1-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        
                        # 更新统计
                        stats.update(label, confidence)
                    
                    # 绘制统计信息
                    display = stats.draw_stats(display)
                    
                    # 显示进度和 FPS
                    progress = frame_count / total_frames if total_frames > 0 else 0
                    process_fps = frame_count / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
                    
                    info = f"Frame: {frame_count}/{total_frames} ({progress*100:.1f}%) | FPS: {process_fps:.1f}"
                    cv2.putText(display, info, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.imshow("Detection", display)
                
                # 键盘控制
                key = cv2.waitKey(1)
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = not paused
                    print(f"{'⏸ 暂停' if paused else '▶ 继续'}")
                elif key == ord('s'):
                    filename = f"frame_{frame_count:06d}.jpg"
                    cv2.imwrite(filename, display)
                    print(f"✓ 已保存: {filename}")
                elif key == ord('r'):
                    stats = DetectionStats()
                    smoother.reset()
                    print("✓ 统计已重置")
                elif key == ord('t'):
                    config.use_temporal_smoothing = not config.use_temporal_smoothing
                    print(f"时序平滑: {'开' if config.use_temporal_smoothing else '关'}")
                elif key == ord('d'):
                    config.show_stats = not config.show_stats
                    print(f"统计显示: {'开' if config.show_stats else '关'}")
            
            cap.release()
            cv2.destroyAllWindows()
            
            # 最终统计
            print("\n" + "=" * 70)
            print("检测完成")
            print("=" * 70)
            print(f"处理帧数: {frame_count}")
            print(f"处理时间: {time.time() - start_time:.2f} 秒")
            print(f"平均 FPS: {frame_count / (time.time() - start_time):.2f}")
            print(f"\n检测统计:")
            for label, count in stats.total_counts.items():
                print(f"  {label}: {count} 次")
            print("=" * 70)
    
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cv2.destroyAllWindows()
        print("\n✓ 程序结束")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        print("用法: python oak_video_detection_fixed.py video.mp4")
        sys.exit(1)
    
    main(video_path)