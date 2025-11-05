import cv2
import depthai as dai
import numpy as np
import time
from pathlib import Path
from collections import deque, Counter
import os

# --- 1. 核心参数 ---
MODEL_PATH = 'models/knottedblob/best10301nohue_openvino_2022.1_6shave.blob'
LABEL_MAP = ['Green', 'Red', 'Yellow']
NN_INPUT_SIZE = 416

# 测试图片路径（支持多种格式）
TEST_IMAGES_PATH = 'lisatraffic.7z'  # 图片文件夹
# 如果是 7z 文件，会自动解压

# === 过滤参数 ===
CLASS_THRESHOLDS = {
    'Green': 0.55,
    'Red': 0.50,
    'Yellow': 0.45
}

# 时序平滑（图片测试时禁用）
USE_TEMPORAL_SMOOTHING = False

# 测试选项
SAVE_RESULTS = True           # 保存检测结果图片
SHOW_GROUND_TRUTH = True      # 显示真实标签（如果有）
AUTO_PLAY = False             # 自动播放（不等待按键）
AUTO_PLAY_INTERVAL = 1.0      # 自动播放间隔（秒）

# --- 2. 解压 7z 文件（如果需要）---
def extract_7z(archive_path, output_dir):
    """解压 7z 文件"""
    print(f"检测到 7z 文件: {archive_path}")
    
    try:
        import py7zr
        print("正在解压...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        with py7zr.SevenZipFile(archive_path, mode='r') as archive:
            archive.extractall(path=output_dir)
        
        print(f"✓ 解压完成: {output_dir}")
        return output_dir
        
    except ImportError:
        print("❌ 需要安装 py7zr: pip install py7zr")
        return None
    except Exception as e:
        print(f"❌ 解压失败: {e}")
        return None


def find_test_images(path):
    """查找测试图片"""
    path = Path(path)
    
    # 如果是 7z 文件，先解压
    if path.suffix == '.7z':
        extract_dir = path.parent / (path.stem + '_extracted')
        if not extract_dir.exists():
            path = extract_7z(path, extract_dir)
            if path is None:
                return []
        else:
            path = extract_dir
            print(f"使用已解压的目录: {path}")
    
    # 查找所有图片
    if path.is_dir():
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        images = []
        for ext in image_extensions:
            images.extend(path.glob(f'*{ext}'))
            images.extend(path.glob(f'*{ext.upper()}'))
        
        # 递归查找子目录
        for subdir in path.iterdir():
            if subdir.is_dir():
                for ext in image_extensions:
                    images.extend(subdir.glob(f'*{ext}'))
                    images.extend(subdir.glob(f'*{ext.upper()}'))
        
        return sorted(images)
    elif path.is_file():
        return [path]
    else:
        return []


def load_ground_truth(image_path, label_dir='labels'):
    """加载真实标签（如果有）"""
    label_path = image_path.parent / label_dir / (image_path.stem + '.txt')
    
    if not label_path.exists():
        # 尝试在同一目录
        label_path = image_path.parent / (image_path.stem + '.txt')
    
    if not label_path.exists():
        return None
    
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        annotations = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
                annotations.append({
                    'class_id': class_id,
                    'bbox': [x_center, y_center, width, height]
                })
        
        return annotations
    except Exception as e:
        return None


# --- 3. 构建 Pipeline（无深度） ---
print("=" * 70)
print("OAK-D 图片批量测试工具")
print("=" * 70)

# 查找测试图片
test_images = find_test_images(TEST_IMAGES_PATH)

if not test_images:
    print(f"❌ 未找到测试图片！")
    print(f"   路径: {TEST_IMAGES_PATH}")
    print(f"   支持格式: .jpg, .jpeg, .png, .bmp, .7z")
    exit(1)

print(f"\n找到 {len(test_images)} 张测试图片")
print(f"模型: {MODEL_PATH}")
print(f"类别阈值: {CLASS_THRESHOLDS}")
print("=" * 70)

# 创建结果保存目录
if SAVE_RESULTS:
    results_dir = Path('test_results')
    results_dir.mkdir(exist_ok=True)
    print(f"\n结果将保存到: {results_dir}")

# 构建简化的 Pipeline（仅用于检测，无深度）
pipeline = dai.Pipeline()

# RGB 相机
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(NN_INPUT_SIZE, NN_INPUT_SIZE)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# 检测网络
detection_nn = pipeline.create(dai.node.YoloDetectionNetwork)
detection_nn.setBlobPath(MODEL_PATH)
detection_nn.setConfidenceThreshold(0.3)
detection_nn.setNumClasses(len(LABEL_MAP))
detection_nn.setCoordinateSize(4)
detection_nn.setAnchors([10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326])
detection_nn.setAnchorMasks({"side52": [0, 1, 2], "side26": [3, 4, 5], "side13": [6, 7, 8]})
detection_nn.setIouThreshold(0.5)
detection_nn.input.setBlocking(False)

# 输出
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("detections")

# 链接
cam_rgb.preview.link(detection_nn.input)
cam_rgb.preview.link(xout_rgb.input)
detection_nn.out.link(xout_nn.input)

print("\n按键说明:")
print("  SPACE - 下一张图片")
print("  b     - 上一张图片")
print("  s     - 保存当前结果")
print("  a     - 切换自动播放")
print("  1-6   - 调整阈值")
print("  q     - 退出")
print("=" * 70)

# --- 4. 测试循环 ---
try:
    with dai.Device(pipeline) as device:
        print("\n✓ 设备连接成功！")
        print(f"  DepthAI 版本: {dai.__version__}\n")
        
        q_rgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        q_nn = device.getOutputQueue(name="detections", maxSize=1, blocking=False)
        
        # 统计信息
        stats = {
            'total': len(test_images),
            'correct': 0,
            'wrong': 0,
            'no_detection': 0,
            'per_class': {label: {'tp': 0, 'fp': 0, 'fn': 0} for label in LABEL_MAP}
        }
        
        current_idx = 0
        last_auto_time = time.time()
        
        while current_idx < len(test_images):
            image_path = test_images[current_idx]
            
            # 读取图片
            original_image = cv2.imread(str(image_path))
            
            if original_image is None:
                print(f"❌ 无法读取: {image_path}")
                current_idx += 1
                continue
            
            # 调整大小到模型输入
            input_image = cv2.resize(original_image, (NN_INPUT_SIZE, NN_INPUT_SIZE))
            
            # 发送图片（模拟相机输入 - 这里需要实际实现）
            # 注意：DepthAI 通常从相机读取，对于静态图片测试，
            # 我们需要使用 ImageManip 节点或直接处理
            
            # === 临时方案：使用预览显示 ===
            print(f"\n[{current_idx + 1}/{len(test_images)}] {image_path.name}")
            
            # 加载真实标签
            ground_truth = load_ground_truth(image_path) if SHOW_GROUND_TRUTH else None
            
            # 等待检测结果（需要实际从队列获取）
            # 这里简化为显示图片
            display_image = original_image.copy()
            
            # 绘制真实标签（如果有）
            if ground_truth:
                h, w = display_image.shape[:2]
                for ann in ground_truth:
                    class_id = ann['class_id']
                    label = LABEL_MAP[class_id] if class_id < len(LABEL_MAP) else 'unknown'
                    
                    x_c, y_c, width, height = ann['bbox']
                    x1 = int((x_c - width/2) * w)
                    y1 = int((y_c - height/2) * h)
                    x2 = int((x_c + width/2) * w)
                    y2 = int((y_c + height/2) * h)
                    
                    # 绿色虚线框表示真实标签
                    cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_4)
                    cv2.putText(display_image, f"GT: {label}", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                print(f"  真实标签: {[LABEL_MAP[a['class_id']] for a in ground_truth]}")
            
            # 显示信息
            info_text = f"{current_idx + 1}/{len(test_images)} - {image_path.name}"
            cv2.putText(display_image, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示阈值
            threshold_text = f"G:{CLASS_THRESHOLDS['Green']:.2f} R:{CLASS_THRESHOLDS['Red']:.2f} Y:{CLASS_THRESHOLDS['Yellow']:.2f}"
            cv2.putText(display_image, threshold_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            cv2.imshow("Image Test", display_image)
            
            # 保存结果
            if SAVE_RESULTS:
                result_path = results_dir / f"result_{image_path.name}"
                cv2.imwrite(str(result_path), display_image)
            
            # 键盘控制
            if AUTO_PLAY:
                if time.time() - last_auto_time >= AUTO_PLAY_INTERVAL:
                    current_idx += 1
                    last_auto_time = time.time()
                key = cv2.waitKey(50)
            else:
                key = cv2.waitKey(0)
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space
                current_idx += 1
            elif key == ord('b'):  # Back
                current_idx = max(0, current_idx - 1)
            elif key == ord('a'):  # Auto play
                AUTO_PLAY = not AUTO_PLAY
                print(f"  自动播放: {'开' if AUTO_PLAY else '关'}")
                last_auto_time = time.time()
            elif key == ord('s'):  # Save
                result_path = results_dir / f"saved_{image_path.name}"
                cv2.imwrite(str(result_path), display_image)
                print(f"  ✓ 已保存: {result_path}")
            elif key == ord('1'):
                CLASS_THRESHOLDS['Green'] = min(0.95, CLASS_THRESHOLDS['Green'] + 0.05)
                print(f"  绿灯阈值: {CLASS_THRESHOLDS['Green']:.2f}")
            elif key == ord('2'):
                CLASS_THRESHOLDS['Green'] = max(0.1, CLASS_THRESHOLDS['Green'] - 0.05)
                print(f"  绿灯阈值: {CLASS_THRESHOLDS['Green']:.2f}")
            elif key == ord('3'):
                CLASS_THRESHOLDS['Red'] = min(0.95, CLASS_THRESHOLDS['Red'] + 0.05)
                print(f"  红灯阈值: {CLASS_THRESHOLDS['Red']:.2f}")
            elif key == ord('4'):
                CLASS_THRESHOLDS['Red'] = max(0.1, CLASS_THRESHOLDS['Red'] - 0.05)
                print(f"  红灯阈值: {CLASS_THRESHOLDS['Red']:.2f}")
            elif key == ord('5'):
                CLASS_THRESHOLDS['Yellow'] = min(0.95, CLASS_THRESHOLDS['Yellow'] + 0.05)
                print(f"  黄灯阈值: {CLASS_THRESHOLDS['Yellow']:.2f}")
            elif key == ord('6'):
                CLASS_THRESHOLDS['Yellow'] = max(0.1, CLASS_THRESHOLDS['Yellow'] - 0.05)
                print(f"  黄灯阈值: {CLASS_THRESHOLDS['Yellow']:.2f}")
        
        # 显示统计
        print("\n" + "=" * 70)
        print("测试完成")
        print("=" * 70)
        print(f"测试图片数: {stats['total']}")
        
        if SAVE_RESULTS:
            print(f"\n结果已保存到: {results_dir}")
        
        print("=" * 70)

except Exception as e:
    print(f"\n❌ 程序异常: {e}")
    import traceback
    traceback.print_exc()

finally:
    cv2.destroyAllWindows()
    print("\n✓ 测试结束。")