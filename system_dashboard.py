import time
import os
import psutil

# --- 配置 ---
TEMP_FILE_PATH = "/sys/class/thermal/thermal_zone0/temp"
BAR_LENGTH = 25  # 可视化条的长度

# --- ANSI 颜色代码 ---
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_RED = "\033[91m"
COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"

def get_cpu_temperature():
    """读取CPU温度 (摄氏度)"""
    try:
        with open(TEMP_FILE_PATH, 'r') as f:
            return float(f.read().strip()) / 1000.0
    except (IOError, ValueError):
        return None

def get_cpu_usage():
    """获取CPU总体使用率 (%)"""
    return psutil.cpu_percent(interval=None)

def get_memory_usage():
    """获取内存使用信息 (已用GB, 总共GB, 使用率%)"""
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024**3)
    used_gb = mem.used / (1024**3)
    return used_gb, total_gb, mem.percent

def get_swap_usage():
    """获取Swap使用信息 (已用MB, 总共MB, 使用率%)"""
    swap = psutil.swap_memory()
    total_mb = swap.total / (1024**2)
    used_mb = swap.used / (1024**2)
    return used_mb, total_mb, swap.percent

def create_bar(percent, color_thresholds):
    """根据百分比和阈值创建彩色的可视化条"""
    bar_fill = int(percent / 100.0 * BAR_LENGTH)
    bar = '█' * bar_fill + '-' * (BAR_LENGTH - bar_fill)
    
    color = COLOR_GREEN
    if percent > color_thresholds[1]:
        color = COLOR_RED
    elif percent > color_thresholds[0]:
        color = COLOR_YELLOW
        
    return f"{color}[{bar}]{COLOR_RESET}"

def main():
    """主函数，循环监控并显示系统状态"""
    print("--- 树莓派实时系统仪表盘 ---")
    print("按 Ctrl+C 退出。")
    
    try:
        while True:
            # 1. 获取所有数据
            cpu_temp = get_cpu_temperature()
            cpu_usage = get_cpu_usage()
            mem_used, mem_total, mem_percent = get_memory_usage()
            swap_used, swap_total, swap_percent = get_swap_usage()

            # 2. 清理屏幕 (ANSI转义码)
            # \033[H: 将光标移动到左上角
            # \033[J: 清除从光标到屏幕末尾的内容
            print("\033[H\033[J", end="")
            
            print("--- 树莓派实时系统仪表盘 ---")
            
            # 3. 格式化并打印输出
            # CPU 温度
            if cpu_temp is not None:
                temp_bar = create_bar(cpu_temp, (60, 80)) # 阈值: 60°C, 80°C
                print(f"CPU 温度 : {cpu_temp:5.2f}°C {temp_bar}")
            else:
                print("CPU 温度 : 无法读取")

            # CPU 使用率
            cpu_bar = create_bar(cpu_usage, (70, 90)) # 阈值: 70%, 90%
            print(f"CPU 使用率: {cpu_usage:5.1f}%   {cpu_bar}")

            # 内存使用率
            mem_bar = create_bar(mem_percent, (75, 90)) # 阈值: 75%, 90%
            print(f"内存 (RAM): {mem_used:4.2f}/{mem_total:4.2f} GB {mem_bar}")

            # Swap 使用率
            if swap_total > 0:
                swap_bar = create_bar(swap_percent, (10, 50)) # 阈值: 10%, 50%
                print(f"交换 (Swap): {swap_used:4.0f}/{swap_total:4.0f} MB {swap_bar}")
            
            print("\n按 Ctrl+C 退出。")
            
            # 刷新间隔
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n监控已停止。")

if __name__ == "__main__":
    main()