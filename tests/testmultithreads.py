
import threading
import time

def worker(thread_id):
    print(f"线程 {thread_id} 开始工作")
    time.sleep(2)  # 模拟耗时操作
    print(f"线程 {thread_id} 工作完成")

if __name__ == "__main__":
    threads = []
    for i in range(3):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    print("所有线程执行完毕")
