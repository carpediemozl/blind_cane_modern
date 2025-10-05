import os
import base64
from volcenginesdkarkruntime import Ark

client = Ark(api_key="f317e062-879b-40af-b520-0f235cf2bf94")

def load_image(image_path):
    """验证并加载本地图片"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片路径不存在: {image_path}")
    
    with open(image_path, "rb") as f:
        return f.read()

if __name__ == "__main__":
     
    # 处理本地图片（示例路径，实际使用时需替换）
    image_path = r"/home/pi/Downloads/111.jpg"
    # 加载并编码图片
    image_data = load_image(image_path)
    base64_image = base64.b64encode(image_data).decode('utf-8')
    resp = client.chat.completions.create(
        model="doubao-1-5-vision-pro-32k-250115",
       
        
        # 构造消息体
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpg;base64,{base64_image}"
                    },
                    {
                        "type": "text",
                        "text": "图片主要讲了什么?"
                    }
                ]
            }
        ]
    )
    print(resp.choices[0].message.content)
