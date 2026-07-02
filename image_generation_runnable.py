from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
import requests
import os

load_dotenv()

def image_gen(prompt:str):
    try:
        nvidia_url = (
            "https://ai.api.nvidia.com/v1/images/generations"
        )
        nvidia_payload = {"prompt": prompt,"model":"qwen-image"}
        nvidia_header = {
            "Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY')}",
            "content-type": "application/json",
        }
        res = requests.post(nvidia_url, json=nvidia_payload, headers=nvidia_header)
        print(res,"===================")
        base64 = res.json()["artifacts"][0]["base64"]

        # converting base64 into http url
        imgurl = "https://api.imgbb.com/1/upload"
        img_payload = {"key": os.getenv("IMGBB_API_KEY"), "image": base64}
        img_res = requests.post(imgurl, data=img_payload)
        return img_res.json()["data"]["url"]
    except Exception as e:
        return f"Error generating image,try again later {str(e)}"

image_generation=RunnableLambda(image_gen)
print(image_generation.invoke("Transformers architecture , "))
