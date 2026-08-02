from dotenv import load_dotenv
load_dotenv()

import base64
import os
from huggingface_hub import InferenceClient

client = InferenceClient(api_key=os.getenv("HF_TOKEN"))

image_path = "../dataset/media/images/img_008.jpg"
with open(image_path, "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

result = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                },
                {
                    "type": "text",
                    "text": (
                        "Describe this image in 2-3 sentences focusing on: "
                        "what it is (poster, screenshot, photo), what it says or shows, "
                        "and any red flags like fake urgency, OTP requests, suspicious links, "
                        "or promotional content."
                    )
                }
            ]
        }
    ],
    max_tokens=200
)

print(result.choices[0].message.content)