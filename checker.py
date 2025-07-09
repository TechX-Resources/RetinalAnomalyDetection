from openai import OpenAI
from PIL import Image
import base64
import io
import os

"""
< In order not to waste computaitonal resources, we'll first check whether the image is valid or not >
< If not valid, then no need to use the model on it >
< Else if, then we use the model >
< In this case, we wouldnt spend time prediciting a visual that the model may be not accurately predict >
"""

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Key not set in environment")

client = OpenAI(api_key=api_key)

def encode_image(image_path):
    with Image.open(image_path).convert("RGB") as img:
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

def is_retinal_image_openai(image_path):
    encoded_image = encode_image(image_path)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a medical image classifier."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Is this image a retinal fundus scan or a human retina? Just say 'yes' or 'no'."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                ],
            },
        ],
        temperature=0,
    )

    answer = response.choices[0].message.content.strip().lower()
    return answer.startswith("yes")
