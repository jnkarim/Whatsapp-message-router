# test_blip.py
from dotenv import load_dotenv
load_dotenv()

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

image = Image.open("../dataset/media/images/img_008.jpg")
inputs = processor(image, return_tensors="pt")
out = model.generate(**inputs)
result = processor.decode(out[0], skip_special_tokens=True)
print(result)