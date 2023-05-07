import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests


class BlipCaption:
	def __init__(self, device_number: str = "0"):
		self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
		self.device = torch.device(f"cuda:{device_number}")
		self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to(self.device)


	def caption_image(self, image_path: str) -> str:
		try:
			raw_image = Image.open(image_path).convert('RGB')

			inputs = self.processor(raw_image, return_tensors="pt").to(self.device, torch.float16)

			out = self.model.generate(**inputs)

			return self.processor.decode(out[0], skip_special_tokens=True, max_new_tokens=50)

		except Exception as e:
			print(e)
			return ""

	def caption_image_from_url(self, image_url: str) -> str:
		try:
			image = Image.open(requests.get(image_url, stream=True).raw)
			inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16)

			out = self.model.generate(**inputs)
			return self.processor.decode(out[0], skip_special_tokens=True, max_new_tokens=50)
		except Exception as e:
			print(e)
			return ""