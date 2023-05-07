import hashlib
import os
from datetime import datetime
from io import BytesIO
import random

import requests
from PIL import Image

from common.captioning.caption import BlipCaption


class Functions:
	def get_hash_from_path(self, in_path: str):
		if os.path.exists(in_path):
			with open(in_path, 'rb') as f_:
				content = f_.read()
				result = hashlib.md5(content).hexdigest()
				return result, content
		else:
			return ""

	def fetch_image(self, x: object, file_list__, file_system) -> object:
		with open('log.txt', 'a') as f_image:
			try:
				url = x['original_url']
				subreddit = x['subreddit']
				image_id = x['id']
				os.makedirs(f"temp\\image\\{subreddit}", exist_ok=True)
				temp_path = f"temp\\image\\{subreddit}\\{image_id}.jpg"
				out_path = f"data/image/{image_id}.jpg"
				if os.path.exists(temp_path):
					md5, content = self.get_hash_from_path(temp_path)
					if md5 != "f17b01901c752c1bb04928131d1661af" or md5 != "d835884373f4d6c8f24742ceabe74946":
						if out_path in file_list__:
							return out_path
						else:
							file_system.upload(temp_path, out_path, overwrite=True)
							return out_path
					else:
						return ""
				else:
					response = requests.get(url)
					md5 = hashlib.md5(response.content).hexdigest()
					if md5 != "f17b01901c752c1bb04928131d1661af" or md5 != "d835884373f4d6c8f24742ceabe74946":
						try:
							raw_image = Image.open(BytesIO(response.content))
							raw_image.save(temp_path)
							raw_image.close()
							if out_path in file_list__:
								return out_path
							else:
								file_system.upload(temp_path, out_path)
								return out_path
						except Exception as ex:
							message = self.write_log_message(x['id'], x['subreddit'], "Failure in fetch_image", ex)
							f_image.write(message)
							return ""
					else:
						return ""
			except Exception as ex:
				message = self.write_log_message(x['id'], x['subreddit'], "Failure in fetch_image", ex)
				f_image.write(message)
				return ""

	def get_name_for_image(self, x: object, file_list__) -> str:
		path = x['path']
		if path != "" and path in file_list__:
			return os.path.basename(path)
		else:
			return ""

	def set_exists(self, x: object) -> bool:
		try:
			sub_reddit = x['subreddit']
			record_id = x['id']
			temp_path = f"temp\\image\\{sub_reddit}\\{record_id}.jpg"
			return os.path.exists(temp_path)
		except Exception as ex:
			return False

	def set_hash(self, x: object):
		sub_reddit = x['subreddit']
		record_id = x['id']
		temp_path = f"temp\\image\\{sub_reddit}\\{record_id}.jpg"
		if os.path.exists(temp_path):
			return hashlib.md5(open(temp_path, 'rb').read()).hexdigest()
		else:
			return ""

	def add_source(self, x: object, source_list) -> str:
		sub_reddit = x['subreddit']
		for source in source_list:
			data_source = source['data']
			source_name = source['name']
			if sub_reddit in data_source:
				return source_name
		return ""

	def write_log_message(self, submission_id: str, subreddit: str, message: str, exception: Exception) -> str:
		date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		return f"{date_time}\t{subreddit}\t{submission_id}\t{message}\t{exception}\n"

	def apply_caption(self, x: object, caption_routine: [BlipCaption, BlipCaption]) -> str:
		with open('log.txt', 'a') as f_3:
			exists = x['exists']
			if not exists:
				return ""
			sub_reddit = x['subreddit']
			record_id = x['id']
			temp_path = f"temp\\image\\{sub_reddit}\\{record_id}.jpg"

			if os.path.exists(temp_path):
				try:
					result = random.choice(caption_routine).caption_image(temp_path)
					return result
				except Exception as ex:
					message = self.write_log_message(x['id'], x['subreddit'], "Failure in apply_caption", ex)
					f_3.write(message)
					return ""
			else:
				return ""

	def fix_path(self, x:object, fl: []) -> str:
		current_path = x['path']
		exists = x['exists']
		if current_path in fl:
			return current_path
		else:
			image_id = x['id']
			if exists:
				return f"data/image/{image_id}.jpg"
			else:
				return ""