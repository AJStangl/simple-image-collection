# NOTE: HISTORICAL CODE, NOT USED IN PRODUCTION

# mport datetime
# import hashlib
# import logging
# import os
#
# import PIL.Image
# import pandas
# import requests
#
# from shared_code.utility.scripts.blip_caption import BlipCaption
# from shared_code.utility.storage.table import TableAdapter
# from shared_code.utility.storage.table_entry import TableEntry
# import logging
# import time
#
# from shared_code.utility.spark.set_environ import set_azure_env
#
# logging.getLogger("diffusers").setLevel(logging.WARNING)
# logging.getLogger("azure.storage").setLevel(logging.WARNING)
#
#
# class RedditDataCollector(object):
# 	def __init__(self, image_out_dir, table_name):
# 		self.table_name: str = table_name
# 		self.out_path: str = image_out_dir
# 		self.table_adapter: TableAdapter = TableAdapter()
# 		self.image_count: int = 0
# 		self.hashes: [] = []
# 		self.ids: [] = []
# 		self.blip = BlipCaption(1)
#
# 	def loop_between_dates(self, start_datetime, end_datetime):
# 		time_interval = datetime.timedelta(weeks=1)
#
# 		# Make sure the start_datetime is always a Monday by shifting the start back to monday
# 		start_datetime = start_datetime - datetime.timedelta(days=start_datetime.weekday())
#
# 		period_start_date = start_datetime
#
# 		while period_start_date < end_datetime:
# 			period_end_date = min(period_start_date + time_interval, end_datetime)
#
# 			yield period_start_date, period_end_date
#
# 			if (period_start_date + time_interval) >= end_datetime:
# 				break
#
# 			period_start_date = period_end_date
#
# 	def make_entry(self, image, text, submission_id, author, url, flair, permalink, sub_reddit, image_hash) -> dict:
# 		entry = TableEntry(PartitionKey=self.table_name,
# 						   RowKey=submission_id,
# 						   image=image,
# 						   text=text,
# 						   id=submission_id,
# 						   author=author,
# 						   url=url,
# 						   flair=flair,
# 						   permalink=permalink,
# 						   subreddit=sub_reddit,
# 						   hash=image_hash,
# 						   caption=None,
# 						   updated_caption=None,
# 						   exists=os.path.exists(image),
# 						   image_name=None,
# 						   small_image=None,
# 						   curated=False)
# 		return entry
#
# 	def download_subreddit_images(self, subreddit, start_date="2022-11-03", end_date=datetime.datetime.today().strftime('%Y-%m-%d')):
#
# 		table_client = self.table_adapter.get_table_client(self.table_name)
#
# 		all_current_images: list[dict] = list(table_client.list_entities())
#
# 		self.hashes = [x['hash'] for x in all_current_images]
#
# 		self.ids = [x['id'] for x in all_current_images]
#
# 		start_date = datetime.datetime.fromisoformat(start_date)
#
# 		end_date = datetime.datetime.fromisoformat(end_date)
#
# 		final_path = os.path.join(self.out_path, subreddit)
#
# 		print(f"== Starting {subreddit} ==")
# 		for start, end in self.loop_between_dates(start_date, end_date):
# 			submission_search_link = ('https://api.pushshift.io/reddit/submission/search/?subreddit={}&after={}&before={}&stickied=0&limit={}&mod_removed=0')
# 			search_link = f"https://api.pushshift.io/reddit/submission/search/?subreddit={subreddit}&after={int(start_date.timestamp())}&stickied=0&limit=100&mod_removed=0"
# 			# submission_search_link = submission_search_link.format(subreddit, int(start.timestamp()), int(end.timestamp()), 100)
# 			submission_response = requests.get(search_link)
# 			try:
# 				data = submission_response.json()
# 			except requests.exceptions.JSONDecodeError as r:
# 				print(f"Error decoding JSON {r}")
# 				continue
#
# 			submissions = data.get('data')
#
# 			if submissions is None:
# 				print(f"No submissions found for {subreddit} between {start} and {end}")
# 				continue
# 			try:
# 				os.mkdir(final_path)
# 			except FileExistsError:
# 				pass
#
# 			for submission in submissions:
# 				self.handle_submission(submission, data, final_path)
#
# 		print(f"All images from {subreddit} subreddit are downloaded")
# 		return self.image_count
#
# 	# note this is buggy if data is not present as a input to the method
# 	def handle_submission(self, submission, data, final_path):
# 		try:
# 			print(f"Handling submission {submission['id']}")
# 			if 'selftext' not in submission:
# 				print(f"Submission {submission['id']} has no selftext")
# 				# ignore submissions with no selftext key (buggy)
# 				return
#
# 			if submission['selftext'] in ['[removed]', '[deleted]']:
# 				print(f"Submission {submission['id']} has been removed or deleted")
# 				# ignore submissions that have no content
# 				return
#
# 			if submission.get('id') in self.ids:
# 				print(f"Submission {submission['id']} already exists")
# 				return
#
# 			if "url" in submission:
# 				image_url = submission['url']
# 				flair = submission.get('link_flair_text')
# 				title = submission.get('title')
# 				submission_id = submission.get('id')
# 				author = submission.get('author')
# 				url = submission.get('url')
# 				permalink = submission.get('permalink')
# 				subreddit = submission.get('subreddit')
#
# 				if image_url.endswith("jpg"):
# 					# Get the image file name from the URL
# 					image_name = image_url.split("/")[-1]
#
# 					# Download the image and save it to the current directory
# 					response = requests.get(image_url)
#
# 					content = response.content
# 					md5 = hashlib.md5(content).hexdigest()
# 					if md5 == "f17b01901c752c1bb04928131d1661af" or md5 == "d835884373f4d6c8f24742ceabe74946" or md5 in self.hashes:
# 						print(f"Skipping {image_name} because it is a duplicate")
# 						return
# 					else:
# 						self.hashes.append(md5)
#
# 					table_client = self.table_adapter.get_table_client(self.table_name)
#
# 					out_image = f"{final_path}/{image_name}"
#
# 					try:
# 						print(f"Writing image {out_image}")
# 						with open(out_image, "wb") as f:
# 							f.write(content)
# 							caption = self.blip.caption_image(out_image)
# 							updated_caption = None
# 							small_image = self.get_resized_image(final_path, image_name)
# 							entity = TableEntry(
# 								PartitionKey='training',
# 								RowKey=submission_id,
# 								image=out_image,
# 								text=title,
# 								id=submission_id,
# 								author=author,
# 								url=url,
# 								flair=flair,
# 								permalink=permalink,
# 								subreddit=subreddit,
# 								hash=md5,
# 								caption=caption,
# 								updated_caption=updated_caption,
# 								exists=os.path.exists(out_image),
# 								small_image=small_image,
# 								image_name=image_name,
# 								curated=False)
#
# 							to_add = entity.__dict__
# 							table_client.upsert_entity(entity=to_add)
# 							self.image_count += 1
# 							print("File downloaded\t" + image_name + "\t" + "count\t" + str(self.image_count))
# 							return
# 					except Exception as e:
# 						print(e)
# 						return
# 			else:
# 				return
#
# 		except Exception as e:
# 			print(e)
# 			return
#
# 	def resize_image(self, path: str):
# 		img = PIL.Image.open(path)
# 		try:
# 			copied_image = img.copy()
# 			result = copied_image.resize((512, 512))
# 			return result
# 		finally:
# 			img.close()
#
# 	def get_resized_image(self, path, name):
# 		try:
# 			original_path: str = path + f"/{name}"
# 			out_path: str = path + f"/thumbnail/"
# 			out_name: str = out_path + f"{name}"
# 			if not os.path.exists(out_path):
# 				os.makedirs(out_path, exist_ok=True)
# 			if os.path.exists(out_name):
# 				print(f"File {out_name} already exists. Skipping...")
# 				return out_name
# 			else:
# 				print(f"Resizing image {original_path} to {out_name}...")
# 				try:
# 					result = self.resize_image(original_path)
# 					result.save(out_name)
# 					result.close()
# 					return out_name
# 				except Exception as e:
# 					print(f"Error resizing image {original_path}...")
# 					return ""
# 		except Exception as e:
# 			print(f"Error resizing image {path}...{e}")
# 			return ""
# 		finally:
# 			pass
