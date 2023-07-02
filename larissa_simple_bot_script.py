import hashlib
import json
import logging
import random
import subprocess
import threading

import praw
import sys
import time
import torch
from praw import Reddit
from praw.reddit import Submission
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from diffusers import DiffusionPipeline, StableDiffusionPipeline

import os

os.environ['AZURE_ACCOUNT_NAME'] = "ajdevreddit"
os.environ["AZURE_TABLE_ENDPOINT"] = "https://ajdevreddit.table.core.windows.net/"
os.environ["AZURE_QUEUE_ENDPOINT"] = "https://ajdevreddit.queue.core.windows.net/"
os.environ["AZURE_ACCOUNT_KEY"] = "+9066TCgdeVignRdy50G4qjmNoUJuibl9ERiTGzdV4fwkvgdV3aSVqgLwldgZxj/UpKLkkfXg+3k+AStjFI33Q=="
os.environ["AZURE_STORAGE_CONNECTION_STRING"] = "DefaultEndpointsProtocol=https;AccountName=ajdevreddit;AccountKey=+9066TCgdeVignRdy50G4qjmNoUJuibl9ERiTGzdV4fwkvgdV3aSVqgLwldgZxj/UpKLkkfXg+3k+AStjFI33Q==;BlobEndpoint=https://ajdevreddit.blob.core.windows.net/;QueueEndpoint=https://ajdevreddit.queue.core.windows.net/;TableEndpoint=https://ajdevreddit.table.core.windows.net/;FileEndpoint=https://ajdevreddit.file.core.windows.net/"
os.environ["AZURE_VISION_API_KEY"] = "2ee8459a379c4b73aef287d1cf1c4b73"
os.environ["AZURE_VISION_ENDPOINT"] = "https://aj-vision-ai.cognitiveservices.azure.com/"
os.environ["client_id"] = "5hVavL0PIRyM_1JSvqT6UQ"
os.environ["client_secret"] = "BjD2kS3WNLnJc59RKY-JJUuc_Z9-JA"
os.environ["password"] = "Guitar!01"
os.environ["reddit_username"] = "KimmieBotGPT"


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from common.utility.storage.blob import BlobAdapter
from common.utility.storage.table import TableAdapter

logging.getLogger("diffusers").setLevel(logging.WARNING)
logging.getLogger("azure.storage").setLevel(logging.WARNING)


class FuckingStatic:
	@staticmethod
	def validate_message(message):
		try:
			import re
			start_end_regex = re.compile("<\|startoftext\|>(.+?)<\|endoftext\|>")
			model_regex = re.compile("<\|model\|>(.+?)<\|title\|>")
			title_regex = re.compile("<\|title\|>(.+?)<\|caption\|>")
			prompt_regex = re.compile("<\|caption\|>(.+?)<\|endoftext\|>")
			found_start_end = start_end_regex.findall(message)
			if len(found_start_end) == 0:
				return "", ""

			generated_prompt = ""
			generated_text = ""
			model_text = ""

			found_model = model_regex.findall(message)
			if len(found_model) > 0:
				model_text = found_model[0]

			found_prompt = prompt_regex.findall(message)
			if len(found_prompt) > 0:
				generated_prompt = found_prompt[0]

			found_text = title_regex.findall(message)
			if len(found_text) > 0:
				generated_text = found_text[0]

			return model_text.strip(), generated_text.strip(), generated_prompt.strip()
		except Exception as e:
			print(e)
			return "", "", ""

	@staticmethod
	def create_enhanced_image(image_path) -> str:
		out_path = "D:\\images\\results"
		# return image_path
		env_copy = os.environ.copy()
		env_copy['CUDA_VISIBLE_DEVICES'] = "''"
		print(f"Starting Script For Face Restoration for {image_path}")

		print(f"Torch Status: {torch.cuda.is_available()}")

		command = f"D:\\workspaces\\General\\venv\\Scripts\\python.exe D:\\code\\repos\\GFPGAN\\inference_gfpgan.py -i {image_path} -v 1.4 -s 2 -o {out_path}"

		print(f"Running command: {command}")

		result = subprocess.call(command, shell=True, env={**env_copy})

		if result == 0:
			final_path = f"{out_path}\\restored_imgs\\{os.path.basename(image_path)}"
			print(
				f"Success: Image Processed and located in {final_path}")
			return final_path
		else:
			print(f"Error: Image Processing Failed")
			return image_path

	@staticmethod
	def create_image(prompt: str, pipe: StableDiffusionPipeline, device_name: str) -> (str, int, int):
		try:
			pipe.to("cuda:" + device_name)
			guidance_scale = random.randint(7, 8)
			num_inference_steps = random.randint(80, 100)
			args = {
				"model": pipe.config_name,
				"guidance_scale": guidance_scale,
				"num_inference_steps": num_inference_steps
			}
			print(json.dumps(args, indent=4))

			height = 512
			width = [512, 768]
			initial_image = pipe(prompt, height=height, width=random.choice(width), guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]

			hash_name = f"{hashlib.md5(prompt.encode()).hexdigest()}"

			upload_file = f"{hash_name}.png"

			image_path = f"D://images//{upload_file}"

			initial_image.save(image_path)

			final_image = FuckingStatic.create_enhanced_image(image_path)

			return final_image, guidance_scale, num_inference_steps

		except Exception as e:
			print(e)
			return None
		finally:
			del pipe
			torch.cuda.empty_cache()


class PipeLineHolder(object):
	pipe_line_name: str
	diffusion_pipeline_path: str
	text_model_path: str

	def __init__(self, pipe_line_name: str, diffusion_pipeline_path: str, text_model_path: str):
		self.pipe_line_name: str = pipe_line_name
		self.diffusion_pipeline_path: str = diffusion_pipeline_path
		self.text_model_path: str = text_model_path


class SimpleBot(threading.Thread):
	def __init__(self, holder: [PipeLineHolder], proc_name: str, instance: int):
		super().__init__(name=proc_name, daemon=True)
		self.holders: [] = holder
		self.table_broker: TableAdapter = TableAdapter()
		self.poll_for_message_worker_thread = threading.Thread(target=self.main_process, args=(), daemon=True,
															   name=proc_name)
		self.things_to_say: [str] = []
		self.counter = 0
		self.instance = instance

	def get_gpt_model(self, model_to_use: PipeLineHolder) -> (GPT2Tokenizer, GPT2LMHeadModel):
		tokenizer = GPT2Tokenizer.from_pretrained(model_to_use.text_model_path)
		model = GPT2LMHeadModel.from_pretrained(model_to_use.text_model_path)
		return tokenizer, model

	def create_prompt(self, pipe_line_holder: PipeLineHolder):
		subs = ['AesPleasingAsianGirls',
				'AmIhotAF',
				'Amicute',
				'AsianInvasion',
				'AsianOfficeLady',
				'CityPorn',
				'CollaredDresses',
				'DLAH',
				'Dresses',
				'DressesPorn',
				'EarthPorn',
				'HotGirlNextDoor',
				'Ifyouhadtopickone',
				'KoreanHotties',
				'PrettyGirls',
				'SFWNextDoorGirls',
				'SFWRedheads',
				'SlitDresses',
				'TrueFMK',
				'WomenInLongDresses',
				'amihot',
				'bathandbodyworks',
				'celebrities',
				'cougars_and_milfs_sfw',
				'fatsquirrelhate',
				'gentlemanboners',
				'hotofficegirls',
				'itookapicture',
				'prettyasiangirls',
				'realasians',
				'selfies',
				'sexygirls',
				'sfwpetite',
				'tightdresses'
				]
		try:
			device = torch.device(f"cuda:{self.instance}")

			tokenizer, model = self.get_gpt_model(pipe_line_holder)

			question = f"<|startoftext|> <|model|> {random.choice(subs)} <|title|>"

			prompt = f"{question}"

			generation_prompt = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")

			model.to(device)

			generation_prompt.to(device)

			inputs = generation_prompt.input_ids

			attention_mask = generation_prompt['attention_mask']

			sample_outputs = model.generate(inputs=inputs,
											attention_mask=attention_mask,
											do_sample=True,
											max_length=50,
											num_return_sequences=1,
											repetition_penalty=1.1)
			subreddit_type = ""
			title = ""
			prompt = ""
			for i, sample_output in enumerate(sample_outputs):
				result = tokenizer.decode(sample_output, skip_special_tokens=False)
				print(result)
				subreddit_type, title, prompt = FuckingStatic.validate_message(result)

			model.to("cpu")
			generation_prompt.to("cpu")
			torch.cuda.empty_cache()

			return subreddit_type, title, prompt

		except Exception as e:
			print(e)
			return self.create_prompt(pipe_line_holder)

	def write_image_to_cloud(self, image_name):
		with open(image_name, "rb") as f:
			image_data = f.read()

			blob_adapter = BlobAdapter("images")

			blob_adapter.upload_blob(blob_name=os.path.basename(image_name), data=image_data)

			final_remote_path = f"https://ajdevreddit.blob.core.windows.net/images/{os.path.basename(image_name)}"

			print(final_remote_path)

			return final_remote_path

	def write_output_to_table_storage_row(self, final_remote_path, prompt):
		random_id = random.randint(0, 123456890)
		entity = {
			"PartitionKey": "General",
			"RowKey": str(random_id),
			"Text": final_remote_path,
			"Prompt": prompt,
			"Sender": "ImageBot",
			"CommentId": random_id,
			"Topic": "General",
			"ConnectionId": "chat-output",
			"IsBot": True,
			"IsThinking": False,
			"Channel": "General"
		}
		table_adapter: TableAdapter = TableAdapter()
		table_adapter.upsert_entity_to_table("messages", entity)

	def main_process(self):
		while True:
			model_index = self.counter % len(self.holders)

			holder: PipeLineHolder = self.holders[model_index]

			print(f":: Using model: {holder.diffusion_pipeline_path}")

			print(f":: Using Device: {self.instance}")

			pipe = StableDiffusionPipeline.from_pretrained(holder.diffusion_pipeline_path, revision="fp16",
														   torch_dtype=torch.float32, safety_checker=None)

			print(":: Model Loaded")

			subreddit_type, title, prompt = self.create_prompt(holder)

			print(":: Prompt Created")

			image_prompt = prompt.replace("little girl", "petite women")

			print("Subreddit Type: " + subreddit_type)
			print("Title: " + title)
			print("Prompt: " + image_prompt)

			try:
				(image_output, guidance, num_steps) = FuckingStatic.create_image(image_prompt, pipe, str(self.instance))
			except Exception as e:
				print(e)
				continue

			try:
				instance: Reddit = praw.Reddit(site_name="KimmieBotGPT")
				sub = instance.subreddit("CoopAndPabloArtHouse")

				flair_map: dict = {
					"CosmicDiffusion": "80192e8a-c116-11ed-9afc-061002270b1c",
					"SexyDiffusion": "6c02c0aa-c116-11ed-a36b-625bab71eac2",
					"MemeDiffusion": "52782e72-c116-11ed-8d42-9226dee3c916",
					"CityDiffusion": "3f3db71e-c116-11ed-bc88-4257f93035d0",
					"NatureDiffusion": "49b00d00-c116-11ed-80c5-7ef7afdcdf7d",
					"RedHeadDiffusion": "4b0844f8-c68c-11ed-8e01-0a0ff85df53d",
					"ITAPDiffusion": "548c5f8a-c70b-11ed-ab4c-d6ecb5af116d",
					"SWFPetite": "494efcc6-ccbb-11ed-b813-aec5374833c4",
					"FatSquirrelDiffusion": "331f2a78-ccef-11ed-b813-beb8ea0d6477",
					"AsianOfficeLadyDiffusion": "0e69c05e-cf41-11ed-a6d7-c265ef3d634d",
					"SexyAsianDiffusion": "e978fd72-d0cc-11ed-802d-922e8d939dd5",
					"MildlyPenisDiffusion": "7aedfca4-d676-11ed-9536-6a42b6ad77bd",
					"TTTDiffusion": "25d50538-d6f7-11ed-9f0f-6a1b95511d30",
					"PrettyGirlDiffusion": "6c02c0aa-c116-11ed-a36b-625bab71eac2"
				}

				submission: Submission = sub.submit_image(
					title=f"{title}",
					image_path=image_output,
					nsfw=False,
					flair_id=flair_map.get(holder.pipe_line_name))

				submission.mod.approve()

				body = f"""
| Prompt         |       Model Name        | Guidance   | Number Of Inference Steps |
|:---------------|:-----------------------:|------------|--------------------------:|
| {prompt}          | {sub} | {guidance} |               {num_steps} |
				"""

				submission.reply(body)
				self.counter += 1

				if self.counter % 100 == 0:
					sub = instance.subreddit("CoopAndPabloPlayHouse")
					submission: Submission = sub.submit_image(
						title=f"{title}",
						image_path=image_output, nsfw=False)
					submission.save()

				# final_remote_path = self.write_image_to_cloud(image_output)
				# self.write_output_to_table_storage_row(final_remote_path, image_prompt)


			except Exception as e:
				print(e)
				self.counter += 1
				continue

	def run(self):
		self.poll_for_message_worker_thread.start()

	def stop(self):
		sys.exit(0)


if __name__ == '__main__':
	prompt_model: str = "D:\\code\\repos\\simple-collection\\notebooks\\pipelines\\images\\sd-prompt-bot"
	pipeline_1 = PipeLineHolder("ContinuousDiffusion", "D:\\models\\ContinuousDiffusion", prompt_model)
	pipe_line_holder_list = [pipeline_1]

	print(":: Starting Bot")
	random.shuffle(pipe_line_holder_list)

	bot: SimpleBot = SimpleBot(pipe_line_holder_list, "SimpleBot", sys.argv[1])
	bot.main_process()
	while True:
		try:
			time.sleep(1)
		except KeyboardInterrupt:
			logging.info('Shutdown')
			bot.stop()
			exit(0)
