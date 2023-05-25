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
from diffusers import StableDiffusionPipeline
from praw import Reddit
from praw.reddit import Submission
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from common import load_dotenv

load_dotenv()

from common.utility.storage.blob import BlobAdapter
from common.utility.storage.table import TableAdapter

logging.getLogger("diffusers").setLevel(logging.WARNING)
logging.getLogger("azure.storage").setLevel(logging.WARNING)

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


class FuckingStatic:
	@staticmethod
	def validate_message(message):
		try:
			import re
			start_end_regex = re.compile("<\|startoftext\|>(.+?)<\|endoftext\|>")
			prompt_regex = re.compile("<\|prompt\|>(.+?)<\|text\|>")
			text_regex = re.compile("<\|text\|>(.+?)<\|endoftext\|>")
			found_start_end = start_end_regex.findall(message)
			if len(found_start_end) == 0:
				return "", ""

			generated_prompt = ""
			generated_text = ""

			found_prompt = prompt_regex.findall(message)
			if len(found_prompt) > 0:
				generated_prompt = found_prompt[0]

			found_text = text_regex.findall(message)
			if len(found_text) > 0:
				generated_text = found_text[0]

			return generated_prompt.strip(), generated_text.strip()
		except Exception as e:
			print(e)
			return "", ""

	@staticmethod
	def create_enhanced_image(image_path) -> str:
		out_path = "D:\\images\\results"
		# return image_path
		env_copy = os.environ.copy()
		env_copy['CUDA_VISIBLE_DEVICES'] = "''"
		print(f"Starting Script For Face Restoration for {image_path}")

		print(f"Torch Status: {torch.cuda.is_available()}")

		command = f"python D:\\code\\repos\\GFPGAN\\inference_gfpgan.py -i {image_path} -v 1.4 -s 2 -o {out_path}"

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
			num_inference_steps = random.randint(80, 81)
			args = {
				"model": pipe.config_name,
				"guidance_scale": guidance_scale,
				"num_inference_steps": num_inference_steps
			}
			print(json.dumps(args, indent=4))

			initial_image = pipe(prompt, height=512, width=512, guidance_scale=guidance_scale,
								 num_inference_steps=num_inference_steps).images[0]

			hash_name = f"{hashlib.md5(prompt.encode()).hexdigest()}"

			upload_file = f"{hash_name}.png"

			image_path = f"D://images//{upload_file}"

			initial_image.save(image_path)

			final_image = FuckingStatic.create_enhanced_image(image_path)

			# final_image = image_path

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

		try:
			device = torch.device(f"cuda:{self.instance}")

			tokenizer, model = self.get_gpt_model(pipe_line_holder)

			question = f"<|startoftext|> <|model|> {pipe_line_holder.pipe_line_name} <|prompt|>"

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

			prompt_for_reddit = ""
			prompt_for_image_generation = ""
			for i, sample_output in enumerate(sample_outputs):
				result = tokenizer.decode(sample_output, skip_special_tokens=False)
				print(result)
				prompt, text = FuckingStatic.validate_message(result)
				prompt_for_reddit = prompt
				prompt_for_image_generation = text

			model.to("cpu")
			generation_prompt.to("cpu")
			torch.cuda.empty_cache()

			return prompt_for_reddit, prompt_for_image_generation

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
														   torch_dtype=torch.float16, safety_checker=None)
			print(":: Model Loaded")

			reddit_text, image_prompt = self.create_prompt(holder)
			print(":: Prompt Created")

			image_prompt = image_prompt.replace("little girl", "petite women")

			print("Reddit Text: " + reddit_text)

			print("Prompt: " + image_prompt)

			gen = f"{reddit_text} : {image_prompt}"
			# gen = image_prompt

			try:
				(image_output, guidance, num_steps) = FuckingStatic.create_image(gen, pipe, str(self.instance))
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
					"TTTDiffusion": "25d50538-d6f7-11ed-9f0f-6a1b95511d30"
				}

				submission: Submission = sub.submit_image(
					title=f"{reddit_text}",
					image_path=image_output,
					nsfw=False,
					flair_id=flair_map.get(holder.pipe_line_name))

				submission.mod.approve()

				body = f"""
| Prompt         |       Model Name        | Guidance   | Number Of Inference Steps |
|:---------------|:-----------------------:|------------|--------------------------:|
| {gen}          | {os.path.split(holder.diffusion_pipeline_path)[-1]} | {guidance} |               {num_steps} |
				"""

				submission.reply(body)
				self.counter += 1

				if self.counter % 100 == 0:
					sub = instance.subreddit("CoopAndPabloPlayHouse")
					submission: Submission = sub.submit_image(
						title=f"{reddit_text}",
						image_path=image_output, nsfw=False)
					submission.save()

				final_remote_path = self.write_image_to_cloud(image_output)
				self.write_output_to_table_storage_row(final_remote_path, image_prompt)


			except Exception as e:
				print(e)
				self.counter += 1
				continue

	def run(self):
		self.poll_for_message_worker_thread.start()

	def stop(self):
		sys.exit(0)


if __name__ == '__main__':

	prompt_model: str = "D:\\models\\sd-prompt-bot-7"

	pipeline_1 = PipeLineHolder("SexyDiffusion", "D:\\models\\SexyDiffusion", prompt_model)

	pipeline_2 = PipeLineHolder("NatureDiffusion", "D:\\models\\NatureDiffusion", prompt_model)

	pipeline_3 = PipeLineHolder("CityDiffusion", "D:\\models\\CityScapes", prompt_model)

	pipeline_4 = PipeLineHolder("CosmicDiffusion", "D:\\models\\CosmicDiffusion", prompt_model)

	pipeline_5 = PipeLineHolder("MemeDiffusion", "D:\\models\\MemeDiffusion", prompt_model)

	pipeline_6 = PipeLineHolder("RedHeadDiffusion", "D:\\models\\RedHeadDiffusion", prompt_model)

	pipeline_7 = PipeLineHolder("ITAPDiffusion", "D:\\models\\ITAPDiffusion", prompt_model)

	pipeline_8 = PipeLineHolder("SWFPetite", "D:\\models\\SWFPetite", prompt_model)

	pipeline_9 = PipeLineHolder("FatSquirrelDiffusion", "D:\\models\\FatSquirrelDiffusion", prompt_model)

	pipeline_10 = PipeLineHolder("SexyAsianDiffusion", "D:\\models\\AsianDiffusion", prompt_model)

	pipeline_12 = PipeLineHolder("MildlyPenisDiffusion", "D:\\models\\MildlyPenisDiffusion", prompt_model)

	pipeline_13 = PipeLineHolder("TTTDiffusion", "D:\\models\\TTTDiffusion", prompt_model)

	pipe_line_holder_list = [
		pipeline_1,
		pipeline_2,
		pipeline_3,
		pipeline_4,
		pipeline_5,  # , pipeline_5, pipeline_5,
		pipeline_6,
		pipeline_7,  # , pipeline_7, pipeline_7,
		pipeline_8,
		pipeline_9,  # , pipeline_9, pipeline_9,  pipeline_9, pipeline_9,
		pipeline_10,
		pipeline_12,  # , pipeline_12, pipeline_12, pipeline_12,
		pipeline_13  # , pipeline_13, pipeline_13, pipeline_13, pipeline_13, pipeline_13
	]

	print(":: Starting Bot")
	random.shuffle(pipe_line_holder_list)

	bot: SimpleBot = SimpleBot(pipe_line_holder_list, "SimpleBot", sys.argv[1])
	bot.start()
	while True:
		try:
			time.sleep(1)
		except KeyboardInterrupt:
			logging.info('Shutdown')
			bot.stop()
			exit(0)
