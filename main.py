import os
import logging
import time

from azure.core.credentials import AzureNamedKeyCredential
from azure.core.paging import ItemPaged
from azure.data.tables import TableClient, TableEntity, TableServiceClient
from tqdm import tqdm
from PIL import Image
import requests
import torch
from common.captioning.caption import BlipCaption
from common.functions.functions import Functions
from adlfs import AzureBlobFileSystem

os.environ["AZURE_ACCOUNT_NAME"] = "ajdevreddit"
os.environ["AZURE_TABLE_ENDPOINT"] = "https://ajdevreddit.table.core.windows.net/"
os.environ["AZURE_QUEUE_ENDPOINT"] = "https://ajdevreddit.queue.core.windows.net/"
os.environ[
	"AZURE_ACCOUNT_KEY"] = "+9066TCgdeVignRdy50G4qjmNoUJuibl9ERiTGzdV4fwkvgdV3aSVqgLwldgZxj/UpKLkkfXg+3k+AStjFI33Q=="
os.environ[
	"AZURE_STORAGE_CONNECTION_STRING"] = "DefaultEndpointsProtocol=https;AccountName=ajdevreddit;AccountKey=+9066TCgdeVignRdy50G4qjmNoUJuibl9ERiTGzdV4fwkvgdV3aSVqgLwldgZxj/UpKLkkfXg+3k+AStjFI33Q==;BlobEndpoint=https://ajdevreddit.blob.core.windows.net/;QueueEndpoint=https://ajdevreddit.queue.core.windows.net/;TableEndpoint=https://ajdevreddit.table.core.windows.net/;FileEndpoint=https://ajdevreddit.file.core.windows.net/"
os.environ["AZURE_VISION_API_KEY"] = "2ee8459a379c4b73aef287d1cf1c4b73"
os.environ["AZURE_VISION_ENDPOINT"] = "https://aj-vision-ai.cognitiveservices.azure.com/"

from common.storage.azure_file_storage import AzureFileStorageAdapter


class TableAdapter(object):
	credential = AzureNamedKeyCredential(os.environ["AZURE_ACCOUNT_NAME"], os.environ["AZURE_ACCOUNT_KEY"])

	def __init__(self):
		self.service: TableServiceClient = TableServiceClient(endpoint=os.environ["AZURE_TABLE_ENDPOINT"],
															  credential=self.credential)
		self.tables = self.service.list_tables()

	def get_table_service_client(self) -> TableServiceClient:
		return self.service

	def perform_odata_query(self, table_name: str, query: str) -> list[dict]:
		table_client: TableClient = self.get_table_client(table_name=table_name)
		entities: ItemPaged[TableEntity] = table_client.query_entities(query)
		return list(entities)

	def get_table_client(self, table_name: str) -> TableClient:
		service: TableServiceClient = self.get_table_service_client()
		return service.get_table_client(table_name=table_name)

	def upsert_entity_to_table(self, table_name: str, entity: dict):
		table_client: TableClient = self.get_table_client(table_name=table_name)
		table_client.upsert_entity(entity=entity)
		return

	def get_all_entities(self, table_name: str) -> list[dict]:
		table_client: TableClient = self.get_table_client(table_name=table_name)
		entities: ItemPaged[TableEntity] = table_client.list_entities()
		return list(entities)

	def get_table_client_instance(self, table_name: str) -> TableClient:
		service: TableServiceClient = self.get_table_service_client()
		return service.get_table_client(table_name=table_name)

	def get_entity(self, table_name: str, partition_key: str, row_key: str) -> TableEntity:
		table_client: TableClient = self.get_table_client(table_name=table_name)
		entity: TableEntity = table_client.get_entity(partition_key=partition_key, row_key=row_key)
		return entity


def caption_image_from_url(model, image_url: str) -> str:
	try:
		image = Image.open(requests.get(image_url, stream=True).raw)
		device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
		inputs = model.processor(image, return_tensors="pt").to(device)

		out = model.cpu_model.generate(**inputs)
		return model.processor.decode(out[0], skip_special_tokens=True, max_new_tokens=200)
	except Exception as e:
		print(e)
		return ""


def run():
	logging.getLogger("azure.storage").setLevel(logging.WARNING)

	print("=== Starting 0-2 Blip Image Captioning ===")

	tqdm.pandas(desc="Progress")

	file_system = AzureFileStorageAdapter('data').get_file_storage()

	caption_0 = BlipCaption("cpu")

	table_client = TableAdapter().get_table_client("curationSecondary")

	while True:
		entities = list(table_client.query_entities("accept eq true and thumbnail_accept eq false and thumbnail_curated eq false and caption eq ''"))
		total = len(entities)

		for elem in tqdm(entities, total=total, desc="Captioning"):
			try:

				thumbnail_path = elem["thumbnail_path"]
				if file_system.exists(thumbnail_path):
					url = file_system.url(thumbnail_path)
					caption = caption_image_from_url(caption_0, url)

					if caption.startswith("ara"):
						caption = " ".join(caption.split(" ")[1:])

					elem['caption'] = caption
					elem["smart_caption"] = caption
					print(caption, elem["id"])
				else:
					print(f"No image found on thumbnail_path for {elem['id']}")

				if file_system.exists(elem['pil_thumbnail_path']):
					url_for_pil = file_system.url(elem["pil_thumbnail_path"])
					caption = caption_image_from_url(caption_0, url_for_pil)

					if caption.startswith("ara"):
						caption = " ".join(caption.split(" ")[1:])

					elem["pil_caption"] = caption
					print(caption, elem["id"])
				else:
					print(f"No existing pil_image for {elem['id']}")

				table_client.upsert_entity(entity=elem)

			except Exception as e:
				continue

		time.sleep(60 * 5)


if __name__ == '__main__':
	run()
