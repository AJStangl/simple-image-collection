
def run(model_name: str):
	from common.storage.azure_file_storage import AzureFileStorageAdapter
	file_system = AzureFileStorageAdapter("data").get_file_storage()
	file_system.download("models/sd-prompt-bot-2", "D:\\models\\sd-prompt-bot", recursive=True, overwrite=True)


if __name__ == '__main__':
	run("ContinuousDiffusion")
