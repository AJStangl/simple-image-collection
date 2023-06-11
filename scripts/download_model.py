
def run(model_name: str):
	from common.storage.azure_file_storage import AzureFileStorageAdapter
	file_system = AzureFileStorageAdapter("data").get_file_storage()
	file_system.download("models/ContinuousDiffusion4", "D:\\models\\ContinuousDiffusion4", recursive=True, overwrite=True)


if __name__ == '__main__':
	run("ContinuousDiffusion")
