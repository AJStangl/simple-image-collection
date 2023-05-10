import os

from adlfs import AzureBlobFileSystem


class AzureFileStorageAdapter(object):
	def __init__(self, container_name: str = "data"):
		self.__account_name: str = os.environ["AZURE_STORAGE_ACCOUNT_NAME"]
		self.__account_key: str = os.environ["AZURE_STORAGE_ACCOUNT_KEY"]
		self.container_name: str = container_name

	def get_file_storage(self) -> AzureBlobFileSystem:
		return AzureBlobFileSystem(
			account_name=self.__account_name,
			account_key=self.__account_key,
			container_name=self.container_name)

	def get_file_storage_root(self) -> AzureBlobFileSystem:
		return AzureBlobFileSystem(account_name=self.__account_name, account_key=self.__account_key,
								   container_name=self.container_name)
