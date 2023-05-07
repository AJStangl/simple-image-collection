from adlfs import AzureBlobFileSystem


class AzureFileStorageAdapter(object):
	def __init__(self, container_name: str):
		self._account_name: str = "ajdevreddit"
		self._account_key: str = "+9066TCgdeVignRdy50G4qjmNoUJuibl9ERiTGzdV4fwkvgdV3aSVqgLwldgZxj/UpKLkkfXg+3k+AStjFI33Q=="
		self.container_name: str = container_name

	def get_file_storage(self) -> AzureBlobFileSystem:
		return AzureBlobFileSystem(
			account_name=self._account_name,
			account_key=self._account_key,
			container_name=self.container_name)

	def get_file_storage_root(self) -> AzureBlobFileSystem:
		return AzureBlobFileSystem(account_name="ajdevreddit",
								   account_key="+9066TCgdeVignRdy50G4qjmNoUJuibl9ERiTGzdV4fwkvgdV3aSVqgLwldgZxj/UpKLkkfXg+3k+AStjFI33Q==",
								   container_name="data")
