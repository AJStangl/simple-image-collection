from adlfs import AzureBlobFileSystem
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import ImageDescription, ImageCaption
from msrest.authentication import CognitiveServicesCredentials
from msrest.pipeline import ClientRawResponse


class AzureCaption:
	def __init__(self, file_system_reference):
		self.subscription_key = "2ee8459a379c4b73aef287d1cf1c4b73"
		self.endpoint = "https://aj-vision-ai.cognitiveservices.azure.com/"
		self.computer_vision_client: ComputerVisionClient = ComputerVisionClient(self.endpoint, CognitiveServicesCredentials(self.subscription_key))
		self.file_system: AzureBlobFileSystem = file_system_reference

	def generate_ai_tags(self, image_path: str) -> ImageDescription:
		try:
			computer_vision_client: ComputerVisionClient = ComputerVisionClient(endpoint=self.endpoint, credentials=CognitiveServicesCredentials(self.subscription_key))
			describe_response: ImageDescription = computer_vision_client.describe_image(self.file_system.url(image_path), max_candidates=10, raw=True)
			captions: [ImageCaption] = describe_response.captions
			tags: [str] = describe_response.tags
			meta_data = describe_response.metadata
			return describe_response
		except Exception as e:
			print(e)
			return {}
