import logging
import os
import random
from typing import Union

import pandas
from adlfs import AzureBlobFileSystem
from azure.ai.vision import VisionServiceOptions, VisionSource, ImageAnalysisOptions, ImageAnalysisFeature, \
	ImageAnalyzer, ImageAnalysisResultDetails
from azure.cognitiveservices.vision.computervision import ComputerVisionClientConfiguration, ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import ImageDescription
from azure.ai.vision import ImageAnalysisResult


class AzureCaption(object):
	def __init__(self, file_system_reference):
		self.__subscription_key: str = os.environ["AZURE_VISION_KEY"]
		self.__endpoint: str = os.environ["AZURE_VISION_ENDPOINT"]
		self.__credentials: ComputerVisionClientConfiguration = ComputerVisionClientConfiguration(
			endpoint=self.__endpoint, credentials=self.__subscription_key)
		self.__vision_client: ComputerVisionClient = ComputerVisionClient(endpoint=self.__endpoint,
																		  credentials=self.__credentials)
		self.__file_system_reference: AzureBlobFileSystem = file_system_reference

	def generate_ai_tags(self, image_path: str) -> Union[ImageDescription, None]:
		try:
			describe_response: ImageDescription = self.__vision_client.describe_image(
				self.__file_system_reference.url(image_path), max_candidates=10, raw=True)
			return describe_response
		except Exception as e:
			print(e)
			return None

	def _get_image_analyzer(self, image_path: str = None, image_url: str = None) -> ImageAnalyzer:

		service_options: VisionServiceOptions = VisionServiceOptions(self.__endpoint, self.__subscription_key)
		vision_source: VisionSource = VisionSource(filename=image_path, url=image_url)
		analysis_options: ImageAnalysisOptions = ImageAnalysisOptions()

		analysis_options.features = (
				ImageAnalysisFeature.CROP_SUGGESTIONS |
				ImageAnalysisFeature.CAPTION |
				ImageAnalysisFeature.DENSE_CAPTIONS |
				ImageAnalysisFeature.OBJECTS |
				ImageAnalysisFeature.PEOPLE |
				ImageAnalysisFeature.TEXT |
				ImageAnalysisFeature.TAGS
		)

		analysis_options.cropping_aspect_ratios = [1.0, 1.0]
		analysis_options.language = "en"

		analysis_options.model_version = "latest"
		analysis_options.gender_neutral_caption = False

		image_analyzer: ImageAnalyzer = ImageAnalyzer(service_options, vision_source, analysis_options)
		return image_analyzer

	def image_analysis(self, image_path: str) -> Union[ImageAnalysisResultDetails, None]:
		"""
		Analyze image from file, all features, synchronous (blocking)
		"""
		try:
			analyzer = self._get_image_analyzer(image_path=None, image_url=image_path)
			result_analysis: ImageAnalysisResult = analyzer.analyze()
			result_details: ImageAnalysisResultDetails = ImageAnalysisResultDetails.from_result(result_analysis)
			return result_details

		except Exception as e:
			print(e)
			return None
