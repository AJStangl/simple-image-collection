import logging
import os
from ctypes import Union

from adlfs import AzureBlobFileSystem
from azure.ai.vision import VisionServiceOptions, VisionSource, ImageAnalysisOptions, ImageAnalysisFeature, \
	ImageAnalyzer, ImageAnalysisResultReason, ImageAnalysisResultDetails, ImageAnalysisErrorDetails
from azure.cognitiveservices.vision.computervision import ComputerVisionClientConfiguration, ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import ImageDescription

logging.basicConfig(level=logging.DEBUG)
logging.getLogger(__name__)


class AzureCaption(object):
	def __init__(self, file_system_reference):
		self.__subscription_key: str = os.environ["AZURE_VISION_KEY"]
		self.__endpoint: str = os.environ["AZURE_VISION_ENDPOINT"]
		self.__credentials: ComputerVisionClientConfiguration = ComputerVisionClientConfiguration(endpoint=self.__endpoint, credentials=self.__subscription_key)
		self.__vision_client: ComputerVisionClient = ComputerVisionClient(endpoint=self.__endpoint, credentials=self.__credentials)
		self.__file_system_reference: AzureBlobFileSystem = file_system_reference

	def generate_ai_tags(self, image_path: str) -> Union([ImageDescription, None]):
		try:
			describe_response: ImageDescription = self.__vision_client.describe_image(
				self.__file_system_reference.url(image_path), max_candidates=10, raw=True)
			return describe_response
		except Exception as e:
			logging.error(e)
			return None

	def image_analysis_sample_analyze(self, image_path: str) -> Union(ImageAnalysisResultDetails, None):
		"""
		Analyze image from file, all features, synchronous (blocking)
		"""

		service_options: VisionServiceOptions = VisionServiceOptions(self.__endpoint, self.__subscription_key)
		vision_source: VisionSource = VisionSource(filename=image_path)
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

		print()
		print(" Please wait for image analysis results...")
		print()

		result = image_analyzer.analyze()

		# Checks result.
		if result.reason == ImageAnalysisResultReason.ANALYZED:
			print(" Image height: {}".format(result.image_height))
			print(" Image width: {}".format(result.image_width))
			print(" Model version: {}".format(result.model_version))

			if result.caption is not None:
				print(" Caption:")
				print("   '{}', Confidence {:.4f}".format(result.caption.content, result.caption.confidence))

			if result.dense_captions is not None:
				print(" Dense Captions:")
				for caption in result.dense_captions:
					print("   '{}', {}, Confidence: {:.4f}".format(caption.content, caption.bounding_box,
																   caption.confidence))

			if result.objects is not None:
				print(" Objects:")
				for object in result.objects:
					print("   '{}', {}, Confidence: {:.4f}".format(object.name, object.bounding_box,
																   object.confidence))

			if result.tags is not None:
				print(" Tags:")
				for tag in result.tags:
					print("   '{}', Confidence {:.4f}".format(tag.name, tag.confidence))

			if result.people is not None:
				print(" People:")
				for person in result.people:
					print("   {}, Confidence {:.4f}".format(person.bounding_box, person.confidence))

			if result.crop_suggestions is not None:
				print(" Crop Suggestions:")
				for crop_suggestion in result.crop_suggestions:
					print("   Aspect ratio {}: Crop suggestion {}"
						  .format(crop_suggestion.aspect_ratio, crop_suggestion.bounding_box))

			if result.text is not None:
				print(" Text:")
				for line in result.text.lines:
					points_string = "{" + ", ".join([str(int(point)) for point in line.bounding_polygon]) + "}"
					print("   Line: '{}', Bounding polygon {}".format(line.content, points_string))
					for word in line.words:
						points_string = "{" + ", ".join([str(int(point)) for point in word.bounding_polygon]) + "}"
						print("     Word: '{}', Bounding polygon {}, Confidence {:.4f}"
							  .format(word.content, points_string, word.confidence))

			result_details: ImageAnalysisResultDetails = ImageAnalysisResultDetails.from_result(result)
			return result_details

		else:

			error_details = ImageAnalysisErrorDetails.from_result(result)
			print(" Analysis failed.")
			print("   Error reason: {}".format(error_details.reason))
			print("   Error code: {}".format(error_details.error_code))
			print("   Error message: {}".format(error_details.message))
			print(" Did you set the computer vision endpoint and key?")
			return None

