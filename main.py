import json
import time

import pandas
from adlfs import AzureBlobFileSystem

from common.captioning.azure_descriptions import AzureCaption
from common.functions import functions
from common.functions.functions import Functions
from common.schemas.pyarrow_schema import schema
from common.storage.azure_file_storage import AzureFileStorageAdapter


def run(_filtered_model, _current_captions):
	total = len(_filtered_model)
	i = 0
	j = 0

	for record in _filtered_model.to_dict(orient='records'):
		try:
			image_id = record['id']
			path = record['path']
			out_path = f"data/caption/{image_id}.json"
			if out_path in _current_captions:
				continue
			try:
				print(f"Captioning {i} of {total}")
				_file_system: AzureBlobFileSystem = AzureFileStorageAdapter('data').get_file_storage()
				remote_path: str = _file_system.url(path)
				caption_reference = AzureCaption(_file_system)
				caption_output = caption_reference.image_analysis(remote_path)

				if caption_output is None:
					print(f"Error In Output is empty for {image_id}")
					continue

				time.sleep(5)
				json_result = caption_output.json_result

				if json_result is None:
					print(f"Error In Json Result is empty for {image_id}")
					continue

				if json.loads(json_result).get('error'):
					print(f"Error In Json Result with: {json_result} for {image_id}")
					continue

				print(f"Writing Captioning For {image_id}")

				with _file_system.open(out_path, 'w', encoding='utf-8') as handle:
					handle.write(json_result)
					j += 1

				if _file_system.size(out_path) == 0:
					print(f"Caption File {out_path} is empty, deleting...")
					_file_system.rm(out_path)
					continue

				else:
					print(f"Captioning For {image_id} Complete new: {j}")
					print(f"Captioning {i} of {total}")
					continue

			except Exception as ex:
				print(f"Error in handle_captioning with exception {ex}")
				print(ex)
				continue

		finally:
			i += 1


if __name__ == '__main__':
	functions: Functions = Functions()

	file_system: AzureBlobFileSystem = AzureFileStorageAdapter('data').get_file_storage()

	print("Loading Current Caption Files...")
	current = file_system.ls("data/caption")
	current_captions = [item.replace('\n', '') for item in current]
	print(f"Total Number Of Caption Files Prior to Removal - {len(current_captions)}")

	print("Removing Empty Caption Files...")
	for caption in current_captions:
		if file_system.size(caption) == 0:
			print(f"Caption File {caption} is empty, deleting...")
			file_system.rm(caption)

	print("Loading Current Caption Files POST Clean Up...")
	current = file_system.ls("data/caption")
	current_captions = [item.replace('\n', '') for item in current]
	print(f"Total Number Of Caption Files - {len(current_captions)}")

	sources = [
		{"name": "CityDiffusion", "data": ["CityPorn"]},
		{"name": "NatureDiffusion", "data": ["EarthPorn"]},
		{"name": "CosmicDiffusion", "data": ["spaceporn"]},
		{"name": "ITAPDiffusion", "data": ["itookapicture"]},
		{"name": "MemeDiffusion", "data": ["memes"]},
		{"name": "TTTDiffusion", "data": ["trippinthroughtime"]},
		{"name": "WallStreetDiffusion", "data": ["wallstreetbets"]},
		{"name": "SexyDiffusion", "data": ["selfies", "Amicute", "amihot", "AmIhotAF", "HotGirlNextDoor", "sexygirls", "PrettyGirls", "gentlemanboners", "hotofficegirls", "tightdresses", "DLAH", "cougars_and_milfs_sfw"]},
		{"name": "FatSquirrelDiffusion", "data": ["fatsquirrelhate"]},
		{"name": "CelebrityDiffusion", "data": ["celebrities"]},
		{"name": "OldLadyDiffusion", "data": ["oldladiesbakingpies"]},
		{"name": "SWFPetite", "data": ["sfwpetite"]},
		{"name": "RedHeadDiffusion", "data": ["SFWRedheads"]},
		{"name": "NextDoorGirlsDiffusion", "data": ["SFWNextDoorGirls"]},
		{"name": "SexyAsianDiffusion", "data": ["realasians", "KoreanHotties", "prettyasiangirls", "AsianOfficeLady", "AsianInvasion"]},
		{"name": "MildlyPenisDiffusion", "data": ["mildlypenis"]},
		{"name": "CandleDiffusion", "data": ["bathandbodyworks"]},
	]
	sources_df = pandas.DataFrame.from_records(sources)

	curated_data = pandas.read_parquet("data/parquet/back.parquet", engine="pyarrow", filesystem=file_system)

	curated_data.set_index("id", inplace=True, drop=False)

	filtered = curated_data.loc[curated_data["accept"] == True, schema.names]

	filtered.dropna(inplace=True)

	filtered.reset_index(inplace=True, drop=True)

	filtered['model'] = filtered.apply(lambda x: functions.add_source(x, sources), axis=1)

	filtered_model = filtered.loc[filtered['model'] != "", schema.names]

	filtered_model.dropna(inplace=True)

	filtered_model.reset_index(inplace=True, drop=True)

	print("Starting Captioning...")

	run(filtered_model, current_captions)


	print(f"Total Number Of Caption Files - {len(file_system.ls('data/caption'))}")

	print("0-2 Azure Image Analysis Process Complete - Shutting Down")
