import json

import pandas
import pandas as pd
import requests
from PIL import Image
from adlfs import AzureBlobFileSystem
from tqdm import tqdm

from common.storage.azure_file_storage import AzureFileStorageAdapter


def create_thumbnail(_image_id: str, _curated_df: pandas.DataFrame, _crops: pandas.DataFrame, _extant_file_names: list):
	try:
		if _image_id is None or _image_id in _extant_file_names:
			print(f'Image {_image_id} already exists, skipping')
			return None

		cropping_information = _crops.loc[_crops['id'] == _image_id]
		if cropping_information is None or len(cropping_information) == 0:
			print(f'No cropping information for {_image_id}, skipping')
			return None

		record = _curated_df.loc[_curated_df['id'] == _image_id]
		image_url = record.to_dict(orient='records')[0]['url']
		original_image = Image.open(requests.get(image_url, stream=True).raw)
		copied_image = original_image.copy()
		original_image.close()

		cropped = copied_image.crop((cropping_information['smart_crop_boundingBox.x'].values[0],
									 cropping_information['smart_crop_boundingBox.y'].values[0],
									 cropping_information['smart_crop_boundingBox.x'].values[0] +
									 cropping_information['smart_crop_boundingBox.w'].values[0],
									 cropping_information['smart_crop_boundingBox.y'].values[0] +
									 cropping_information['smart_crop_boundingBox.h'].values[0]))
		copied_image.close()

		resized = cropped.resize((512, 512), 1)
		resized.save('temp.jpg')
		file_system.upload('temp.jpg', f'data/image/thumbnail/{_image_id}.jpg', overwrite=True)
		print(f'Thumbnail created for {_image_id}')
		return None

	except Exception as e:
		print(f'Error creating thumbnail for {_image_id}: {e}')
		return None


def main():
	tqdm.pandas(desc="Progress")

	file_system: AzureBlobFileSystem = AzureFileStorageAdapter('data').get_file_storage()

	curated_df = pandas.read_parquet('data/parquet/back.parquet', filesystem=file_system, engine='pyarrow')

	accepted = curated_df.loc[curated_df["accept"] == True]

	accepted.dropna(inplace=True)

	accepted.reset_index(inplace=True, drop=True)

	print(accepted)

	captions = pd.read_parquet('data/parquet/image_captions.parquet', filesystem=file_system, engine='pyarrow')
	tags = pd.read_parquet('data/parquet/image_tags.parquet', filesystem=file_system, engine='pyarrow')
	crops = pd.read_parquet('data/parquet/image_cropping.parquet', filesystem=file_system, engine='pyarrow')

	print(captions.shape)
	print(tags.shape)
	print(crops.shape)

	current_captions = file_system.ls("data/caption")
	print(len(current_captions))

	all_data = []
	for caption_file in tqdm(current_captions, total=len(current_captions), desc='Reading caption files'):
		try:
			caption_data = json.loads(file_system.read_text(caption_file, encoding='utf-8'))
			dense_caption_result = caption_data.get('denseCaptionsResult')
			metadata = caption_data.get('metadata')
			tags_result = caption_data.get('tagsResult')
			smart_crop_result = caption_data.get('smartCropsResult')
			basic_caption = caption_data.get('captionResult')
			image_id = caption_file.split('/')[-1].split('.')[0]
			caption_data["id"] = image_id
			filtered_data = {
				"id": image_id,
				"captions": [basic_caption],
				"dense_captions": dense_caption_result['values'],
				"meta": [metadata],
				"tags": tags_result['values'],
				"smart_crop": smart_crop_result['values']
			}
			all_data.append(filtered_data)
		except Exception as e:
			print(e)
			continue

	new_captions = pandas.json_normalize(data=all_data, record_path=['dense_captions'], meta=['id'],
										 record_prefix='dense_captions_')
	new_tags = pandas.json_normalize(data=all_data, record_path=['tags'], meta=['id'], record_prefix='tags_')
	new_crops = pandas.json_normalize(data=all_data, record_path=['smart_crop'], meta=['id'],
									  record_prefix='smart_crop_')

	new_basic_captions = pandas.json_normalize(data=all_data, record_path=['captions'], meta=['id'],
											   record_prefix='captions_')
	meta = pandas.json_normalize(data=all_data, record_path=['meta'], meta=['id'], record_prefix='meta_')

	print(new_captions.shape)
	print(new_tags.shape)
	print(new_crops.shape)
	print(new_basic_captions.shape)
	print(meta.shape)

	merge_singles = pandas.merge(new_basic_captions, meta, on='id').set_index(keys=['id'], drop=False)
	merge_singles.drop_duplicates(inplace=True)
	merge_singles.reset_index(drop=True, inplace=True)
	print(merge_singles)

	merged_to_curate = pandas.merge(merge_singles, curated_df, on='id', how='outer').set_index(keys=['id'], drop=False)
	merged_to_curate.fillna(value='', inplace=True)
	print(merged_to_curate)

	merged_captions = pandas.concat([new_captions, captions])
	merged_captions.set_index(keys=['id', 'dense_captions_text', 'dense_captions_confidence'], inplace=True, drop=False)
	merged_captions.drop_duplicates(inplace=True)
	merged_captions.reset_index(drop=True, inplace=True)

	print(f'{merged_captions.shape[0] - captions.shape[0]} new rows added to captions')

	merged_captions.to_parquet('data/parquet/image_captions.parquet', filesystem=file_system, engine='pyarrow')
	print(pandas.read_parquet('data/parquet/image_captions.parquet', filesystem=file_system, engine='pyarrow'))

	merged_tags = pandas.concat([new_tags, tags])
	merged_tags.set_index(keys=['id', 'tags_name', 'tags_confidence'], inplace=True, drop=False)
	merged_tags.drop_duplicates(inplace=True)
	merged_tags.reset_index(drop=True, inplace=True)

	print(f'{merged_tags.shape[0] - tags.shape[0]} new rows added to tags')

	merged_tags.to_parquet('data/parquet/image_tags.parquet', filesystem=file_system, engine='pyarrow')
	print(pandas.read_parquet('data/parquet/image_tags.parquet', filesystem=file_system, engine='pyarrow'))

	merged_crops = pandas.concat([new_crops, crops])
	merged_crops.set_index(keys=['id'], inplace=True, drop=False)
	merged_crops.drop_duplicates(inplace=True)
	merged_crops.reset_index(drop=True, inplace=True)

	print(f'{merged_crops.shape[0] - crops.shape[0]} new rows added to crops')

	merged_crops.to_parquet('data/parquet/image_cropping.parquet', filesystem=file_system, engine='pyarrow')

	print(pandas.read_parquet('data/parquet/image_cropping.parquet', filesystem=file_system, engine='pyarrow'))

	import os

	extant = [os.path.basename(item.replace('\n', '').split('.')[0]) for item in file_system.ls('data/image/thumbnail')]
	print(extant)

	foo = curated_df.loc[curated_df['accepted'] == True]
	foo = foo.loc[foo['id'].isin(crops.id.unique())]
	foo = foo.loc[~foo['id'].isin(extant)]
	foo.dropna(inplace=True)
	foo.reset_index(inplace=True, drop=True)

	print(foo)
	print(crops)


# for primary_image_record in tqdm(foo.values, total=len(foo.values), desc='Creating thumbnails'):
# 	create_thumbnail(primary_image_record, curated_df, crops, extant)
# for primary_image_record in tqdm(curated_df.id.values, total=len(curated_df.id.values), desc='Creating thumbnails'):
# 	create_thumbnail(primary_image_record, curated_df, crops, extant)
if __name__ == '__main__':
	main()
