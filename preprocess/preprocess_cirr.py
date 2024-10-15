# import random
# import os
# import json
# from PIL import Image
# import requests

# random.seed(42)

# # Load the split train and test datasets from JSON files
# with open('dataset/CIRR/cirr/captions/cap.rc2.train.json', 'r') as f:
#     train_dataset = json.load(f)

# with open('dataset/CIRR/cirr/captions/cap.rc2.val.json', 'r') as f:
#     test_dataset = json.load(f)

# random.shuffle(train_dataset)
# random.shuffle(test_dataset)

# identifier_to_url = {}
# with open('dataset/CIRR/cirr/train.jsonl', 'r') as f:
#     for line in f:
#         data = json.loads(line.strip())
#         identifier = data['identifier'][:-2]
#         left_url = data['left_url']
#         right_url = data['right_url']
#         # Map identifier to corresponding URLs
#         identifier_to_url[identifier] = {"left_url": left_url, "right_url": right_url}

# with open('dataset/CIRR/cirr/dev.jsonl', 'r') as f:
#     for line in f:
#         data = json.loads(line.strip())
#         identifier = data['identifier'][:-2]
#         left_url = data['left_url']
#         right_url = data['right_url']
#         # Map identifier to corresponding URLs
#         identifier_to_url[identifier] = {"left_url": left_url, "right_url": right_url}

# # Function to download an image from a URL and save it
# def download_image(url, save_path):
#     try:
#         response = requests.get(url, timeout=30)
#         if response.status_code == 200:
#             with open(save_path, 'wb') as f:
#                 f.write(response.content)
#             print(f'Successfully downloaded {save_path}')
#             return True
#         else:
#             print(f'Failed to download {save_path}: {response.status_code}')
#     except Exception as e:
#         print(f'Error downloading {save_path}: {e}')
#     return False

# # Folder structure
# output_folder = 'dataset/CIRR'
# train_folder = os.path.join(output_folder, 'train')
# test_folder = os.path.join(output_folder, 'test')
# image_folder = os.path.join(output_folder, 'images')

# if not os.path.exists(train_folder):
#     os.makedirs(train_folder)
# if not os.path.exists(test_folder):
#     os.makedirs(test_folder)
# if not os.path.exists(image_folder):
#     os.makedirs(image_folder)

# # Instructions for task prompts
# task_instruction = [
#     "Presented with a reference and a target image along with a caption, your task is to determine whether the caption correctly describes the transition from the reference image to the target image. You must choose your answer from the Choice List. ",
#     "Given a reference and target image, along with a descriptive caption, determine if the caption properly explains the change from the reference image to the target image. Choose from the Choice List. ",
#     "Provided with a reference and a target image, along with a caption, your task is to assess whether the caption accurately reflects the transformation from reference to target. Choose your answer from the Choice List. ",
#     "You are given a reference image and a target image with a caption explaining the transition. Your job is to evaluate if the caption correctly describes this change. Select your answer from the Choice List. ",
#     "Your task is to decide if the provided caption appropriately explains the transition from the reference image to the target image. Select your answer from the Choice List. ",
# ]

# # Function to process a dataset (train or test)
# def process_dataset(dataset, folder_name, subset, max_size):
#     json_data_list = []

#     for id, item in enumerate(dataset):
#         reference_img = item['reference']
#         target_img = item['target_hard']
#         caption = item['caption']
#         img_set_members = item['img_set']['members']
        
#         # True sample: caption correctly describes reference to target_hard
#         if id % 2 == 0:
#             answer = 'True'
#         else:
#             # False sample: Either use a wrong target from img_set or swap reference and target
#             wrong_target_img = random.choice([img for img in img_set_members if img != target_img])
#             if random.random() > 0.5:
#                 # Swap reference and target
#                 reference_img, target_img = target_img, reference_img
#             else:
#                 # Use a random wrong target image
#                 target_img = wrong_target_img
#             answer = 'False'
#         question = f'Reference Image:<image> Target Image:<image> Caption: {caption}'
#         inst_idx = int(id * len(task_instruction) / len(dataset))
        
#         # download image
#         reference_identifier = reference_img.rsplit('-', 1)[0]
#         target_identifier = target_img.rsplit('-', 1)[0]
#         # Download images using URLs from the identifier mapping
        
#         if reference_identifier in identifier_to_url:
#             if reference_img[-4:] == 'img0':
#                 url = identifier_to_url[reference_identifier]['left_url']
#             elif reference_img[-4:] == 'img1':
#                 url = identifier_to_url[reference_identifier]['right_url']
#             ref_img_path = os.path.join(image_folder, f'{subset}_{id}_ref.png')
#             if not download_image(url, ref_img_path):
#                 continue
#         else:
#             print(f"Reference identifier {reference_identifier} not found in URL mapping.")
#             continue

#         if target_identifier in identifier_to_url:
#             if target_img[-4:] == 'img0':
#                 url = identifier_to_url[target_identifier]['left_url']
#             elif target_img[-4:] == 'img1':
#                 url = identifier_to_url[target_identifier]['right_url']
#             tgt_img_path = os.path.join(image_folder, f'{subset}_{id}_tgt.png')
#             if not download_image(url, tgt_img_path):
#                 continue
#         else:
#             print(f"Target identifier {target_identifier} not found in URL mapping.")
#             continue
#         json_data = {
#             "id": id,
#             "image": [ref_img_path, tgt_img_path],
#             "conversations": [
#                 {
#                     "from": "human",
#                     "value": task_instruction[inst_idx] + question + '\nChoice list:[True, False]. Your answer is:'
#                 },
#                 {
#                     "from": "gpt",
#                     "value": answer
#                 }
#             ]
#         }
        
#         json_data_list.append(json_data)
#         print(len(json_data_list))
#         if len(json_data_list) >= max_size:
#             break

#     # Save JSON file for the processed dataset
#     with open(f'{folder_name}/dataset-0.json', 'w') as json_file:
#         json.dump(json_data_list, json_file, indent=4)

#     print(f'Processed {len(json_data_list)} samples in {folder_name}')

# # Process train and test datasets
# process_dataset(train_dataset, train_folder, subset='train', max_size=10000)
# process_dataset(test_dataset, test_folder, subset='test', max_size=2000)

import json
from PIL import Image
import numpy as np
Image.MAX_IMAGE_PIXELS = None

with open('dataset/CIRR/train/dataset-1.json', 'r') as fp:
    datalist = json.load(fp)

new_datalist = []
for item in datalist:
    valid = True
    try:
        for img_path in item['image']:
            image = Image.open(img_path).convert('RGB')
            w, h = image.size
            if w*h == 1:
                valid=False
            # else:
            #     image.save(img_path)
        if not valid:
            continue
    except:
        continue
    new_datalist.append(item)
print(len(new_datalist))
with open('dataset/CIRR/train/dataset-1.json', 'w') as fp:
    json.dump(new_datalist, fp, indent=4)
    

# with open('dataset/CIRR/test/dataset-1.json', 'r') as fp:
#     datalist = json.load(fp)

# new_datalist = []
# for item in datalist:
#     valid = True
#     try:
#         for img_path in item['image']:
#             image = Image.open(img_path).convert('RGB')
#             if np.array(image).shape[-1] == 0:
#                 valid = False
#             # else:
#             #     image.save(img_path)
#         if not valid:
#             continue
#     except:
#         continue
#     new_datalist.append(item)

# print(len(new_datalist))
# with open('dataset/CIRR/test/dataset-1.json', 'w') as fp:
#     json.dump(new_datalist, fp, indent=4)