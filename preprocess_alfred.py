from PIL import Image
from io import BytesIO
import requests
import os
import json
import glob
import shutil
import numpy as np

np.random.seed(42)

# Function to categorize tasks
def categorize_sample(sample):
    human_text = sample['conversations'][0]['value'].lower()
    
    # Category 1: Kitchen-related tasks
    kitchen_keywords = ['microwave', 'sink', 'fridge', 'spoon', 'knife', 'plate', 'food', 'apple', 'tomato', 'bread', 'counter', 'table', 'potato']
    if any(keyword in human_text for keyword in kitchen_keywords):
        return "kitchen"

    # Category 2: Living Room/Bedroom tasks
    living_room_keywords = ['remote', 'couch', 'tv', 'phone', 'desk', 'shelf', 'keys', 'book', 'cabinet', 'chair']
    if any(keyword in human_text for keyword in living_room_keywords):
        return "living_room_bedroom"

    # Category 3: Bathroom/Utility tasks
    bathroom_keywords = ['towel', 'sink', 'bathroom', 'toilet', 'soap', 'plunger']
    if any(keyword in human_text for keyword in bathroom_keywords):
        return "bathroom_utility"

    # Category 4: Miscellaneous Object Manipulation tasks
    return "miscellaneous"

dir = 'dataset/ALFRED'
with open(dir+'/full/full.json', 'r') as fp:
    full_data = json.load(fp)

meta_data = full_data['metadata']
full_data = full_data['data']

total_len = len(full_data)

train_test_ratio = 0.2

idx_list = list(range(total_len))
test_idx = list(range(int(total_len*0.8), total_len))

subset_folder = os.path.join(dir, 'train')
if not os.path.exists(subset_folder):
    os.makedirs(subset_folder)
    
subset_folder = os.path.join(dir, 'test')
if not os.path.exists(subset_folder):
    os.makedirs(subset_folder)


# Split dataset into categories
categorized_data = {
    "kitchen": [],
    "living_room_bedroom": [],
    "bathroom_utility": [],
    "miscellaneous": []
}

for idx in range(total_len):
    item = full_data[idx]
    new_item = {}
    new_item['id'] = item['sample_id']
    new_item['image'] = [os.path.join(dir, 'full/images', img) for img in item['task_instance']['images_path']]
    
    if len(new_item['image']) > 7:
        breakpoint()
    
    question = item['task_instance']['context']
    for i in range(len(new_item['image'])):
        rmv_i = '{image#%d}'% (i+1)
        rmv_t = '{table#%d}'% (i+1)
        question = question.replace(rmv_i, '<image>')
        question = question.replace(rmv_t, '<image>')
    
    new_item['conversations'] = [
        {
            "from": "human",
            "value": meta_data['task_instruction'][item['task_instruction_id']] + question
        },
        {
            "from": "gpt",
            "value": item['response']
        }
    ]
    category = categorize_sample(new_item)
    categorized_data[category].append(new_item)

print(len(categorized_data['kitchen']))
print(len(categorized_data['living_room_bedroom']))
print(len(categorized_data['bathroom_utility']))
print(len(categorized_data['miscellaneous']))

json_data_train_1 = categorized_data['kitchen'][:int(len(categorized_data['kitchen'])*0.8)]
json_data_test_1 = categorized_data['kitchen'][int(len(categorized_data['kitchen'])*0.8):]

print(len(json_data_train_1))
print(len(json_data_test_1))

if len(json_data_train_1) > 10000:
    json_data_train_1 = np.random.choice(json_data_train_1, size=10000, replace=False).tolist()
if len(json_data_test_1) > 2000:
    json_data_test_1 = np.random.choice(json_data_test_1, size=2000, replace=False).tolist()

print(len(json_data_train_1))
print(len(json_data_test_1))

with open(f'{dir}/train/dataset-1.json', 'w') as json_file:
    json.dump(json_data_train_1, json_file, indent=4)
with open(f'{dir}/test/dataset-1.json', 'w') as json_file:
    json.dump(json_data_test_1, json_file, indent=4)


json_data_train_2 = [
    categorized_data['living_room_bedroom'][:int(len(categorized_data['living_room_bedroom'])*0.8)],
    categorized_data['bathroom_utility'][:int(len(categorized_data['bathroom_utility'])*0.8)],
    categorized_data['miscellaneous'][:int(len(categorized_data['miscellaneous'])*0.8)]
]

json_data_test_2 = [
    categorized_data['living_room_bedroom'][int(len(categorized_data['living_room_bedroom'])*0.8):],
    categorized_data['bathroom_utility'][int(len(categorized_data['bathroom_utility'])*0.8):],
    categorized_data['miscellaneous'][int(len(categorized_data['miscellaneous'])*0.8):],
]

train_len_2 = len(json_data_train_2[0]) + len(json_data_train_2[1]) + len(json_data_train_2[2])
test_len_2 = len(json_data_test_2[0]) + len(json_data_test_2[1]) + len(json_data_test_2[2])

print(train_len_2)
print(test_len_2)

if train_len_2 > 10000:
    json_data_train_2 = np.random.choice(json_data_train_2[0], size=3737, replace=False).tolist() + np.random.choice(json_data_train_2[1], size=2527, replace=False).tolist() + np.random.choice(json_data_train_2[2], size=3736, replace=False).tolist()
if test_len_2 > 2000:
    json_data_test_2 = np.random.choice(json_data_test_2[0], size=686, replace=False).tolist() + np.random.choice(json_data_test_2[1], size=632, replace=False).tolist() + np.random.choice(json_data_test_2[2], size=682, replace=False).tolist()

print(len(json_data_train_2))
print(len(json_data_test_2))

with open(f'{dir}/train/dataset-2.json', 'w') as json_file:
    json.dump(json_data_train_2, json_file, indent=4)
with open(f'{dir}/test/dataset-2.json', 'w') as json_file:
    json.dump(json_data_test_2, json_file, indent=4)