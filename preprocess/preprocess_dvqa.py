from datasets import load_dataset
import os
import random
import numpy as np
import json
import re  # For digit and number matching

random.seed(42)
np.random.seed(42)

# ds = load_dataset("lmms-lab/LLaVA-OneVision-Data", "dvqa(cauldron,llava_format)", cache_dir='/data1/thkim/FederatedCL/dataset/')
# print(len(ds))
# print(ds.keys())
# print(len(ds['train']))

output_folder = 'dataset/dvqa/'
# json_data_list = []

# num_per_samples = 2
# train_num = 20000
# test_num = 4000

# train_folder = os.path.join(output_folder, 'train')
# test_folder = os.path.join(output_folder, 'test')
# image_folder = os.path.join(output_folder, 'images')

# if not os.path.exists(train_folder):
#     os.makedirs(train_folder)
# if not os.path.exists(test_folder):
#     os.makedirs(test_folder)

# if not os.path.exists(image_folder):
#     os.makedirs(image_folder)

# total_indices = list(range(len(ds['train'])))
# indices = random.sample(total_indices, int((train_num + test_num)))

# # Categories for the answers
# yes_no_data = []
# digit_data = []
# number_word_data = []
# rest_data = []

# # Define regex patterns for digits (including negative numbers) and number words
# digit_pattern = re.compile(r'^-?\d+$')  # For digits with optional minus sign
# number_word_list = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand"]
# number_word_pattern = re.compile(r'\b(?:' + '|'.join(number_word_list) + r')\b', re.IGNORECASE)

# for index in indices:
#     item = ds['train'][index]
#     image = item['image']
#     image_path = os.path.join(image_folder, item['id'].split('/')[-1])

#     # Uncomment this to save the image
#     image.save(image_path)

#     conv_indices = random.sample(list(range(int(len(item['conversations'])/2))), num_per_samples)
#     for i in range(num_per_samples):
#         new_item = {}
#         new_item['id'] = item['id'].split('/')[-1] + f'_{i}'
#         new_item['image'] = image_path
#         new_item['conversations'] = item['conversations'][conv_indices[i]*2:(conv_indices[i]+1)*2]
#         new_item['conversations'][1]['value'] = new_item['conversations'][1]['value'][:-1]
        
#         answer = new_item['conversations'][1]['value'].strip().lower()

#         # Categorize the sample based on the answer
#         if "yes" in answer or "no" in answer:
#             yes_no_data.append(new_item)
#         elif digit_pattern.match(answer):  # Match digits and negative digits
#             digit_data.append(new_item)
#         elif number_word_pattern.search(answer):
#             number_word_data.append(new_item)
#         else:
#             rest_data.append(new_item)

# # Function to save data into JSON files
# def save_json_data(data, folder, file_name):
#     json_output_path = os.path.join(folder, file_name)
#     with open(json_output_path, 'w') as json_file:
#         json.dump(data, json_file, indent=4)
#     print(f"Saved {len(data)} samples to {json_output_path}")

# # Split data into train/test and save for each category
# json_data_train = yes_no_data[:train_num] + digit_data[:train_num] + number_word_data[:train_num] + rest_data[:train_num]
# json_data_test = yes_no_data[train_num:] + digit_data[train_num:] + number_word_data[train_num:] + rest_data[train_num:]

# save_json_data(yes_no_data, train_folder, 'dataset-yesno.json')
# save_json_data(digit_data, train_folder, 'dataset-digit.json')
# save_json_data(number_word_data, train_folder, 'dataset-numberword.json')
# save_json_data(rest_data, train_folder, 'dataset-rest.json')

# # Test data
# for item in json_data_test:
#     if 'single word or phrase.' not in item['conversations'][0]['value']:
#         item['conversations'][0]['value'] += '\nAnswer the question using a single word or phrase.'

# save_json_data(json_data_train, train_folder, 'dataset-train.json')
# save_json_data(json_data_test, test_folder, 'dataset-test.json')

with open(os.path.join(output_folder, 'train/dataset-digit.json')) as fp:
    digit_data = json.load(fp)

json_data_list = []

for item in digit_data:
    answer = int(item['conversations'][1]['value'])
    
    answer_idx = random.randint(0, 3)
    if answer > 100:
        gap = int(answer/4)
    if answer % 100 != 0:
        gap = random.choice([1,2,5])
    else:
        gap = random.choice([1,2,5,10,20,100])
    
    choice_list = []
    for i in range(4):
        # if i == answer_idx:
        #     choice_list.append(answer)  # Add correct answer
        # else:
        #     # Add incorrect answers by subtracting or adding the gap
        wrong_choice = answer + (i - answer_idx) * gap
        choice_list.append(str(wrong_choice))
    item['conversations'][0]['value'] = "<image>\n" + item['conversations'][0]['value'] + f"\nChoice List: [{', '.join(choice_list)}]. Your answer is:"
    json_data_list.append(item)
ratio = int(len(json_data_list)*0.8)
json_data_list_train = json_data_list[:ratio]
json_data_list_test = json_data_list[ratio:]

print(len(json_data_list_train))
print(len(json_data_list_test))


if len(json_data_list_train) > 10000:
    json_data_list_train_digit = np.random.choice(json_data_list_train, size=5000, replace=False).tolist()
    json_data_list_train = np.random.choice(json_data_list_train, size=10000, replace=False).tolist()
if len(json_data_list_test) > 2000:
    json_data_list_test_digit = np.random.choice(json_data_list_test, size=1000, replace=False).tolist()
    json_data_list_test = np.random.choice(json_data_list_test, size=2000, replace=False).tolist()
    
print(len(json_data_list_train))
print(len(json_data_list_test))    

json_output_path = os.path.join(output_folder, 'train/dataset-1.json')
with open(json_output_path, 'w') as json_file:
    json.dump(json_data_list_train, json_file, indent=4)
    
json_output_path = os.path.join(output_folder, 'test/dataset-1.json')
with open(json_output_path, 'w') as json_file:
    json.dump(json_data_list_test, json_file, indent=4)

with open(os.path.join(output_folder, 'train/dataset-rest.json')) as fp:
    digit_data = json.load(fp)
    
import jsonlines
choices = []
with jsonlines.open(f"{output_folder}/choices.jsonl") as f:
    for line in f.iter():
        choices.append(line)

json_data_list = []

for idx, item in enumerate(digit_data):
    assert item['id'] == choices[idx]['id']
    choice_list = choices[idx]['content']
    
    item['conversations'][0]['value'] = "<image>\n" +  item['conversations'][0]['value'] + "\nChoice List: " + choice_list[12:]+ ". Your answer is:"
    json_data_list.append(item)
ratio = int(len(json_data_list)*0.8)
json_data_list_train = json_data_list[:ratio]
json_data_list_test = json_data_list[ratio:]

print(len(json_data_list_train))
print(len(json_data_list_test))


if len(json_data_list_train) > 10000:
    json_data_list_train_choices = np.random.choice(json_data_list_train, size=5000, replace=False).tolist()
    json_data_list_train = np.random.choice(json_data_list_train, size=10000, replace=False).tolist()
if len(json_data_list_test) > 2000:
    json_data_list_test_choices = np.random.choice(json_data_list_test, size=1000, replace=False).tolist()
    json_data_list_test = np.random.choice(json_data_list_test, size=2000, replace=False).tolist()
print(len(json_data_list_train))
print(len(json_data_list_test))    

json_output_path = os.path.join(output_folder, 'train/dataset-2.json')
with open(json_output_path, 'w') as json_file:
    json.dump(json_data_list_train, json_file, indent=4)
    
json_output_path = os.path.join(output_folder, 'test/dataset-2.json')
with open(json_output_path, 'w') as json_file:
    json.dump(json_data_list_test, json_file, indent=4)
    

json_data_list_train = json_data_list_train_digit + json_data_list_train_choices
json_data_list_test = json_data_list_test_digit + json_data_list_test_choices

print(len(json_data_list_train))
print(len(json_data_list_test))    

json_output_path = os.path.join(output_folder, 'train/dataset-0.json')
with open(json_output_path, 'w') as json_file:
    json.dump(json_data_list_train, json_file, indent=4)
    
json_output_path = os.path.join(output_folder, 'test/dataset-0.json')
with open(json_output_path, 'w') as json_file:
    json.dump(json_data_list_test, json_file, indent=4)