from datasets import load_dataset
import random
import os
import json
from PIL import Image

random.seed(42)

ds = load_dataset("UCSC-VLAA/HQ-Edit", cache_dir='/data1/thkim/FederatedCL/dataset/')
print(len(ds))
print(ds.keys())
print(len(ds['train']))

task_instruction = [
    "Presented with two images and a sentence that depicts the editing request from the source image to the target image, your task is to determine if the sentence accurately describes the edit request. You must choose your answer from the Choice List. ",
    "Given two images and an edit instruction, your responsibility is to judge whether the sentence is correct instruction to transform the source image to the target image. You must choose your answer from the Choice List. ",
    "Upon receiving two images and a statement that illustrates the editng process, your job is to decide whether the sentence correctly depict the editing process from the source image to the target image. You must choose your answer from the Choice List. ",
    "Using two images and a sentence that depicts the change from the source image to the target image, your task is to evaluate whether the sentence accurately portrays the transformation between them. You must choose your answer from the Choice List. ",
    "With two images and an edit instruction describing the shift from the source image to the target image, your job is to assess whether the sentence correctly represents this change. You must choose your answer from the Choice List.",
    "Provided with two images and a statement that illustrates the transition from the source image to the target image, your role is to determine if the sentence accurately conveys the transformation. You must choose your answer from the Choice List.",
    "Given two images and a sentence explaining the modification from the source image to the target image, your task is to judge if the sentence correctly communicates this transformation. You must choose your answer from the Choice List.",
    "Using two images and a sentence that presents the change from the source image to the target image, your responsibility is to evaluate whether the sentence appropriately depicts the transformation. You must choose your answer from the Choice List.",
    "With two images and a sentence expressing the alteration from the source image to the target image at hand, your job is to decide whether the sentence correctly interprets this transformation. You must choose your answer from the Choice List.",
    "Presented with two images and a sentence that specifies the change from the source image to the target image, your duty is to assess if the sentence accurately describes the transformation. You must choose your answer from the Choice List.",
]

output_folder = 'dataset/HQ_Edit'
json_data_list_train = []
json_data_list_test = []

train_num = 10000
test_num = 2000

train_folder = os.path.join(output_folder, 'train')
test_folder = os.path.join(output_folder, 'test')
image_folder = os.path.join(output_folder, 'images')

if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(test_folder):
    os.makedirs(test_folder)
    
if not os.path.exists(image_folder):
    os.makedirs(image_folder)
    

total_indices = list(range(len(ds['train'])))
indices = random.sample(total_indices, int((train_num + test_num)))

for id, index in enumerate(indices):
    item = ds['train'][index]
    source_img = item['input_image']
    target_img = item['output_image']
    
    src_img_path = os.path.join(image_folder, f'{id}_src.png')
    tgt_img_path = os.path.join(image_folder, f'{id}_tgt.png')
    
    img = Image.open(src_img_path)
    
    images = [src_img_path, tgt_img_path]
    
    # source_img.save(src_img_path)
    # target_img.save(tgt_img_path)
    
    if id % 2 == 0:
        edit=item['edit']
        answer='True'
    else:
        edit=item['inverse_edit']
        answer='False'
    question = f'Source Image:<image> Target Image:<image> Instruction:{edit}'
    
    inst_idx = int(id*len(task_instruction)/(train_num + test_num))
    
    json_data = {
        "id": id,
        "image": images,
        "conversations": [
            {
                "from": "human",
                "value": task_instruction[inst_idx] + question + '\nChoice list:[True, False]. Your answer is:'
            },
            { 
                "from": "gpt",
                "value": answer
            }
        ]
    }
    
    if (id)%((train_num + test_num)/len(task_instruction))<1000:
        json_data_list_train.append(json_data)
    else:
        json_data_list_test.append(json_data)
    
print(len(json_data_list_train))
print(len(json_data_list_test))
        
with open(f'{train_folder}/dataset-0.json', 'w') as json_file:
    json.dump(json_data_list_train, json_file, indent=4)
with open(f'{test_folder}/dataset-0.json', 'w') as json_file:
    json.dump(json_data_list_test, json_file, indent=4)