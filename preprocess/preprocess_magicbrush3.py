from datasets import load_dataset
import random
import os
import json
from PIL import Image

random.seed(42)

ds = load_dataset("osunlp/MagicBrush")
print(len(ds))
print(ds.keys())
print(len(ds['train']))
print(len(ds['dev']))

breakpoint()

task_instruction = [
    "Presented with a source image, a target image, and an edit instruction, your task is to determine if the instruction accurately describes the transformation from source to target. You must choose your answer from the Choice List. ", 
    "Given a source and target image along with an edit instruction, your responsibility is to judge whether the instruction correctly details the change from source to target. You must choose your answer from the Choice List. ",
    "Upon receiving a source image, a target image, and an edit instruction, your job is to decide if the instruction correctly characterizes the transformation from source to target. You must choose your answer from the Choice List. ",
    "Using a source image, a target image, and an edit instruction, your task is to evaluate if the instruction accurately portrays the change from source to target. You must choose your answer from the Choice List. ",
    "With a source image, a target image, and an edit instruction at your disposal, your job is to assess whether the instruction correctly represents the transformation from source to target. You must choose your answer from the Choice List. ",
    "Provided with a source image, a target image, and an edit instruction, your role is to determine if the instruction accurately conveys the transformation from source to target. You must choose your answer from the Choice List. ",
    "Given a source image, a target image, and an edit instruction explaining their transformation, your task is to judge if the instruction correctly communicates the change from source to target. You must choose your answer from the Choice List. ",
    "Using a source image, a target image, and an edit instruction that presents their modification, your responsibility is to evaluate if the instruction appropriately depicts the transformation from source to target. You must choose your answer from the Choice List. ",
    "With a source image, a target image, and an edit instruction expressing their modification at hand, your job is to decide if the instruction correctly interprets the transformation from source to target. You must choose your answer from the Choice List. ",
    "Presented with a source image, a target image, and an edit instruction that specifies their transformation, your duty is to assess if the instruction accurately describes the change from source to target. You must choose your answer from the Choice List. ",
]

output_folder = 'dataset/MagicBrush'

train_folder = os.path.join(output_folder, 'train')
test_folder = os.path.join(output_folder, 'test')
image_folder = os.path.join(output_folder, 'images')

if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(test_folder):
    os.makedirs(test_folder)
    
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

def save_datalist(datalist, output_folder, image_folder):
    json_datalist = []
    count1 = 0
    count2 = 0
    for i, item in enumerate(datalist):
        img_id = item['img_id']
        turn_idx = item['turn_index']
        
        src_img_path = os.path.join(image_folder, f'{img_id}_{turn_idx}_src.png')
        tgt_img_path = os.path.join(image_folder, f'{img_id}_{turn_idx}_tgt.png')
        
        if i % 2 == 0:
            images = [src_img_path, tgt_img_path]
            instruction = item['instruction']
            answer = 'True'
        else:
            if turn_idx > 1:
                images = [src_img_path, tgt_img_path]
                instruction = datalist[i - 1]['instruction']
                count1 += 1
            else:
                images = [tgt_img_path, src_img_path]
                instruction = item['instruction']
                count2 += 1
            
            answer = 'False'
                
        question = f'\nSource Image:<image> Target Image:<image> Instruction:{instruction}\nChoice list:[True, False].'
        
        
        inst_idx = i % len(task_instruction)
        
        json_data = {
            "id": f'{img_id}_{turn_idx}',
            "image": images,
            "conversations": [
                {
                    "from": "human",
                    "value": task_instruction[inst_idx] + question
                },
                { 
                    "from": "gpt",
                    "value": answer
                }
            ]
        }
        json_datalist.append(json_data)
    
    with open(f'{output_folder}/dataset-2.json', 'w') as json_file:
        json.dump(json_datalist, json_file, indent=4)
    print(count1, count2)


save_datalist(ds['train'], train_folder, image_folder)
save_datalist(ds['dev'], test_folder, image_folder)
