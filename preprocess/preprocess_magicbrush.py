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
    "Please give a editing Request to describe the transformation from the source image to the target image.",
    "What is the correct image edit instruction that can transfrom the source image to target image?",
    "What editing actions would you suggest to modify the source image into the target image?",
    "Please provide instructions for editing the source image to match the target image.",
    "Describe the editing process to convert the source image to the target image.",
    "What editing procedures should be implemented to transition the source image into the target image?",
    "How can the source image be edited to resemble the target image?",
    "Please give directions for the necessary edits to morph the source image into the target image.",
    "What transformations must be applied to the source image to achieve the target image?",
    "Can you describe the editing process to transform the source image into the target image?"
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

def save_datalist_with_img(datalist, output_folder, image_folder):
    json_datalist = []
    for i, item in enumerate(datalist):
        source_img = item['source_img']
        target_img = item['target_img']
        
        img_id = item['img_id']
        turn_idx = item['turn_index']
        
        src_img_path = os.path.join(image_folder, f'{img_id}_{turn_idx}_src.png')
        tgt_img_path = os.path.join(image_folder, f'{img_id}_{turn_idx}_tgt.png')
        
        source_img.save(src_img_path)
        target_img.save(tgt_img_path)
        
        images = [src_img_path, tgt_img_path]
        
        answer = item['instruction']
        
        question = f'\nSource Image:<image> Target Image:<image> Instruction:'
        
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
    
    with open(f'{output_folder}/dataset-0.json', 'w') as json_file:
        json.dump(json_datalist, json_file, indent=4)


save_datalist_with_img(ds['train'], train_folder, image_folder)
save_datalist_with_img(ds['dev'], test_folder, image_folder)
