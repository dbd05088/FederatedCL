# from datasets import load_dataset
from PIL import Image
from io import BytesIO
import requests
import os
import json
import glob
import random

random.seed(42)

captioning_prompts = [
    "Describe <image> concisely.",
    "Write a caption of <image>.",
    "Provide a brief description of <image>.",
    "Come up with a concise caption that captures the essense of <image>.",
    "Encapsulate the key information presented in <image> in a brief statement.",
    "I need a succinct caption for <image>.",
    "Please provide a pithy summary of <image> that effectively communicates its message.",
    "Can you provide a snappy caption that perfectly encapsulates the message conveyed by <image>?",
    "Please write a brief but compelling caption that grabs the reader's attention and draws them into <image>.",
    "Give a short caption that accurately conveys the main idea of <image>."
]

def process_and_save(dataset, output_folder, subset_name):
    # Define image subfolder within output folder
    
    image_subfolder = os.path.join(output_folder, 'images')


    
    


def save_dataset(dataset_name, output_folder, subset_name):
    subset_folder = os.path.join(output_folder, subset_name)
    if not os.path.exists(subset_folder):
        os.makedirs(subset_folder)
    
    img_list = glob.glob(f"dataset/SciCap/SciCap-Yes-Subfig-Img/{subset_name}/*.png")
    
    json_data_list = []
    
    for img_path in img_list:
        json_path = f"./dataset/SciCap/SciCap-Caption-All/{subset_name}/{img_path.split('/')[-1][:-4]}.json"
        with open(json_path) as fp:
            item = json.load(fp)
        
        answers = [item["1-lowercase-and-token-and-remove-figure-index"]['caption']]
        unique_answers = list(set(answers))
        formatted_answers = ", ".join(unique_answers)


        # Structure for LLaVA JSON
        json_data = {
            "id": img_path.split('/')[-1][:-4],
            "image": img_path,
            "conversations": [
                {
                    "from": "human",
                    "value": random.choice(captioning_prompts)
                },
                { 
                    "from": "gpt",
                    "value": formatted_answers
                }
            ]
        }
        json_data_list.append(json_data)
    # Save the JSON data list to a file
    json_output_path = os.path.join(output_folder, subset_name, 'dataset-0.json')
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list, json_file, indent=4)
        
    img_list = glob.glob(f"dataset/SciCap/SciCap-No-Subfig-Img/{subset_name}/*.png")
    json_data_list = []
    
    for img_path in img_list:
        json_path = f"./dataset/SciCap/SciCap-Caption-All/{subset_name}/{img_path.split('/')[-1][:-4]}.json"
        with open(json_path) as fp:
            item = json.load(fp)
        
        answers = [item["1-lowercase-and-token-and-remove-figure-index"]['caption']]
        unique_answers = list(set(answers))
        formatted_answers = ", ".join(unique_answers)


        # Structure for LLaVA JSON
        json_data = {
            "id": img_path.split('/')[-1][:-4],
            "image": img_path,
            "conversations": [
                {
                    "from": "human",
                    "value": random.choice(captioning_prompts)
                },
                { 
                    "from": "gpt",
                    "value": formatted_answers
                }
            ]
        }
        json_data_list.append(json_data)
    # Save the JSON data list to a file
    json_output_path = os.path.join(output_folder, subset_name, 'dataset-1.json')
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list, json_file, indent=4)

# Usage example
output_folder = 'dataset/SciCap'
save_dataset('SciCap', output_folder, 'test')
save_dataset('SciCap', output_folder, 'train')

