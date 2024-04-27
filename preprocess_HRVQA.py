# from datasets import load_dataset
from PIL import Image
from io import BytesIO
import requests
import os
import json
import uuid


def process_and_save(dataset, output_folder, subset_name):
    question_list = dataset[0]['questions']
    answer_list = dataset[1]['annotations']
    # Initialize list to hold all JSON data
    json_data_list = []
    
    subset_folder = os.path.join(output_folder, subset_name)
    if not os.path.exists(subset_folder):
        os.makedirs(subset_folder)

    assert len(question_list) == len(answer_list)
    
    for i in range(len(question_list)):
        question = question_list[i]
        answer = answer_list[i]
        
        assert question['question_id'] == answer['question_id']
        
        image_id = question['image_id']
        image_name = "dataset/HRVQA-1.0/images/" + str(image_id) + ".png"
        input_q = question['question']
        output_a = answer['multiple_choice_answer']

        # Structure for LLaVA JSON
        json_data = {
            "id": str(image_id),
            "image": image_name,
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\n" + input_q
                },
                {
                    "from": "gpt",
                    "value": output_a
                }
            ]
        }


        # Append to list
        json_data_list.append(json_data)


    # Save the JSON data list to a file
    json_output_path = os.path.join(output_folder, subset_name, 'dataset.json')
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list, json_file, indent=4)


def save_dataset(dataset_name, output_folder):
    # Load the dataset from Hugging Face
    # dataset = load_dataset(dataset_name, split=subset_name)


    # Filter for images with the specified class in 'question_type'
    # filtered_dataset = [item for item in dataset if item['question_type'] == class_name]


    # Determine the split for training and validation
    # if val_samples is not None and subset_name == 'train':
    #     train_dataset = filtered_dataset[val_samples:]
    #     val_dataset = filtered_dataset[:val_samples]
    # else:
    #     train_dataset = filtered_dataset
    #     val_dataset = []
    with open(f"dataset/{dataset_name}/jsons/train_question.json") as fp:
        train_questions = json.load(fp)
    with open(f"dataset/{dataset_name}/jsons/train_answer.json") as fp:
        train_answers = json.load(fp)
    
    with open(f"dataset/{dataset_name}/jsons/val_question.json") as fp:
        val_questions = json.load(fp)
    with open(f"dataset/{dataset_name}/jsons/val_answer.json") as fp:
        val_answers = json.load(fp)
    
    # Process and save the datasets
    for subset, data in [('train', [train_questions, train_answers]), ('validation', [val_questions, val_answers])]: 
        if data:
            process_and_save(data, output_folder, subset)

# Usage example
# output_folder = 'dataset/AQUA'
output_folder = 'dataset/HRVQA-1.0'
# class_name = 'other'
# val_samples = 300

# save_dataset('AQUA', output_folder)
save_dataset('HRVQA-1.0', output_folder)

# save_dataset('Multimodal-Fatima/OK-VQA_train', output_folder, class_name, 'train', val_samples)
# save_dataset('Multimodal-Fatima/OK-VQA_test', output_folder, class_name, 'test')
