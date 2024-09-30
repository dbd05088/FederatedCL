# from datasets import load_dataset
from PIL import Image
from io import BytesIO
import requests
import os
import json
import uuid
import shutil


def process_and_save(dataset_dir, output_folder, subset_name):
    if subset_name == 'train':
        img_folder = os.path.join(output_folder, dataset_dir, 'Train_images')
        qa_list_file = open(os.path.join(output_folder, dataset_dir,"All_QA_Pairs_train.txt"),'r')
    elif subset_name == 'validation':
        img_folder = os.path.join(output_folder, dataset_dir,'Val_images')
        qa_list_file = open(os.path.join(output_folder, dataset_dir,"All_QA_Pairs_val.txt"),'r')
    qa_list = qa_list_file.readlines()
    qa_list_file.close()
    
    # Initialize list to hold all JSON data
    json_data_list = []

    # move img files
    # if subset_name == 'train':
    #     img_file_list = open(os.path.join(output_folder, dataset_dir,'train_ImageIDs.txt'))
    # elif subset_name == 'validation':
    #     img_file_list = open(os.path.join(output_folder, dataset_dir,'val_ImageIDs.txt'))
    # img_files = img_file_list.readlines()
    # img_file_list.close()
    
    # for i in range(len(img_files)):
    #     imgname = img_files[i][:-1] if i < len(img_files) - 1 else img_files[i]
    #     cur_path = os.path.join(img_folder, imgname + '.jpg')
    #     new_path = os.path.join(output_folder,'images',imgname+'.jpg')
    #     shutil.copyfile(cur_path, new_path)
    
    for i in range(len(qa_list)):
        qa_line = qa_list[i]
        qas = qa_line.split('|')
        
        image_id = qas[0]
        image_name = "dataset/VQA-MED/images/" + image_id + ".jpg"
        input_q = qas[1]
        output_a = qas[2][:-1] if i < len(qa_list) - 1 else qas[2]

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

    train_folder = 'ImageClef-2019-VQA-Med-Training'
    validation_folder = 'ImageClef-2019-VQA-Med-Validation'
    
    # Process and save the datasets
    for subset, data in [('train', train_folder), ('validation', validation_folder)]: 
        if data:
            process_and_save(data, output_folder, subset)

# Usage example
# output_folder = 'dataset/AQUA'
output_folder = 'dataset/VQA-MED'
# class_name = 'other'
# val_samples = 300

# save_dataset('AQUA', output_folder)
save_dataset('VQA-MED', output_folder)

# save_dataset('Multimodal-Fatima/OK-VQA_train', output_folder, class_name, 'train', val_samples)
# save_dataset('Multimodal-Fatima/OK-VQA_test', output_folder, class_name, 'test')
