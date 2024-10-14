import json
import random
import os

random.seed(42)

def split_q_instruct(datalist, output_folder, train_samples, test_samples):
    train_folder = os.path.join(output_folder, 'train')
    test_folder = os.path.join(output_folder, 'test')
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    
    json_data_list_1 = []
    json_data_list_2 = []
    json_data_list_3 = []
    json_data_list_4 = []
    
    for item in datalist:
        item['image'] = output_folder + '/' + item['image']
        
        if "what" in item['conversations'][0]['value'].lower() and item['conversations'][1]['value'] in ['A.','B.','C.','D.', 'E.']:
            item['conversations'][1]['value'] = item['conversations'][1]['value'][:-1]
            json_data_list_1.append(item)
        elif "how" in item['conversations'][0]['value'].lower() and item['conversations'][1]['value'] in ['A.','B.','C.','D.', 'E.']:
            item['conversations'][1]['value'] = item['conversations'][1]['value'][:-1]
            json_data_list_2.append(item)
        elif "yes" in item['conversations'][0]['value'].lower() and "no" in item['conversations'][0]['value'].lower() and item['conversations'][1]['value'] in ['A.','B.','C.','D.', 'E.']:
            item['conversations'][1]['value'] = item['conversations'][1]['value'][:-1]
            json_data_list_3.append(item)
        elif len(item['conversations'][1]['value'].split(' ')) > 3:
            json_data_list_4.append(item)
    
    print(len(json_data_list_1), len(json_data_list_2), len(json_data_list_3), len(json_data_list_4))
    # Shuffle the final list to mix types
    random.shuffle(json_data_list_1)
    random.shuffle(json_data_list_2)
    random.shuffle(json_data_list_3)
    random.shuffle(json_data_list_4)
    
    
    json_data_list_train = json_data_list_1[:int(train_samples/2)] + json_data_list_2[:int(train_samples/2)]
    json_data_list_test = json_data_list_1[-int(test_samples/2):] + json_data_list_2[-int(test_samples/2):]
    
    json_data_list_2_train = json_data_list_3[:int(train_samples)]
    json_data_list_2_test = json_data_list_3[:int(test_samples)]
    
    json_data_list_3_train = json_data_list_4[:int(train_samples)]
    json_data_list_3_test = json_data_list_4[:int(test_samples)]
    
    # Save the JSON data list to a file
    json_output_path = os.path.join(train_folder, f'dataset-0.json')
    print(f"Total samples: {len(json_data_list_train)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_train, json_file, indent=4)
        
    json_output_path = os.path.join(test_folder, f'dataset-0.json')
    print(f"Total samples: {len(json_data_list_test)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_test, json_file, indent=4)

    json_output_path = os.path.join(train_folder, f'dataset-1.json')
    print(f"Total samples: {len(json_data_list_2_train)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_2_train, json_file, indent=4)
        
    json_output_path = os.path.join(test_folder, f'dataset-1.json')
    print(f"Total samples: {len(json_data_list_2_test)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_2_test, json_file, indent=4)
    
    json_output_path = os.path.join(train_folder, f'dataset-2.json')
    print(f"Total samples: {len(json_data_list_3_train)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_3_train, json_file, indent=4)
        
    json_output_path = os.path.join(test_folder, f'dataset-2.json')
    print(f"Total samples: {len(json_data_list_3_test)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_3_test, json_file, indent=4)
        
        
def split_multi_q_instruct(datalist, output_folder, train_samples, test_samples):
    train_folder = os.path.join(output_folder, 'train')
    test_folder = os.path.join(output_folder, 'test')
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    
    json_data_list_1 = []
    json_data_list_2 = []
    json_data_list_3 = []
    
    for item in datalist:
        if isinstance(item['image'], list):
            item['image'] = [output_folder + '/' + img for img in item['image']]
        
        if len(item['image']) == 2:
            json_data_list_1.append(item)
        elif len(item['image']) == 3:
            json_data_list_2.append(item)
        elif len(item['image']) == 4:
            json_data_list_3.append(item)
    
    print(len(json_data_list_1), len(json_data_list_2), len(json_data_list_3))
    # Shuffle the final list to mix types
    random.shuffle(json_data_list_1)
    random.shuffle(json_data_list_2)
    random.shuffle(json_data_list_3)
    
    
    json_data_list_train = json_data_list_1[:int(train_samples)]
    json_data_list_test = json_data_list_1[:int(test_samples)]
    
    json_data_list_2_train = json_data_list_2[:int(train_samples)]
    json_data_list_2_test = json_data_list_2[:int(test_samples)]
    
    json_data_list_3_train = json_data_list_3[:int(train_samples)]
    json_data_list_3_test = json_data_list_3[:int(test_samples)]
    
    # Save the JSON data list to a file
    json_output_path = os.path.join(train_folder, f'dataset-3.json')
    print(f"Total samples: {len(json_data_list_train)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_train, json_file, indent=4)
        
    json_output_path = os.path.join(test_folder, f'dataset-3.json')
    print(f"Total samples: {len(json_data_list_test)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_test, json_file, indent=4)

    json_output_path = os.path.join(train_folder, f'dataset-4.json')
    print(f"Total samples: {len(json_data_list_2_train)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_2_train, json_file, indent=4)
        
    json_output_path = os.path.join(test_folder, f'dataset-4.json')
    print(f"Total samples: {len(json_data_list_2_test)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_2_test, json_file, indent=4)
    
    # json_output_path = os.path.join(train_folder, f'dataset-5.json')
    # print(f"Total samples: {len(json_data_list_3_train)}")
    # with open(json_output_path, 'w') as json_file:
    #     json.dump(json_data_list_3_train, json_file, indent=4)
        
    # json_output_path = os.path.join(test_folder, f'dataset-5.json')
    # print(f"Total samples: {len(json_data_list_3_test)}")
    # with open(json_output_path, 'w') as json_file:
    #     json.dump(json_data_list_3_test, json_file, indent=4)


with open('./dataset/Co-Instruct-DB/coinstruct_562k_llava_format.json', 'r') as fp:
    datalist = json.load(fp)

original_q_instruct = datalist[:200277]
multi_q_instruct = datalist[200277:302409]

split_q_instruct(original_q_instruct, 'dataset/Co-Instruct-DB', 10000, 2000)

split_multi_q_instruct(multi_q_instruct, 'dataset/Co-Instruct-DB', 10000, 2000)

