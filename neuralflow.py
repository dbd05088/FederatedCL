"""
    NeuralFlow - Plot intermediate output of Mistral 7B

    Copyright (C) 2024 Lukas Valine

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
    
    https://github.com/valine/NeuralFlow/blob/master/src/neural_flow.py
"""

import datetime
import os
import json
import random
from tqdm import tqdm

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import transformers
from transformers import BitsAndBytesConfig
from configuration.VLM_config import ModelArguments, DataArguments, TrainingArguments
from utils.train_utils import get_VLMmodel
from utils.data_loader_VLM import GenerationDataset

device_0 = "cuda:0"
model_folder = "/models/OpenHermes-2.5-Mistral-7B"
image_output_folder = "./"


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model, tokenizer, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)
    
    train_datalists, test_datalists = get_datalists(training_args, training_args.scenario)
    samples_per_round_per_client = [len(train_datalists[i]) // training_args.num_rounds for i in range(training_args.num_clients)]
    
    client_id = 2
    client_state_dict = torch.load(f'./client_states_{training_args.note}/{client_id}_client_model_round{training_args.round_to_eval}.pth', map_location='cpu')
    
    test_datalist = test_datalists[client_id]
    for data_info in test_datalist:
        model.load_state_dict(client_state_dict, strict=False)
        dataset = GenerationDataset(data_info['data'], tokenizer, data_args)
        
        iterator = iter(dataset)
        probe_results = []
        for i in tqdm(range(len(dataset))):

        # Probe results is an array so that you can plot the changes to the
        # output over time. The plot_embedding_flow will generate an animated gif.
        # Call compute_model_output multiple times and append the results to
        # probe_results.
        
            sample = next(iterator)
            probe_result = compute_model_output(model, tokenizer, sample)
            probe_results.append(probe_result)
            # print(len(probe_result))

        plot_embedding_flow(probe_results)
        
        breakpoint()


def compute_model_output(base_model, tokenizer, sample):
    with torch.no_grad():
        layer_output = []

        # encoding = tokenizer(ground_truth, return_tensors="pt")
        # input_ids = encoding['input_ids'].to(device_0)
        input_ids = sample['input_ids'].unsqueeze(0).to(device_0)
        images = sample['image'].unsqueeze(0).to(device=device_0, dtype=torch.bfloat16)
        

        (input_ids,
        position_ids,
        attention_mask,
        _,
        inputs_embeds,
        _) = base_model.prepare_inputs_labels_for_multimodal(
            input_ids, 
            None,
            None,
            None,
            None,
            images,
        )
        hidden_states = inputs_embeds
        
        sequence_length = hidden_states.shape[1]
        batch_size = hidden_states.shape[0]
        # position_ids = torch.arange(sequence_length, device=device_0).unsqueeze(0)
        # position_ids = position_ids.expand(batch_size, -1)

        attention_mask = torch.triu(torch.full(
            (sequence_length, sequence_length), float('-inf')), diagonal=1)
        attention_mask = attention_mask.to(device_0)
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

        # Loop over layers
        for layer in base_model.model.model.layers:
            output = layer(hidden_states,
                           attention_mask=attention_mask,
                           position_ids=position_ids,
                           output_attentions=True)
            hidden_states = output[0]
            layer_output.append(hidden_states.cpu())
        return layer_output


def vectorized_get_color_rgb(value_tensor, max_value=1.0):
    h = (value_tensor * 1.0) / max_value
    s = torch.ones_like(h)
    v = torch.ones_like(h)

    c = v * s
    x = c * (1 - torch.abs((h * 6) % 2 - 1))
    m = v - c

    h1 = (h * 6).int()
    rgb = torch.stack((
        torch.where((h1 == 0) | (h1 == 5), c, torch.where((h1 == 1) | (h1 == 4), x, 0)),
        torch.where((h1 == 1) | (h1 == 2), c, torch.where((h1 == 0) | (h1 == 3), x, 0)),
        torch.where((h1 == 3) | (h1 == 4), c, torch.where((h1 == 2) | (h1 == 5), x, 0)),
    ), dim=-1) + m.unsqueeze(-1)

    return rgb


def generate_filename(prefix, extension):
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{formatted_time}.{extension}"
    return filename


def plot_layers(all_words, title, file_path, normalize=True):
    sequence_length = all_words[0].shape[1]
    paths = []

    all_words_cat = torch.cat(all_words, dim=0)
    global_min_val = torch.min(all_words_cat)
    global_max_val = torch.max(all_words_cat)

    global_mean = torch.mean(all_words_cat)
    global_var = torch.var(all_words_cat) * 25

    if normalize:
        min_val = global_mean - global_var
        max_val = global_mean + global_var
    else:
        min_val = global_min_val
        max_val = global_max_val

    for i in range(sequence_length):
        list_of_tensors = []
        for tensor in all_words:
            list_of_tensors.append(tensor[:, i, :])

        # Step 1: Concatenate tensors along width
        full_tensor = torch.cat(list_of_tensors, dim=0)  # Shape: [1, 4096 * 31]
        height = 512

        tensor_split = [F.pad(t, (0, max(0, height - t.shape[1])),
                              'constant', max_val.item()) for t in
                        torch.split(full_tensor, height, dim=1)]
        reshaped_tensor = torch.cat(tensor_split, dim=0)
        reshaped_tensor = torch.abs(reshaped_tensor)

        # Normalize data
        normalized_data = (reshaped_tensor - min_val) / (max_val - min_val)
        color_tensor = vectorized_get_color_rgb(normalized_data)

        # Generate image
        array = (color_tensor.cpu().float().numpy() * 255).astype(np.uint8)
        image = Image.fromarray(array, 'RGB')

        # Save the image
        tmp_name = "raw_values_tmp" + str(i)
        filename = title + "_" + generate_filename(tmp_name + str(i), "png")
        full_path = os.path.join(file_path, filename)
        image.save(full_path)

        paths.append(full_path)

    # Create gif from images
    filename = title + "_" + generate_filename("layers", "gif")
    gif_path = os.path.join(file_path, filename)
    with imageio.get_writer(gif_path, mode='I', fps=15, loop=0) as writer:
        for filename in paths:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Remove temporary files
    for filename in paths:
        os.remove(filename)

    # open_image(gif_path)
    return gif_path


def plot_embedding_flow(probe_results):
    layer_count = len(probe_results[0])
    layer_embeddings = []
    for l_index in range(layer_count):
        sequence_embedding = []
        for probe_result in probe_results:
            embedding = probe_result[l_index][:, -1, :]
            sequence_embedding.append(embedding)
        layer_embedding = torch.stack(sequence_embedding, dim=1)
        layer_embeddings.append(layer_embedding)

    # Plot current progress
    path = plot_layers(layer_embeddings, "probe_results", image_output_folder)
    return path

def get_datalists(args, scenario_num):
    with open(f"./scenarios/scenario-{scenario_num}.json") as fp:
        scenario = json.load(fp)
    assert args.num_clients == len(scenario)
    
    train_datalists = {}
    test_datalists = {}
    
    for client_data in scenario:
        client_id = client_data['client_id']
        train_datalist = []
        test_datalist = []
        eval_cnt = 0
        for data in client_data['datasets']:
            with open(f"./dataset/{data['dataset']}/train/dataset-{str(data['subset_id'])}.json") as fp:
                datalist = json.load(fp)
            random.shuffle(datalist)
            train_datalist.extend(datalist)
            
            with open(f"./dataset/{data['dataset']}/test/dataset-{str(data['subset_id'])}.json") as fp:
                datalist = json.load(fp)
            test_datalist.append({
                "data_name": f"{data['dataset']}-{data['subset_id']}",
                "data": datalist,
                "eval_cnt": eval_cnt})
            eval_cnt += len(datalist)
            
        train_datalists[client_id] = train_datalist
        test_datalists[client_id] = test_datalist
    
    return train_datalists, test_datalists


if __name__ == "__main__":
    main()