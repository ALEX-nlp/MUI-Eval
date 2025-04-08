import torch
import argparse
import json
from transformer_lens import HookedTransformer
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, LlamaTokenizer, pipeline, AutoConfig, BitsAndBytesConfig
import os
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import time
from utils_neuron import *
import itertools
from lm_saes import SparseAutoEncoder

TOTAL_FEATURE = 131072
def create_collate_fn(tokenizer):
    pad_token_id = tokenizer.pad_token_id  

    def collate_fn(batch):
        input_ids = pad_sequence([item['input_ids'].squeeze() for item in batch], batch_first=True, padding_value=pad_token_id)
        attention_mask = pad_sequence([item['attention_mask'].squeeze() for item in batch], batch_first=True, padding_value=0)
        prompt_indices = torch.tensor([item['prompt_index'] for item in batch])
        prompt_end_indices = torch.tensor([item['prompt_end_index'] for item in batch])
        prompts_text = [item['prompt_text'] for item in batch]
        responses_text = [item['response_text'] for item in batch]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'prompt_indices': prompt_indices,
            'prompt_end_indices': prompt_end_indices,
            'prompts_text': prompts_text,
            'responses_text': responses_text
        }

    return collate_fn

class ResponseDataset(Dataset):
    def __init__(self, datajson, tokenizer, model, task):
        self.datajson = datajson
        self.tokenizer = tokenizer
        self.model = model
        self.task = task

    def __len__(self):
        return len(self.datajson)

    def __getitem__(self, idx):
        row = self.datajson[idx]
        sys = row['sys_input']
        question = row['question_input']
        response = row['response']
        prompt, prompt_index, prompt_end_index = get_formated_instruction(sys, question, response, self.model, self.tokenizer, self.task)
        return {
            'input_ids': prompt['input_ids'],
            'attention_mask': prompt['attention_mask'],
            'prompt_index': prompt_index,
            'prompt_end_index': prompt_end_index,
            'prompt_text': row['question_input'],
            'response_text': row['response']
        }


def get_formated_instruction(sys, prompt, response, model_name, tokenizer, task = None):
    """
    Formats a given prompt based on the specific requirements of different language models and tasks.

    Returns:
    - A tuple containing:
        1. Tokenized and formatted input and response combined, ready for model processing.
        2. The starting index of the response part in the tokenized sequence.
        3. The ending index of the response in the tokenized sequence.

    """

    if "llama3.1-base" in model_name:
        return_sample = prompt
    else:
        return_sample = prompt
    
    prompt_model_inputs = tokenizer([return_sample], return_tensors="pt")
    index_response = len(prompt_model_inputs.input_ids[0])
    prompt_model_inputs_with_response = tokenizer([return_sample + response], max_length=2048, return_tensors="pt") #  truncate inputs to a maximum token length of 2048, as dictated by Llama Scope’s encoder limitations.
    index_response_end = len(prompt_model_inputs_with_response.input_ids[0])

    return prompt_model_inputs_with_response, index_response, index_response_end



def get_feature():
    def get_model_response_file(model=None, task=None, os_result="../result/"):
        response_json = []
        with open(f'{os_result}/{task}/{args.model}/{args.model}.jsonl', "r") as file:
            for line in file:
                response_json.append(json.loads(line.strip()))
        return response_json

    print(f"===================Loading Model {args.model}=================")

    if args.model_path is None:
        model_path = MODEL_PATHS[args.model]
    else:
        model_path = args.model_path
    
    #load_type = torch.float16 # Adjust the calculation precision according to the size of gpu memory
    load_type = torch.float32
  
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    print("Load tokenizer successfully")

    model_config = AutoConfig.from_pretrained(model_path)
    print("Load config successfully")

    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=load_type,
        config=model_config,
        device_map={"": "cuda:0"}
    ).eval()
    model.generation_config.use_cache = False

    hook_model = HookedTransformer.from_pretrained_no_processing(
        "meta-llama/Llama-3.1-8B",
        device="cuda:1",
        hf_model=model,
        tokenizer=tokenizer,
        dtype=torch.float32,
    ).eval()

    vocab_size = tokenizer.vocab_size
    unembedding = model.lm_head.weight.data.T.cuda()[:, :vocab_size]

    num_layers = model.config.num_hidden_layers

    # Load SAE models for each layer (on GPU2~3)
    sae_models = {}
    for layer in range(model.config.num_hidden_layers):
        if layer < 10:
            sae_device = "cuda:2"  
        else:
            sae_device = "cuda:3"  
        
        sae_layer_path = f"../../model/SAE/Llama3_1-8B-Base-LXR-32x/Llama3_1-8B-Base-L{layer}R-32x"
        sae_models[layer] = SparseAutoEncoder.from_pretrained(sae_layer_path).to(sae_device)
        if sae_models[layer].cfg.norm_activation == "dataset-wise":
            sae_models[layer].standardize_parameters_of_dataset_norm(None)
        if sae_models[layer].cfg.sparsity_include_decoder_norm:
            sae_models[layer].transform_to_unit_decoder_norm()

    print("Loaded SAE models for all layers")

    res_df = get_model_response_file(model=args.model, task=args.task)

    collate_fn = create_collate_fn(tokenizer)
    dataset = ResponseDataset(res_df, tokenizer, args.model, args.task)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn) #Build data loader


    output_neuron_dir = f'get_SAE_result/{args.model}/'
    file_path = f'{output_neuron_dir}/{args.model}_{args.task}_result.jsonl'
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # Features gettting
    for batch in tqdm(dataloader, desc='Processing batches'):
        act = {n: [] for n in range(model.config.num_hidden_layers)}  # Initialize activations storage

        prompts = {
            'input_ids': batch['input_ids'].cuda(device='cuda:0'),  # Model on GPU0
            'attention_mask': batch['attention_mask'].cuda(device='cuda:0')  # Model on GPU0
        }
        prompt_indices = batch['prompt_indices'].cuda(device='cuda:1')
        prompt_end_indices = batch['prompt_end_indices'].cuda(device='cuda:1')
        prompts_text = batch['prompts_text']
        responses_text = batch['responses_text']

        forward_start_time = time.time()
        with torch.no_grad():
            _, cache = hook_model.run_with_cache(prompts["input_ids"])
        

        forward_end_time = time.time()
        forward_duration = forward_end_time - forward_start_time
        print(f"==========Time for forward pass: {forward_duration:.4f} seconds==========")

        for n in range(model.config.num_hidden_layers):
            act[n].append(cache[f"blocks.{n}.hook_resid_post"])

        activations = [act[n][0].to(torch.bfloat16).to("cuda:1") for n in range(model.config.num_hidden_layers)]
        del act, cache


        batch_size = activations[0].shape[0]
        seq_length = activations[0].shape[1]
        hidden_size = activations[0].shape[2]

        score_start_time = time.time()

        feature_acts_all_layers = {layer_idx: {b_idx: [] for b_idx in range(batch_size)} for layer_idx in range(num_layers)}

        for layer_idx in range(num_layers):
            sae = sae_models[layer_idx]
            sae_device = next(sae.parameters()).device
            feature_acts, _ = sae.encode(activations[layer_idx].to(sae_device), return_hidden_pre=True)  # [batch_size, seq, feature_size]
            for b_idx in range(batch_size):
                prompt_index = prompt_indices[b_idx].item()  
                prompt_end_index = prompt_end_indices[b_idx].item()

                active_token_indices = torch.arange(prompt_index, prompt_end_index).to(sae_device)
                active_feature_acts = feature_acts[b_idx, active_token_indices, :]

                pos_indices, feat_indices = (active_feature_acts > 0).nonzero(as_tuple=True) # get feature for each response token
                values = active_feature_acts[pos_indices, feat_indices]
                feature_acts_all_layers[layer_idx][b_idx].append((pos_indices.cpu(), feat_indices.cpu(), values.cpu()))

                del active_feature_acts, pos_indices, feat_indices, values
                
            del feature_acts
            
        

        score_end_time = time.time()
        score_duration = score_end_time - score_start_time
        print(f"==========Total time for score computation: {score_duration:.4f} seconds==========")
        
        # Saving the features 
        if not os.path.exists(output_neuron_dir):os.makedirs(output_neuron_dir)
        saving_start_time = time.time()

        for i in range(batch_size):
            results = {
                'prompt': prompts_text[i],
                'response': responses_text[i],
                'features': [],
            }
            for layer in range(num_layers):

                layer_data = feature_acts_all_layers[layer][i]
                feature_list = []
                for feature_data in layer_data:
                    pos_indices, feat_indices, values = feature_data
                    feature_info = [{
                        'pos_index': int(pos_idx),
                        'feat_index': int(feat_idx),
                        'value': float(val)
                    } for pos_idx, feat_idx, val in zip(pos_indices, feat_indices, values)]

                    feature_list.append({
                        "layer": layer,
                        "feature_info": feature_info
                    })
                
                results['features'].extend(feature_list)
                
            with open(f'{output_neuron_dir}/{args.model}_{args.task}_result.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps(results, ensure_ascii=False) + '\n')
        
        saving_end_time = time.time()
        saving_duration = saving_end_time - saving_start_time
        print(f"==========Time for SAE saving: {saving_duration:.4f} seconds==========")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Choose your model(s) & task(s)')
    parser.add_argument('--model',type=str,default="llama3.1-base",help='Provide the model you want to use. Check and choose from the key values of the MODEL_PATHS variable.')
    parser.add_argument('--model_path', type=str,default=None, help='Provide the directory for the model checkpoint')
    parser.add_argument('--task',type=str,default="mmlu", help='Provide the task name')
    parser.add_argument('--batch_size', type=int, default=1, help="Dataloader batch size")
    args = parser.parse_args()

    get_feature()