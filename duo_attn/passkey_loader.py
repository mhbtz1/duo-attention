import os
import torch
import json
import random
import gzip
from pathlib import Path
from datasets import Dataset
from torchvision import transforms
from typing import Dict, Callable
from PIL import Image
from transformers import AutoProcessor
from typing import List, Dict, Any
from duo_attn.data import get_dataset
import matplotlib.pyplot as plt


class PasskeyDataset(torch.utils.data.Dataset):
    def __init__(self, processor: AutoProcessor):
        super(PasskeyDataset, self).__init__()
        
        self.processor = processor
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # note: some weird issue with pixel_values being empty after applying self.processor, might be some issue with images      
        self.all_images = [self.transform(Image.open(os.path.join(os.path.join(os.environ["ROOT_DIR"], "augmented_images"), file))) 
                        for file in os.listdir(os.path.join(os.environ["ROOT_DIR"], "augmented_images"))]
        self.all_prompts = ["This is a placeholder prompt. Answer how you see fit." for _ in range(len(self.all_images))]
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        ridx = random.randint(0, len(self.all_images)-1)
        '''
        print(f"Image at index {ridx} : {self.all_images[ridx]}")
        print(f"type(self.all_images[ridx]) : {type(self.all_images[ridx])}")
        plt.imshow(torch.transpose(self.all_images[ridx], 0, 2).numpy())
        plt.savefig('output.png')
        plt.close()  # Close the figure to free memory
        '''
    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        tokenized_input = self.processor(images=[self.all_images[idx]], text=self.all_prompts[idx], padding=True, return_tensors="pt").to(self.device)
        print(tokenized_input['input_ids'].shape, tokenized_input['attention_mask'].shape, tokenized_input['pixel_values'].shape)
        processor_tuple = (tokenized_input["input_ids"].tolist(), tokenized_input["attention_mask"].tolist(), tokenized_input["pixel_values"].tolist())
        return processor_tuple, self.all_prompts[idx] # by default, just make the VLM mimic its text output


def save_as_hf_dataset_advanced(
    dataset: torch.utils.data.Dataset,
    output_path: str,
    special_handlers: Dict[str, Callable] = None
) -> None:
    """
    Advanced version with custom type handling.
    
    Args:
        dataset: PyTorch Dataset object
        output_path: Path for output JSON file
        feature_names: List of feature names
        special_handlers: Dict mapping feature names to conversion functions
    """
    special_handlers = special_handlers or {}
    data_dicts = []
    
    for idx in range(len(dataset)):
        item = dataset[idx]
        
        if isinstance(item, tuple):
            item_dict = {"tokenized_input" : item[0], "response": item[1]}
        elif isinstance(item, dict):
            item_dict = {}
            for k, v in item.items():
                if k in special_handlers:
                    v = special_handlers[k](v)
                elif isinstance(v, torch.Tensor):
                    v = v.tolist()
                item_dict[k] = v
                
        data_dicts.append(item_dict)

    
    with gzip.open(output_path, 'wt') as f:
        for idx, element in enumerate(data_dicts):
            tok = element['tokenized_input']
            print(len(tok))
            if len(tok[0]) == 0:
                print(f"input_ids on index {idx} invalid, skipping...")
                continue
            if len(tok[1]) == 0:
                print(f"pixel values on index {idx} invalid, skipping...")
                continue
            if len(tok[2]) == 0:
                print(f"attention mask on index {idx} invalid, skipping...")
                continue

            train_dict = {'input_ids': tok[0], 'pixel_values': tok[1], 'attention_mask' : tok[2], 'split': 'train'}
            json.dump(train_dict, f)
            f.write("\n")
    
    #save_dataset_compressed(data_dicts, output_path)   


def save_dataset_compressed(data_dicts: List[Dict[str, Any]], output_path: str):
    dataset = Dataset.from_list(data_dicts)
    dataset.save_to_disk(output_path, compression='zstd')



if __name__ == '__main__':
    if 'DECOMPRESS' not in os.environ.keys():
        proc = AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')
        output_dir = os.path.join(os.environ['ROOT_DIR'], 'passkey_images.json.gz')
        dataset = PasskeyDataset(proc)
        save_as_hf_dataset_advanced(dataset, output_dir)
    else:
        print("Fetching passkey_images.json.gz")
        pth = os.path.join(os.environ["ROOT_DIR"], 'passkey_images.json.gz')
        lines = gzip.open(pth, 'rt').readlines()
        dataset = get_dataset(os.path.join(os.environ["ROOT_DIR"], 'passkey_images.json.gz'))
        print(f"type(dataset): {type(dataset)}")
        print(f"split: {dataset.split}")
        print(f"features: {dataset.features}")
        print(f"info: {dataset.info}")

        def is_gz_file(filepath):
            with open(filepath, 'rb') as test_f:
                return test_f.read(2) == b'\x1f\x8b'

        print(f"Is file gzipped: {"yes" if is_gz_file(os.path.join(os.environ["ROOT_DIR"], 'passkey_images.json.gz')) else "no"}")