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
from transformers import AutoProcessor, CLIPVisionConfig
from typing import List, Dict, Any
from duo_attn.data import get_dataset
import matplotlib.pyplot as plt
from duo_attn.data import MultiplePasskeyRetrievalDataset


class PassKeyDataset(MultiplePasskeyRetrievalDataset):

    def __init__(self, haystack_dataset=None, processor=None):
        print("PasskeyDataset.__init__ called with:", haystack_dataset, processor)
        super(PassKeyDataset, self).__init__(haystack_dataset, processor.tokenizer, context_length_min=0, context_length_max=2000)
        
        self.processor = processor
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])  
        image_processor = self.processor.image_processor
        processed_image = lambda x: image_processor(x, return_tensors="pt")


        # note: some weird issue with pixel_values being empty after applying self.processor, might be some issue with images      
        self.all_images = [
                    torch.tensor(processed_image(Image.open(os.path.join(os.path.join(os.environ["ROOT_DIR"], "augmented_images"), file)))["pixel_values"])
                    for file in os.listdir(os.path.join(os.environ["ROOT_DIR"], "augmented_images"))]

        print(f"type of self.all_images[0]: {type(self.all_images[0])}")

        print(f"Sampled image random dimensions: {self.all_images[random.randint(0, len(self.all_images)-1)].size()}")

        self.all_items = [super(PassKeyDataset, self).__getitem__(random.randint(0, len(self.context_length_intervals) - 1)) for _ in range(len(self.all_images))]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        '''
        print(f"Image at index {ridx} : {self.all_images[ridx]}")
        print(f"type(self.all_images[ridx]) : {type(self.all_images[ridx])}")
        plt.imshow(torch.transpose(self.all_images[ridx], 0, 2).numpy())
        plt.savefig('output.png')
        plt.close()  # Close the figure to free memory
        '''
        print(f"Finished initialization. len(self.all_items): {len(self.all_items)}")
    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        #print(tokenized_input['input_ids'].shape, tokenized_input['attention_mask'].shape, tokenized_input['pixel_values'].shape)
        processor_dict = dict({"pixel_values": self.all_images[idx], "input_ids": self.all_items[idx]["input_ids"], "labels": self.all_items[idx]["labels"]})
        return processor_dict


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
            train_dict = {**element, 'split': 'train'}
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

        print(f"Is file gzipped: {'yes' if is_gz_file(os.path.join(os.environ['ROOT_DIR'], 'passkey_images.json.gz')) else 'no'}")
