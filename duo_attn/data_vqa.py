import torch
import datasets
from datasets import load_dataset, DownloadMode, enable_caching
from dataclasses import dataclass
from typing import Sequence, Dict
import torch
import transformers
from torch.utils.data import Dataset, IterableDataset
import os
import csv
import json
from multiprocessing.sharedctypes import Value
import os
from pathlib import Path
import datasets
from PIL import Image

DEFAULT_IMAGE_TOKEN = "<image>"


print(f"HF_DATASETS_CACHE: {datasets.config.HF_DATASETS_CACHE}")
print(f"HF_CACHE_HOME: {datasets.config.HF_CACHE_HOME}")


def get_dataset(data_files, split="train", size=None):
    print(f"dataset files; {data_files}")
    #print(f"data files: {data_files}")
    dataset = load_dataset("json", data_files=data_files, split=split)
    if size is not None:
        dataset = dataset.select(range(size))
    print(f"type(dataset): {type(dataset)}")
    return dataset

# IMAGE_PATH = os.getenv("VQA_IMAGES") 
# if not IMAGE_PATH:
#     IMAGE_PATH = "/root/coco2014/images/train2014"
# QUESTION_PATH = os.getenv("VQA_QUESTIONS")
# if not QUESTION_PATH:
#     QUESTION_PATH = "/root/coco2014/v2_OpenEnded_mscoco_train2014_questions.json"

_CITATION = """\
@InProceedings{VQA,
author = {Stanislaw Antol and Aishwarya Agrawal and Jiasen Lu and Margaret Mitchell and Dhruv Batra and C. Lawrence Zitnick and Devi Parikh},
title = {VQA: Visual Question Answering},
booktitle = {International Conference on Computer Vision (ICCV)},
year = {2015},
} 
"""

_DESCRIPTION = """\
VQA is a new dataset containing open-ended questions about images. These questions require an understanding of vision, language and commonsense knowledge to answer.
"""

_HOMEPAGE = "https://visualqa.org"

_LICENSE = "CC BY 4.0"  # TODO need to credit both ms coco and vqa authors!

_URLS = {
    "questions": {
        "train": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
        "val": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
        "test-dev": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip",
        "test": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip",
    },
    "annotations": {
        "train": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
        "val": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
    },
    "images": {
        "train": "http://images.cocodataset.org/zips/train2014.zip",
        "val": "http://images.cocodataset.org/zips/val2014.zip",
        "test-dev": "http://images.cocodataset.org/zips/test2015.zip",
        "test": "http://images.cocodataset.org/zips/test2015.zip",
    },
}
_SUB_FOLDER_OR_FILE_NAME = {
    "questions": {
        "train": "v2_OpenEnded_mscoco_train2014_questions.json",
        "val": "v2_OpenEnded_mscoco_val2014_questions.json",
        "test-dev": "v2_OpenEnded_mscoco_test-dev2015_questions.json",
        "test": "v2_OpenEnded_mscoco_test2015_questions.json",
    },
    "annotations": {
        "train": "v2_mscoco_train2014_annotations.json",
        "val": "v2_mscoco_val2014_annotations.json",
    },
    "images": {
        "train": "train2014",
        "val": "val2014",
        "test-dev": "test2015",
        "test": "test2015",
    },
}


class VQADataset(Dataset):
    def __init__(self, 
                 tokenizer: transformers.PreTrainedTokenizer,
                 questions_path="/workspace/coco2014/v2_OpenEnded_mscoco_train2014_questions.json",
                 images_path="/workspace/images",
                 annotations_path="/workspace/coco2014/v2_mscoco_train2014_annotations.json",
                 split="train", 
                 size=None):

        self.tokenizer = tokenizer
        self.tokenizer_encode = self._get_encoder(self.tokenizer)
        self.questions = json.load(open(questions_path, "r"))
        self.images = images_path
        self.annotations = json.load(open(annotations_path, "r"))
        print(f'{len(self.questions["questions"])} {len(self.annotations["annotations"])}')
        assert(len(self.questions["questions"]) == len(self.annotations["annotations"]))

    def __len__(self):
        return len(self.questions)


    def __getitem__(self, idx):
        question, image_file = None, "asdfjhasdkflhaslkfhaskfh"
        while not os.path.exists(image_file):
            question = self.questions["questions"][idx]["question"]
            image_id = self.questions["questions"][idx]["image_id"]
            image_file = f"{self.images}/COCO_{self.images}_{image_id:0>12}.jpg"
            
        answer_choices = self.annotations["annotations"][idx]["answers"] 
        answer = None
        answer_yes_count = dict()
        for choice in answer_choices:
            if choice["answer_confidence"] == "yes":
                answer_yes_count[choice["answer"]] = answer_yes_count.get(choice["answer"], 0) + 1
        answer = max(answer_yes_count, key=answer_yes_count.get)
        
        #Currently a list of answers; to get one possible answer, you can do answer_choices[0]["answer"]
        assert(self.questions["questions"][idx]["question_id"] == self.annotations["annotations"][idx]["question_id"])
        question_tokens = self.tokenizer_encode(question["question"])
        answer_tokens = self.tokenizer_encode(answer)

        image = Image.open(image_file).convert('RGB')
        pixel_values = torch.tensor(self.image_processor(image).pixel_values[0])#.unsqueeze(0)
        
        input_ids = torch.cat((question_tokens, answer_tokens))
        labels = torch.cat((torch.tensor([-100] * len(question_tokens)), answer_tokens))

        return dict(input_ids=input_ids, labels=labels, pixel_values=pixel_values)
    
    def _get_encoder(self, tokenizer):
        def f(text, start=False): #a_s_t ignored
            tokens = tokenizer(text, return_tensors="pt")
            tokens = tokens.input_ids.squeeze(0)
            if not start:
                tokens = tokens[1:]#Remove extra start token
            return tokens
        return f


from transformers.image_utils import get_image_size, to_numpy_array

@dataclass
class DataCollator(object):
    """Collate examples for supervised fine-tuning."""

    #tokenizer: transformers.PreTrainedTokenizer
    def __init__(self, processor: transformers.LlavaProcessor):
        self.processor = processor
        self.image_token = self.processor.tokenizer.encode(DEFAULT_IMAGE_TOKEN)[-1]

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        #pad_sequence expects 1D inputs, but tokenizer has output with shape [1, X]
        '''
        print(f"length of instances: {len(instances)}")
        print(f"keys in instances: {instances[0].keys()}")
        print(f"input_ids shape: {instances[0]['input_ids'].size()}")
        print(f"pixel_values shape: {instances[0]['pixel_values'].size()}")
        print(f"labels shape: {instances[0]['labels'].size()}" )
        '''
        input_ids, labels, pixel_values = tuple(
            [instance[key].squeeze() for instance in instances] for key in ("input_ids", "labels", "pixel_values")
        )

        #Everywhere there is an image token (32000), we need to replace it with many instances to match the size of an image
        #Bacuse this is all batched, we can assume all images in the batch have the same size. 
        height, width = get_image_size(to_numpy_array(pixel_values[0])) 
        num_image_tokens = (height // self.processor.patch_size) * (
            width // self.processor.patch_size
        ) + 1 #1 == self.processor.num_additional_image_tokens 
        if self.processor.vision_feature_select_strategy == "default":
            num_image_tokens -= 1
        #print("Target:", num_image_tokens)
        repeated_image = torch.ones(num_image_tokens, dtype=input_ids[0].dtype) * self.image_token
        repeated_labels = torch.ones(num_image_tokens, dtype=labels[0].dtype) * -100 #Hardcoded 
        for i in range(len(input_ids)):
            ids = input_ids[i]
            lab = labels[i]
            image_idx = (ids == self.image_token).nonzero()[0]
            for j, idx in enumerate(image_idx):
                idx += j * (num_image_tokens-1) #Correct for previous insertions
                ids = torch.cat((ids[:idx], repeated_image, ids[idx+1:]))
                lab = torch.cat((lab[:idx], repeated_labels, lab[idx+1:]))
            input_ids[i] = ids
            labels[i] = lab


        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        pixel_values = torch.stack(pixel_values)

        ret_dict = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.processor.tokenizer.pad_token_id), #Left from text-only
            pixel_values=pixel_values,
        )
        for key in instances[0].keys():
            if key not in ret_dict: #This should add the image stuff to the return dict
                ret_dict[key] = torch.stack([instance[key] for instance in instances])
        return ret_dict

def get_supervised_dataloader(
    dataset, processor, batch_size, num_workers=4, shuffle=True, sampler=None
):
    collator = DataCollator(processor)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collator,
        shuffle=None if sampler is not None else shuffle,
        sampler=sampler,
    )
    return dataloader
