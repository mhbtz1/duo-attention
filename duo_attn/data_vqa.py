from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
import datasets
import json
import os
import torch
import transformers

from .data import DEFAULT_IMAGE_TOKEN

print(f"HF_DATASETS_CACHE: {datasets.config.HF_DATASETS_CACHE}")
print(f"HF_CACHE_HOME: {datasets.config.HF_CACHE_HOME}")


def get_dataset(data_files, split="train", size=None):
    print(f"dataset files; {data_files}")
    # print(f"data files: {data_files}")
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
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        image_processor,
        questions_path="/workspace/coco2014/v2_OpenEnded_mscoco_train2014_questions.json",
        images_path="/workspace/coco2014/images/train2014",
        annotations_path="/workspace/coco2014/v2_mscoco_train2014_annotations.json",
        split="train",
        size=None,
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.tokenizer_encode = self._get_encoder(self.tokenizer)
        self.questions = json.load(open(questions_path, "r"))
        self.images = images_path
        self.annotations = json.load(open(annotations_path, "r"))
        print(
            f'{len(self.questions["questions"])} {len(self.annotations["annotations"])}'
        )
        assert len(self.questions["questions"]) == len(self.annotations["annotations"])

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        # print(f'{idx=}')
        question = self.questions["questions"][idx]["question"]
        image_id = self.questions["questions"][idx]["image_id"]
        image_file = f"{self.images}/COCO_train2014_{image_id:0>12}.jpg"
        if not os.path.exists(image_file):
            print(f"Can't find image_file: {image_file}")
            assert False

        # add image placeholder to question
        question = DEFAULT_IMAGE_TOKEN + " " + question

        answer_choices = self.annotations["annotations"][idx]["answers"]
        answer = None
        answer_yes_count = dict()
        for choice in answer_choices:
            if choice["answer_confidence"] == "yes":
                answer_yes_count[choice["answer"]] = (
                    answer_yes_count.get(choice["answer"], 0) + 1
                )
        answer = max(answer_yes_count, key=answer_yes_count.get)

        # Currently a list of answers; to get one possible answer, you can do answer_choices[0]["answer"]
        assert (
            self.questions["questions"][idx]["question_id"]
            == self.annotations["annotations"][idx]["question_id"]
        )
        question_tokens = self.tokenizer_encode(question)
        answer_tokens = self.tokenizer_encode(answer)

        image = Image.open(image_file).convert("RGB")
        pixel_values = torch.tensor(
            self.image_processor(image).pixel_values[0]
        )  # .unsqueeze(0)

        input_ids = torch.cat((question_tokens, answer_tokens))
        labels = torch.cat((torch.tensor([-100] * len(question_tokens)), answer_tokens))

        return dict(input_ids=input_ids, labels=labels, pixel_values=pixel_values)

    def _get_encoder(self, tokenizer):
        def f(text, start=False):  # a_s_t ignored
            tokens = tokenizer(text, return_tensors="pt")
            tokens = tokens.input_ids.squeeze(0)
            if not start:
                tokens = tokens[1:]  # Remove extra start token
            return tokens

        return f
