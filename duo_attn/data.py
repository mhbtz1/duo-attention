import torch
import datasets
from datasets import load_dataset
from dataclasses import dataclass
from typing import Sequence, Dict
import transformers
from torch.utils.data import Dataset

DEFAULT_IMAGE_TOKEN = "<image>"


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


class MultiplePasskeyRetrievalDataset(Dataset):
    PASSKEY_ALPHABET = [
        "alpha",
        "bravo",
        "charlie",
        "delta",
        "echo",
        "foxtrot",
        "golf",
        "hotel",
        "india",
        "juliett",
        "kilo",
        "lima",
        "mike",
        "november",
        "oscar",
        "papa",
        "quebec",
        "romeo",
        "sierra",
        "tango",
        "uniform",
        "victor",
        "whiskey",
        "xray",
        "yankee",
        "zulu",
    ]

    ORDINAL_NUMBERS = [
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
        "eleventh",
        "twelfth",
        "thirteenth",
        "fourteenth",
        "fifteenth",
        "sixteenth",
        "seventeenth",
        "eighteenth",
        "nineteenth",
        "twentieth",
    ]

    def __init__(
        self,
        haystack_dataset,
        tokenizer: transformers.PreTrainedTokenizer,
        max_length=None,
        passkey_length=32,
        num_passkeys=10,
        needle="Remeber this sequence of words, it's the {ordinal_number} passkey to the vault: ",
        retrieval_question="Based on the content of the book, what is the {ordinal_number} passkey to the vault?\nPasskey: ",
        prompt1="<|im_start|> This is a very long story book: <book> ",
        prompt2=" </book>.\n\n",
        buffer_size=300,
        seperator="\n\n",
        min_depth_ratio=0.1,
        max_depth_ratio=0.9,
        context_lengths_num_intervals=20,
        depth_ratio_num_intervals=20,
        context_length_min=None,
        context_length_max=None,
        pad_to_multiple_of=16,
    ):
        super(MultiplePasskeyRetrievalDataset, self).__init__()

        self.tokenizer = tokenizer

        self.max_length = (
            max_length if max_length is not None else tokenizer.model_max_length
        )
        self.max_depth_ratio = max_depth_ratio
        self.min_depth_ratio = min_depth_ratio
        self.context_lengths_num_intervals = context_lengths_num_intervals
        self.depth_ratio_num_intervals = depth_ratio_num_intervals

        if context_length_min is None or context_length_max is None:
            self.context_length_min = self.context_length_max = self.max_length
        else:
            self.context_length_min = context_length_min
            self.context_length_max = context_length_max

        print(f"context_lengths_num_intervals: {context_lengths_num_intervals}")
        print(f"context_length_min: {context_length_min}")
        print(f"context_length_max: {context_length_max}")
        self.context_length_intervals = torch.linspace(
            self.context_length_min,
            self.context_length_max,
            context_lengths_num_intervals,  # length of each irrelevant block of text in the haystack
            dtype=torch.int64,
        )

        self.depth_ratio_intervals = torch.linspace(
            min_depth_ratio, max_depth_ratio, depth_ratio_num_intervals
        )

        self.passkey_length = passkey_length

        self.num_passkeys = num_passkeys

        self.haystack = ""

        for sample in haystack_dataset["text"]:
            if self._get_token_nums(self.haystack) >= self.context_length_max:
                break
            self.haystack += sample

        self.haystack = self._trim(self.haystack, self.context_length_max)

        self.needle = needle
        self.needle_tokens_list = [
            self.tokenizer.encode(
                self.needle.format(ordinal_number=ordinal_number),
                add_special_tokens=False,
            )
            for ordinal_number in self.ORDINAL_NUMBERS[: self.num_passkeys]
        ]
        self.retrieval_question_tokens_list = [
            self.tokenizer.encode(
                retrieval_question.format(ordinal_number=ordinal_number),
                add_special_tokens=False,
            )
            for ordinal_number in self.ORDINAL_NUMBERS[: self.num_passkeys]
        ]

        self.haystack_tokens = self.tokenizer.encode(
            self.haystack, add_special_tokens=False
        )
        self.seperator_tokens = self.tokenizer.encode(
            seperator, add_special_tokens=False
        )
        self.prompt1_tokens = self.tokenizer.encode(prompt1, add_special_tokens=True)
        self.prompt2_tokens = self.tokenizer.encode(prompt2, add_special_tokens=False)

        passkey = self._generate_passkey()
        passkey_tokens = self.tokenizer.encode(passkey, add_special_tokens=False)
        needle_tokens = self.needle_tokens_list[0] + passkey_tokens

        other_input_len = (
            len(self.prompt1_tokens)
            + len(self.prompt2_tokens)
            + (
                len(self.seperator_tokens)
                + len(needle_tokens)
                + len(self.seperator_tokens)
                + len(self.retrieval_question_tokens_list[0])
                + len(passkey_tokens)
            )
            * self.num_passkeys
        )
        if (
            len(self.haystack_tokens) + other_input_len
            > self.context_length_max - buffer_size
        ):
            self.haystack_tokens = self.haystack_tokens[
                : self.context_length_max - buffer_size - other_input_len
            ]

    def _generate_passkey(self):
        random_seq = torch.randint(
            0, len(self.PASSKEY_ALPHABET), (self.passkey_length,)
        )
        passkey = " ".join([self.PASSKEY_ALPHABET[i] for i in random_seq])
        return passkey

    def __len__(self):
        return len(self.context_length_intervals)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        context_length = self.context_length_intervals[i]
        # randomly sample self.num_passkeys depth ratios in self.depth_ratio_intervals
        depth_ratios = (
            self.depth_ratio_intervals[
                torch.randperm(self.depth_ratio_num_intervals)[: self.num_passkeys]
            ]
            .sort()
            .values
        )
        passkey_tokens_list = [
            self.tokenizer.encode(self._generate_passkey(), add_special_tokens=False)
            for _ in range(self.num_passkeys)
        ]
        context = self._insert_needle(context_length, depth_ratios, passkey_tokens_list)
        return self._construct_input(context, passkey_tokens_list)

    def _trim(self, context, context_length):
        tokens = self.tokenizer.encode(context, add_special_tokens=False)
        if len(tokens) > context_length:
            context = self.tokenizer.decode(tokens[:context_length])
        return context

    def _get_token_nums(self, context):
        return len(self.tokenizer.encode(context))

    def _insert_needle(self, context_length, depth_ratios, passkey_tokens_list):
        haystack_tokens = self.haystack_tokens[:context_length]

        context = []
        last_insertion_point = 0

        for i, (depth_ratio, passkey_tokens) in enumerate(
            zip(depth_ratios, passkey_tokens_list)
        ):
            insertion_point = int(len(haystack_tokens) * depth_ratio)

            needle_tokens = self.needle_tokens_list[i] + passkey_tokens

            context += (
                haystack_tokens[last_insertion_point:insertion_point]
                + self.seperator_tokens
                + needle_tokens
                + self.seperator_tokens
            )
            last_insertion_point = insertion_point

        context += haystack_tokens[last_insertion_point:]

        return context

    def _construct_input(self, context_tokens, passkey_tokens_list):
        qa_tokens = []
        for i, (passkey_tokens, retrieval_question_tokens) in enumerate(
            zip(passkey_tokens_list, self.retrieval_question_tokens_list)
        ):
            qa_tokens += (
                retrieval_question_tokens + passkey_tokens + self.seperator_tokens
            )

        context_tokens = self.prompt1_tokens + context_tokens

        # pad to multiple of 16
        if len(context_tokens) + len(qa_tokens) + len(self.prompt2_tokens) % 16 != 0:
            pad_len = (
                16
                - (len(context_tokens) + len(qa_tokens) + len(self.prompt2_tokens)) % 16
            )
            context_tokens += self.haystack_tokens[-pad_len:]

        context_tokens += self.prompt2_tokens

        input_ids = torch.tensor(context_tokens + qa_tokens)

        assert input_ids.size(0) % 16 == 0

        labels = torch.tensor([-100] * len(context_tokens) + qa_tokens)
        return dict(input_ids=input_ids, labels=labels)


from transformers.image_utils import get_image_size, to_numpy_array


@dataclass
class DataCollator(object):
    """Collate examples for supervised fine-tuning."""

    # tokenizer: transformers.PreTrainedTokenizer
    def __init__(self, processor: transformers.LlavaProcessor):
        self.processor = processor
        self.image_token = self.processor.tokenizer.encode(DEFAULT_IMAGE_TOKEN)[-1]

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # pad_sequence expects 1D inputs, but tokenizer has output with shape [1, X]
        """
        print(f"length of instances: {len(instances)}")
        print(f"keys in instances: {instances[0].keys()}")
        print(f"input_ids shape: {instances[0]['input_ids'].size()}")
        print(f"pixel_values shape: {instances[0]['pixel_values'].size()}")
        print(f"labels shape: {instances[0]['labels'].size()}" )
        """
        input_ids, labels, pixel_values = tuple(
            [instance[key].squeeze() for instance in instances]
            for key in ("input_ids", "labels", "pixel_values")
        )

        # Everywhere there is an image token (32000), we need to replace it with many instances to match the size of an image
        # Bacuse this is all batched, we can assume all images in the batch have the same size.
        height, width = get_image_size(to_numpy_array(pixel_values[0]))
        num_image_tokens = (height // self.processor.patch_size) * (
            width // self.processor.patch_size
        ) + 1  # 1 == self.processor.num_additional_image_tokens
        if self.processor.vision_feature_select_strategy == "default":
            num_image_tokens -= 1
        # print("Target:", num_image_tokens)
        repeated_image = (
            torch.ones(num_image_tokens, dtype=input_ids[0].dtype) * self.image_token
        )
        repeated_labels = (
            torch.ones(num_image_tokens, dtype=labels[0].dtype) * -100
        )  # Hardcoded
        for i in range(len(input_ids)):
            ids = input_ids[i]
            lab = labels[i]
            image_idx = (ids == self.image_token).nonzero()[0]
            for j, idx in enumerate(image_idx):
                idx += j * (num_image_tokens - 1)  # Correct for previous insertions
                ids = torch.cat((ids[:idx], repeated_image, ids[idx + 1 :]))
                lab = torch.cat((lab[:idx], repeated_labels, lab[idx + 1 :]))
            input_ids[i] = ids
            labels[i] = lab

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        pixel_values = torch.stack(pixel_values)

        ret_dict = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(
                self.processor.tokenizer.pad_token_id
            ),  # Left from text-only
            pixel_values=pixel_values,
        )
        for key in instances[0].keys():
            if (
                key not in ret_dict
            ):  # This should add the image stuff to the return dict
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
