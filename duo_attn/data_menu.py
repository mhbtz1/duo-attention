import torch
from datasets import load_dataset
from dataclasses import dataclass
from typing import Sequence, Dict

import random
import transformers
from torch.utils.data import Dataset, IterableDataset


def get_dataset(dataset_name, split="train", size=None):
    dataset = load_dataset("json", data_files=dataset_name, split=split)
    if size is not None:
        dataset = dataset.select(range(size))
    return dataset


class MenuPriceRetrievalDataset(Dataset):
    FOODS = [
        "alpha",
        "bravo",
        "charlie",
        "delta",
        "echo",
        "foxtrot",
        "golf",
        "hotel",
        "india",
        "juliet",
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

    MEALS = ["breakfast", "lunch", "dinner", "snack"]

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

    NUMBERS = [
        "one", 
        "two", 
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen", 
        "fifteen", 
        "sixteen",
        "seventeen", 
        "eighteen", 
        "nineteen", 
        "twenty"
    ]

    def __init__(
        self,
        haystack_dataset,
        tokenizer: transformers.PreTrainedTokenizer, #Need to set?
        max_length=None,
        #passkey_length=32,
        max_price=10,
        num_items=5,
        needle="Remeber this sequence of words, it's the {ordinal_number} item on the menu: a {food} costs ",
        retrieval_question="Based on the content of the book, what is the price of my {meal}?\nPrice: ",
        prompt1="<s> Here is a picture of my {meal}: <image>. Now, this is a very long story book: <book> ",
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
        super(MenuPriceRetrievalDataset, self).__init__()

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

        self.context_length_intervals = torch.linspace(
            self.context_length_min,
            self.context_length_max,
            context_lengths_num_intervals,
            dtype=torch.int,
        )

        self.depth_ratio_intervals = torch.linspace(
            min_depth_ratio, max_depth_ratio, depth_ratio_num_intervals
        )

        self.num_items = num_items
        self.max_price = max_price
        self.buffer_size = buffer_size

        self.haystack = ""

        for sample in haystack_dataset["text"]:
            if self._get_token_nums(self.haystack) >= self.context_length_max:
                break
            self.haystack += sample

        self.haystack = self._trim(self.haystack, self.context_length_max)

        self.haystack_tokens = self.tokenizer.encode(
            self.haystack, add_special_tokens=False
        )
        self.seperator_tokens = self.tokenizer.encode(
            seperator, add_special_tokens=False
        )
        self.prompt1_tokens_list = [
            self.tokenizer.encode(
                prompt1.format(meal=meal),
                add_special_tokens=False
            )
            for meal in self.MEALS
        ]   
        
        self.tokenizer.encode(prompt1, add_special_tokens=True)
        self.prompt2_tokens = self.tokenizer.encode(prompt2, add_special_tokens=False)


        self.retrieval_question_tokens_list = [
            self.tokenizer.encode(
                retrieval_question.format(meal=meal),
                add_special_tokens=False
            )
            for meal in self.MEALS#[: self.num_passkeys]
        ]       

        self.price_tokens_list = [
            self.tokenizer.encode(num + " dollars", add_special_tokens=False)
            for num in self.NUMBERS[: self.max_price]
        ]

        #Different foods might be encoded by different #s of tokens, so can't pre-compute a fixed length
        #But, we can do some trimming
        #passkey_tokens = self.tokenizer.encode(passkey, add_special_tokens=False)
        needle_tokens_ex = self.tokenizer.encode(self.needle.format(ordinal_number=self.ORDINAL_NUMBERS[0], food="a"), add_special_tokens=False)

        other_input_len = (
            len(self.prompt1_tokens_list[0])
            + len(self.prompt2_tokens)
            + (
                len(self.seperator_tokens)
                + len(needle_tokens_ex)
                + len(self.price_tokens_list[0]) * 2
                + len(self.seperator_tokens)
                + len(self.retrieval_question_tokens_list[0])
            )
            * self.num_items
        )
        if (
            len(self.haystack_tokens) + other_input_len
            > self.context_length_max - self.buffer_size
        ):
            self.haystack_tokens = self.haystack_tokens[
                : self.context_length_max - self.buffer_size - other_input_len
            ]

        


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        context_length = self.context_length_intervals[i]
        # randomly sample self.num_items depth ratios in self.depth_ratio_intervals
        depth_ratios = (
            self.depth_ratio_intervals[
                torch.randperm(self.depth_ratio_num_intervals)[: self.num_items]
            ]
            .sort()
            .values
        )

        #Generate a "menu"
        menu_items = random.sample(self.FOODS, self.num_items)
        menu_prices= random.sample(range(1,self.max_price+1), self.num_items)
        #menu = zip(menu_items, menu_prices)
        order_idx = random.randint(self.num_items)
        meal_idx = random.randint(len(self.MEALS))


        needle_tokens_list = [
            self.tokenizer.encode(self.needle.format(
                ordinal_number=self.ORDINAL_NUMBERS[i], food=item),
                add_special_tokens=False)
            for i,item in enumerate(menu_items)
        ]

        price_tokens_list = [self.price_tokens_list[p] for p in menu_prices]
        
        context_tokens = self._insert_needle(context_length, depth_ratios, needle_tokens_list, price_tokens_list)
        
        qa_tokens = self.retrieval_question_tokens[meal_idx] + price_tokens_list[order_idx] 

        context_tokens = self.prompt1_tokens[meal_idx] + context_tokens

        #Trim a bit more if needed
        if len(context_tokens) + len(qa_tokens) + len(self.prompt2_tokens) > self.context_length_max - self.buffer_size:
            context_tokens = context_tokens[: self.context_length_max - self.buffer_size]
        # pad to multiple of 16 -
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
        #TODO: select and return an appropriate image
        return dict(input_ids=input_ids, labels=labels, image_class=menu_items[order_idx])
          

    def __len__(self):
        return len(self.context_length_intervals)

    def _get_token_nums(self, context):
        return len(self.tokenizer.encode(context))

    def _insert_needle(self, context_length, depth_ratios, needle_tokens_list, price_tokens_list): #Updated
        haystack_tokens = self.haystack_tokens[:context_length]

        context = []
        last_insertion_point = 0

        for depth_ratio, needle_tokens, price_tokens in zip(
            depth_ratios, needle_tokens_list, price_tokens_list):
            insertion_point = int(len(haystack_tokens) * depth_ratio)

            insert_tokens = needle_tokens + price_tokens

            context += (
                haystack_tokens[last_insertion_point:insertion_point]
                + self.seperator_tokens
                + insert_tokens
                + self.seperator_tokens
            )
            last_insertion_point = insertion_point

        context += haystack_tokens[last_insertion_point:]

        return context



@dataclass
class DataCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            ([instance["tokenized_input"][key] for instance in instances] + instance["response"]) for key in ("input_ids", "pixel_values", "attention_mask")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

        ret_dict = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        for key in instances[0].keys():
            if key not in ret_dict:
                ret_dict[key] = torch.stack([instance[key] for instance in instances])
        return ret_dict


def get_supervised_dataloader(
    dataset, tokenizer, batch_size, num_workers=4, shuffle=True, sampler=None
):
    collator = DataCollator(tokenizer)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collator,
        shuffle=None if sampler is not None else shuffle,
        sampler=sampler,
    )
    return dataloader
