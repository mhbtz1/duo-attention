import torch
from datasets import load_dataset
from dataclasses import dataclass
from typing import Sequence, Dict
from PIL import Image

import random
import os
import transformers
from torch.utils.data import Dataset, IterableDataset
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

def get_dataset(dataset_name, split="train", size=None):
    dataset = load_dataset("json", data_files=dataset_name, split=split)
    if size is not None:
        dataset = dataset.select(range(size))
    return dataset


class MenuPriceRetrievalDataset(Dataset):
    #FOODS = []
    
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
        #Get these from loading pretrained model
        tokenizer: transformers.PreTrainedTokenizer,
        image_processor,
        device,
        model_config,
        max_length=None,
        #passkey_length=32,
        max_price=10,
        num_items=5,
        needle="Remeber this sequence of words, it's the {ordinal_number} item on the menu: ",
        retrieval_question="Based on the content of the book, what is the price of my {meal}?\nPrice: ",
        prompt1="Here is a picture of my {meal}: " + DEFAULT_IMAGE_TOKEN + " Now, this is a very long story book: <book> ",
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
        self.tokenizer_encode = self._get_encoder(self.tokenizer)
        self.image_processor = image_processor
        self.device = device
        self.model_config = model_config

        #Populate available foods
        self.image_path = os.getenv("FOOD_IMAGES") 
        if not self.image_path:
            self.image_path = "/data/cb/dschaffe/vlm/llava/menu_images/MAFood121/images"
        subdirs = [d for d in os.listdir(self.image_path) if os.path.isdir(os.path.join(self.image_path, d))]
        self.FOODS = subdirs 
        breakpoint()

        self.max_length = (
            max_length if max_length is not None else tokenizer.model_max_length
        )
        self.max_depth_ratio = max_depth_ratio
        self.min_depth_ratio = min_depth_ratio
        self.context_lengths_num_intervals = context_lengths_num_intervals
        self.depth_ratio_num_intervals = depth_ratio_num_intervals
        self.pad_to_multiple_of = pad_to_multiple_of

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

        self.needle = needle
        self.needle_tokens_list = [
            self.tokenizer_encode(
                self.needle.format(ordinal_number=ordinal_number),
                add_special_tokens=False
            )
            for ordinal_number in self.ORDINAL_NUMBERS[: self.num_items]
        ]

        self.food_tokens_list = [
            self.tokenizer_encode(food.replace("_", " ") + " costs", 
                add_special_tokens=False)
            for food in self.FOODS
        ]

        self.haystack_tokens = self.tokenizer_encode(
            self.haystack, add_special_tokens=False
        )
        self.seperator_tokens = self.tokenizer_encode(
            seperator, add_special_tokens=False
        )
        self.prompt1_tokens_list = [
            self.tokenizer_encode(
                prompt1.format(meal=meal),
                add_special_tokens=False
            )
            for meal in self.MEALS
        ]   
        
        self.tokenizer_encode(prompt1, add_special_tokens=True)
        self.prompt2_tokens = self.tokenizer_encode(prompt2, add_special_tokens=False)


        self.retrieval_question_tokens_list = [
            self.tokenizer_encode(
                retrieval_question.format(meal=meal),
                add_special_tokens=False
            )
            for meal in self.MEALS#[: self.num_passkeys]
        ]       

        price_string = "{price} dollars"
        self.price_tokens_list = [
            self.tokenizer_encode(
                price_string.format(price=num), add_special_tokens=False
            )
            for num in self.NUMBERS[: self.max_price]
        ]

        #Different foods might be encoded by different #s of tokens, so can't pre-compute a fixed length
        #But, we can do some trimming
        #passkey_tokens = self.tokenizer_encode(passkey, add_special_tokens=False)
        other_input_len = (
            len(self.prompt1_tokens_list[0])
            + len(self.prompt2_tokens)
            + (
                len(self.seperator_tokens)
                + len(self.needle_tokens_list[0])
                + 1 #For minimum-size food
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
        menu_indices = random.sample(range(len(self.FOODS)), self.num_items) #better sampling?
        menu_prices= random.sample(range(self.max_price), self.num_items)
        #menu = zip(menu_items, menu_prices)
        order_idx = random.randrange(self.num_items)
        meal_idx = random.randrange(len(self.MEALS))

        food_tokens_list = [self.food_tokens_list[i] for i in menu_indices]
        price_tokens_list = [self.price_tokens_list[p] for p in menu_prices]
        
        context_tokens = self._insert_needle(context_length, depth_ratios, food_tokens_list, price_tokens_list)
        
        qa_tokens = torch.cat((self.retrieval_question_tokens_list[meal_idx], price_tokens_list[order_idx]))

        context_tokens = torch.cat((self.prompt1_tokens_list[meal_idx], context_tokens))

        #Trim a bit more if needed
        if len(context_tokens) + len(qa_tokens) + len(self.prompt2_tokens) > self.context_length_max - self.buffer_size:
            context_tokens = context_tokens[: self.context_length_max - self.buffer_size]
        # pad to multiple of 16 -
        if len(context_tokens) + len(qa_tokens) + len(self.prompt2_tokens) % self.pad_to_multiple_of != 0:
            pad_len = (
                self.pad_to_multiple_of
                - (len(context_tokens) + len(qa_tokens) + len(self.prompt2_tokens)) % self.pad_to_multiple_of
            )
            context_tokens = torch.cat((context_tokens, self.haystack_tokens[-pad_len:]))

        context_tokens = torch.cat((context_tokens, self.prompt2_tokens))
         
        input_ids = torch.cat((context_tokens, qa_tokens))

        assert input_ids.size(0) % self.pad_to_multiple_of == 0

        labels = torch.cat((torch.tensor([-100] * len(context_tokens)), qa_tokens))
        input_ids = input_ids.unsqueeze(0).to(self.device)
        labels = labels.to(self.device)
        
        image_class=self.FOODS[menu_indices[order_idx]]
        image_file = self._choose_image(image_class)
        image = Image.open(image_file).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)
        if type(image_tensor) is list:
            image_tensor = [image.to(self.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(self.device, dtype=torch.float16)

        breakpoint()
        return dict(input_ids=input_ids, labels=labels, image_tensor=image_tensor, image_size=image.size)

    def _choose_image(self, subdir):
        image_path = os.path.join(self.image_path, subdir)
        files = [os.path.join(image_path, f) for f in os.listdir(image_path)] #if os.path.isfile(os.path.join(image_path, f))]
        image_files = [f for f in files if os.path.splitext(f)[1].lower() in  (".jpg", ".jpeg")]
        return random.choice(image_files)



    def __len__(self):
        return len(self.context_length_intervals)

    def _trim(self, context, context_length):
        tokens = self.tokenizer_encode(context, add_special_tokens=False)
        if len(tokens) > context_length:
            context = self.tokenizer.decode(tokens[:context_length])
        return context
    
    def _get_token_nums(self, context):
        return len(self.tokenizer_encode(context))

    def _insert_needle(self, context_length, depth_ratios, food_tokens_list, price_tokens_list): #Updated
        haystack_tokens = self.haystack_tokens[:context_length]

        context = None #Really should be empty tensor
        last_insertion_point = 0

        for i, (depth_ratio, food_token, price_tokens) in enumerate(
            zip(depth_ratios, food_tokens_list, price_tokens_list)
        ):
            insertion_point = int(len(haystack_tokens) * depth_ratio)

            needle_tokens = torch.cat((self.needle_tokens_list[i], food_token, price_tokens))

            if context is None: #TODO: improve
                context = torch.cat((
                    haystack_tokens[last_insertion_point:insertion_point],
                    self.seperator_tokens,
                    needle_tokens,
                    self.seperator_tokens
                ))
            else:
                context = torch.cat((
                    context,
                    haystack_tokens[last_insertion_point:insertion_point],
                    self.seperator_tokens,
                    needle_tokens,
                    self.seperator_tokens
                ))
            last_insertion_point = insertion_point

        context = torch.cat((context, haystack_tokens[last_insertion_point:]))

        return context

    def _get_encoder(self, tokenizer):
        def f(prompt, add_special_tokens=False): #a_s_t ignored
            tokens = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt') #.unsqueeze(0).to(model.device)
            return tokens[1:-1] #Remove extra start/stop tokens
        return f
    

