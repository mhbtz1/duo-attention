import torch
import random
from PIL import Image
import os
from duo_attn.data_vqa import VQADataset
from duo_attn.patch import load_full_attention_heads
from duo_attn.utils import (
    get_model,
    get_tokenizer,
    parse_args,
    to_device,
    load_attn_pattern,
    seed_everything,
    sparsify_attention_heads,
)
from duo_attn.patch import enable_duo_attention_eval
from utils import bench_func
from llava.model.builder import load_pretrained_model

from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    LlavaForConditionalGeneration,
    AutoProcessor,
)


if __name__ == "__main__":


    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    image_processor, tokenizer = processor.image_processor, processor.tokenizer

    ROOT_DIR = os.getenv("ROOT_DIR")
    if not ROOT_DIR:
        ROOT_DIR="/workspace/"
    questions_path=os.path.join(ROOT_DIR, "coco2014/v2_OpenEnded_mscoco_train2014_questions.json")
    images_path=os.path.join(ROOT_DIR, "coco2014/images/train2014")
    annotations_path=os.path.join(ROOT_DIR, "coco2014/v2_mscoco_train2014_annotations.json")

    
    vqa_dataset = VQADataset(processor, questions_path=questions_path, images_path=images_path, annotations_path=annotations_path)

    if args.config_name is not None:
        config = AutoConfig.from_pretrained(args.config_name)
    else:
        config = AutoConfig.from_pretrained(args.model_name)

    if args.rope_theta is not None:
        # print(f"Setting rope_theta from {config.rope_theta} to {args.rope_theta}")
        config.rope_theta = args.rope_theta

    login(token=os.getenv("HF_API_KEY"))


    with torch.no_grad():
        model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf",
            config=config,
            # Modified these per https://huggingface.co/llava-hf/llava-1.5-7b-hf
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2"
            # use_flash_attention_2=True,
            attn_implementation="eager",
        )


    #if model.config.model_type == "mistral":
    #    model.model._prepare_decoder_attention_mask = lambda *args, **kwargs: None
    #elif model.config.model_type == "llama":
    #    model.model._prepare_decoder_attention_mask = lambda *args, **kwargs: None
    #elif model.config.model_type == "llava":
    #    model.model.language_model._prepare_decoder_attention_mask = lambda *args, **kwargs: None
    
    model = to_device(model, args.device)

    if args.attn_load_dir is not None:
        full_attention_heads, sink_size, recent_size = load_attn_pattern(
            args.attn_load_dir
        )

        full_attention_heads, sparsity = sparsify_attention_heads(
            full_attention_heads, args.threshold, args.sparsity
        )
        enable_duo_attention_eval(
            model.language_model,
            full_attention_heads,
            128,
            256,
        )

    text = "a\n\n" * args.max_length

    random.seed(0)

    def sample_data():
        vqa_sample = vqa_dataset[random.randrange(len(vqa_dataset))]
        input_ids, pixel_values = vqa_sample["input_ids"], vqa_sample["pixel_values"]
        return input_ids.unsqueeze(0).to("cuda"), pixel_values.unsqueeze(0).to("cuda")

        input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")[
            :, :args.max_length - 1
        ]

        random_image = random.sample(os.listdir(os.path.join(os.path.join(os.environ["ROOT_DIR"], "augmented_images"))), 1)
        random_image = Image.open(random_image)
        pixel_values = image_processor(random_image, return_tensors="pt")

    # pre-filling
    torch.cuda.reset_peak_memory_stats()

    def func1():
        with torch.no_grad():
            input_ids, pixel_values = sample_data()
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=torch.ones_like(input_ids),
                past_key_values=None,
                use_cache=True,
            )

    # ctx_latency, ctx_memory = bench_func(func1, num_steps=20, num_warmup_steps=10)
    ctx_latency, ctx_memory = 0, 0

    with torch.no_grad():
        input_ids, pixel_values = sample_data()
        outputs = model(
            input_ids=input_ids,
            past_key_values=None,
            use_cache=True,
        )

    print(
        f"Peak memory usage in the pre-filling stage: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB"
    )
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]

    def func2():
        with torch.no_grad():
            _ = model(
                input_ids=pred_token_idx,
                past_key_values=past_key_values,
                use_cache=True,
            )

    gen_latency, gen_memory = bench_func(func1, num_steps=100, num_warmup_steps=10)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "benchmark_result.txt"), "w") as f:
            print(f"Average generation time: {gen_latency:.2f} ms", file=f)
            print(f"Peak generation memory usage: {gen_memory:.2f} MB", file=f)
            print(f"Average context time: {ctx_latency:.2f} ms", file=f)
            print(f"Peak context memory usage: {ctx_memory:.2f} MB", file=f)
            print(f"Model name: {args.model_name}", file=f)
            print(f"Context length: {args.max_length}", file=f)
            print(f"Sparsity: {sparsity}", file=f)
