def set_llava_full_attention_heads(model):
    for layer_idx, layer in enumerate(model.model.layers):
        if not hasattr(params, "full_attention_heads"):
            continue    
        module = layer.self_attn
        module.full_attention_heads.data = full_attention_heads[layer_idx].to(
            module.full_attention_heads.device, module.full_attention_heads.dtype
        )