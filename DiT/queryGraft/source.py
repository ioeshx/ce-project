"""Token groups and masked source-only attention controls."""
from __future__ import annotations

import torch
import torch.nn.functional as F


SOURCE_MODES = ("anchor", "joint", "logits", "values", "extrapolate")


def text_message_attention(query, key, value, attention_mask, *, text_count, flux, **controls):
    """Read text-value contributions with the native joint softmax denominator.

    Inputs contain only native image/prompt tokens, with concept tokens removed.
    Source-only logit/value controls use the same operator as the full attention.
    """
    image_slice = slice(text_count, None) if flux else slice(None, key.shape[2] - text_count)
    text_values = value.clone()
    text_values[:, :, image_slice] = 0
    if controls:
        return controlled_attention(
            query, key, text_values, attention_mask, text_count=text_count, flux=flux,
            **controls)[:, :, image_slice]
    image_mask = attention_mask
    if image_mask is not None and image_mask.shape[-2] != 1:
        image_mask = image_mask[..., image_slice, :]
    return F.scaled_dot_product_attention(
        query[:, :, image_slice], key, text_values, attn_mask=image_mask, dropout_p=0.0)


def joint_prompts(prompts, anchor):
    return [f"{prompt.rstrip()}, {anchor}" for prompt in prompts]


def token_groups(tokenizer, prompts, joint, length):
    """Locate appended anchor content in the native, jointly encoded sequence.

    Tokenize the prefix with the same separator to support slow CLIP/T5
    tokenizers without offset mappings. Special/padding tokens belong to neither
    group. A boundary token spanning the append point belongs to the anchor.
    """
    encoded = tokenizer(joint, padding="max_length", max_length=length, truncation=True,
                        return_special_tokens_mask=True, return_tensors="pt")
    prefixes = tokenizer([f"{prompt.rstrip()}, " for prompt in prompts],
                         add_special_tokens=False)["input_ids"]
    content = encoded["attention_mask"].bool() & ~encoded["special_tokens_mask"].bool()
    user, anchor = torch.zeros_like(content), torch.zeros_like(content)
    for row, prefix in enumerate(prefixes):
        positions = content[row].nonzero(as_tuple=True)[0]
        ids = encoded["input_ids"][row, positions].tolist()
        shared = 0
        for actual, expected in zip(ids, prefix):
            if actual != expected:
                break
            shared += 1
        user[row, positions[:shared]] = True
        anchor[row, positions[shared:]] = True
    return user, anchor


def controlled_attention(query, key, value, attention_mask, *, text_count, flux,
                         user_tokens, anchor_tokens, region, mode, user_weight, anchor_weight):
    """Keep native text queries and unselected image queries unchanged.

    Logit bias changes softmax probabilities. Value scaling uses the identical
    Q/K and attention mask, so its current-layer probabilities remain native.
    The second SDPA call only evaluates image queries and avoids a full N x N
    bias allocation for spatial gating.
    """
    native = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)
    if user_weight == 1 and anchor_weight == 1:
        return native
    image_slice = slice(text_count, None) if flux else slice(None, query.shape[2] - text_count)
    image_query = query[:, :, image_slice]
    weights = torch.ones(query.shape[0], key.shape[2], device=query.device, dtype=torch.float32)
    text_weights = torch.ones_like(user_tokens, device=query.device, dtype=torch.float32)
    text_weights.masked_fill_(user_tokens.to(query.device), user_weight)
    text_weights.masked_fill_(anchor_tokens.to(query.device), anchor_weight)
    text_slice = slice(None, text_count) if flux else slice(key.shape[2] - text_count, None)
    weights[:, text_slice] = text_weights
    image_mask = attention_mask
    if image_mask is not None and image_mask.shape[-2] != 1:
        image_mask = image_mask[..., image_slice, :]
    if mode == "logits":
        bias = weights.log()[:, None, None, :]
        if image_mask is not None:
            if image_mask.dtype == torch.bool:
                bias = bias.masked_fill(~image_mask, -torch.inf)
            else:
                bias = bias + image_mask
        changed = F.scaled_dot_product_attention(image_query, key, value, attn_mask=bias.to(query.dtype))
    else:
        scaled_value = value * weights[:, None, :, None].to(value.dtype)
        changed = F.scaled_dot_product_attention(image_query, key, scaled_value, attn_mask=image_mask)
    result = native.clone()
    result[:, :, image_slice] = torch.where(region.to(query.device)[:, None, :, None],
                                            changed, native[:, :, image_slice])
    return result
