from typing import Optional, Union

import torch
from diffusers import StableDiffusionXLPipeline

__all__ = ["encode_text", "encode_text_sdxl", "full_encode_sdxl"]


@torch.no_grad()
def encode_text(
    tokenizer,
    text_encoder,
    text: str,
    context_sentence: Optional[str] = None,
    remove_special_tokens: bool = True,
    padding=False,
) -> "torch.Tensor":
    """Encode a text into a sequence of tokens based on the stable diffusion
    text encoder and tokenizer.

    Arguments
    ---------
    text : str
        The text to encode.
    context_sentence : str
        The context sentence to encode. If None, the text is used as context.

    Returns
    -------
    torch.Tensor
        The encoded text. Shape: (tokens, embedding_size)
    """

    device = text_encoder.device

    if context_sentence is None:
        context_sentence = text

    tokens = tokenizer(context_sentence, padding=padding, return_tensors="pt")

    text_embeddings = text_encoder(
        tokens.input_ids.to(device), attention_mask=tokens.attention_mask.to(device)
    )
    text_embeddings = text_embeddings[0][0]  # Discard hidden states

    # Discard special tokens (<SOT>, <EOT>)
    if remove_special_tokens:
        text_embeddings = text_embeddings[1:-1, :]

    # Discard tokens that are not in the text
    if text != context_sentence:
        text_input_ids = (
            tokenizer(text, padding=False, return_tensors="pt")
            .input_ids.numpy()
            .flatten()
        )
        token_input_ids = tokens.input_ids.numpy().flatten()
        if remove_special_tokens:
            token_input_ids = token_input_ids[1:-1]
            text_input_ids = text_input_ids[1:-1]

        selected_ids = [
            token_input_id in text_input_ids for token_input_id in token_input_ids
        ]
        text_embeddings = text_embeddings[selected_ids, :]

    return text_embeddings


@torch.no_grad()
def encode_text_sdxl(
    tokenizer,
    text_encoder,
    text: str,
    context_sentence: Optional[str] = None,
    remove_special_tokens: bool = True,
    padding=False,
) -> "torch.Tensor":
    """Encode a text into a sequence of tokens based on the stable diffusion
    text encoder and tokenizer.

    Arguments
    ---------
    text : str
        The text to encode.
    context_sentence : str
        The context sentence to encode. If None, the text is used as context.

    Returns
    -------
    torch.Tensor
        The encoded text. Shape: (tokens, embedding_size)
    """

    device = text_encoder.device

    if context_sentence is None:
        context_sentence = text

    tokens = tokenizer(context_sentence, padding=padding, return_tensors="pt")

    text_embeddings = text_encoder(
        tokens.input_ids.to(device),
        attention_mask=tokens.attention_mask.to(device),
        output_hidden_states=True,
        return_dict=True,
    )
    pooled_prompt_embeds = text_embeddings[0]
    text_embeddings = text_embeddings.hidden_states[-2]  # Discard hidden states

    if (
        text_embeddings.ndim == 3
    ):  # one of the tokenizers returns B the other one does not
        text_embeddings = text_embeddings[0, ...]
    # Discard special tokens (<SOT>, <EOT>)
    if remove_special_tokens:
        text_embeddings = text_embeddings[1:-1, :]

    # Discard tokens that are not in the text
    if text != context_sentence:
        text_input_ids = (
            tokenizer(text, padding=False, return_tensors="pt")
            .input_ids.numpy()
            .flatten()
        )
        token_input_ids = tokens.input_ids.numpy().flatten()
        if remove_special_tokens:
            token_input_ids = token_input_ids[1:-1]
            text_input_ids = text_input_ids[1:-1]

        selected_ids = [
            token_input_id in text_input_ids for token_input_id in token_input_ids
        ]
        text_embeddings = text_embeddings[selected_ids, :]

    return pooled_prompt_embeds, text_embeddings


def full_encode_sdxl(
    model: "StableDiffusionXLPipeline",  # In theory its also StableDiffusionXLDAAM, don't wanna reference here
    text: str,
    context_sentence: Optional[str] = None,
    remove_special_tokens: bool = True,
    padding=False,
    return_pool: bool = False,
) -> torch.Tensor:

    # We check that we both have text_encoder and text_encoder_2
    assert hasattr(model, "text_encoder_2") and hasattr(
        model, "text_encoder"
    ), f"Model must have text_encoder_2 and text_encoder"

    _, text_embd = encode_text_sdxl(
        tokenizer=model.tokenizer,
        text_encoder=model.text_encoder,
        text=text,
        context_sentence=context_sentence,
        remove_special_tokens=remove_special_tokens,
        padding=padding,
    )

    pooled, text_embd_2 = encode_text_sdxl(
        tokenizer=model.tokenizer_2,
        text_encoder=model.text_encoder_2,
        text=text,
        context_sentence=context_sentence,
        remove_special_tokens=remove_special_tokens,
        padding=padding,
    )

    text_embd = torch.concat([text_embd, text_embd_2], dim=-1)

    if not return_pool:
        return text_embd
    # Technically we use this in forward pass conditioning of SDXL
    # OVAM just ignores it as it uses the original conditioning
    # Added here just in case
    return text_embd, pooled
