from typing import TYPE_CHECKING, Callable, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import einops
from ..base.daam_module import DAAMModule
from ..utils.attention_ops import apply_activation, apply_aggregation
from ..utils.text_encoding import encode_text_sdxl, full_encode_sdxl
if TYPE_CHECKING:
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline

    from .daam_block import CrossAttentionDAAMBlock


def pack_interpolate_unpack(att, size, interpolation_mode):
    if att.shape[-2:] == size:
        # If the attention has the same size as the latent size, do nothing
        return att
    att, ps = einops.pack(att, "* c h w")
    att = F.interpolate(
        att,
        size=size,
        mode=interpolation_mode,
    )
    return torch.stack(einops.unpack(att, ps, "* c h w"))

class StableDiffusionDAAM(DAAMModule):
    """Generic DAAMModule implementation for Stable Diffusion models. It is used to
    save the hidden states of the cross attention blocks and to build a callable
    DAAM function.

    Arguments
    ---------
    blocks : List[DAAMBlock]
        The list of DAAM blocks to use.
    tokenizer : Callable
        The tokenizer to use to encode the text.
    text_encoder : Callable
        The text encoder to use to encode the text.
    heatmaps_activation : str or Callable
        The activation function to apply to the heatmaps. If a string, it must be one of
        "relu", "sigmoid", "tanh" or None (identity function).
    heatmaps_aggregation : str or Callable
        The aggregation function to apply to the heatmaps. If a string, it must be one of
        "mean", "sum", "max" or None (identity function).
    expand_size : Tuple[int, int]
        The size to expand the heatmaps to. If None, the heatmaps are not expanded.
    expand_interpolation_mode : str
        The interpolation mode to use when expanding the heatmaps. Default: "bilinear"

    """

    def __init__(
        self,
        blocks: List["CrossAttentionDAAMBlock"],
        pipeline: "StableDiffusionPipeline",
        heatmaps_activation: Optional[
            Union[Literal["sigmoid", "tanh", "relu"], "Callable"]
        ] = None,
        heatmaps_aggregation: Union[Literal["mean", "sum"], "Callable"] = "mean",
        block_latent_size: Optional[Tuple[int, int]] = None,
        block_interpolation_mode: str = "bilinear",
        expand_size: Optional[Tuple[int, int]] = None,
        expand_interpolation_mode: str = "bilinear",
    ):
        super().__init__(blocks)

        self.block_latent_size = block_latent_size
        self.block_interpolation_mode = block_interpolation_mode
        self.tokenizer = pipeline.tokenizer
        self.text_encoder = pipeline.text_encoder
        self.heatmaps_activation = heatmaps_activation
        self.expand_size = expand_size
        self.expand_interpolation_mode = expand_interpolation_mode
        self.heatmaps_aggregation = heatmaps_aggregation

    def forward(
        self, x: Union["torch.Tensor", str], remove_special_tokens: bool = False
    ) -> "torch.Tensor":
        """Compute the attention for a given input x.

        Arguments
        ---------
        x : torch.Tensor or str
            The input to compute the attention for. If a string, it is encoded
            using the text encoder.

        Returns
        -------
        torch.Tensor
            The attention heatmaps. Shape: (n_images, n_tokens, block_latent_size[0], block_latent_size[1])
        """
        if isinstance(x, str):
            # Encode text
            x = self.encode_text(x, remove_special_tokens=remove_special_tokens)

        attention = []
        for block in self.blocks.values():
            attention.append(block.forward(x))

        # shape: (n_blocks, n_images, n_tokens, block_latent_size[0], block_latent_size[1])
        # By default block_latent_size = (64, 64)
        # Infer latent size as the maximum size of the blocks
        if self.block_latent_size is None:
            # Get the latent size from the first block
            a, b = 0, 0
            for att in attention:
                a = max(a, att.shape[-2])
                b = max(b, att.shape[-1])
            block_latent_size = (a, b)
        else:
            block_latent_size = self.block_latent_size

        # Interpolate all attentions to the same size
        # Note(Alex): Added einops for interpolation across unaggragated attn
        attentions = []
        for att in attention:
            attentions.append(pack_interpolate_unpack(att, block_latent_size, self.block_interpolation_mode))
        # Remove reference to attention without interpolation
        del attention
        # check if all shapes match
        _cond = all([att.shape == attentions[0].shape for att in attentions])
        if not _cond:
            attentions = [apply_activation(att, self.heatmaps_activation) for att in attentions]
            attentions = [pack_interpolate_unpack(att, self.expand_size, self.expand_interpolation_mode)
                           for att in attentions]
            return attentions
        attentions = torch.stack(attentions, dim=0)
        attentions = apply_aggregation(
            attentions, self.heatmaps_aggregation
        )  # Collapse dim 0

        # Shape (n_images, n_tokens, block_latent_size[0], block_latent_size[1])
        attentions = apply_activation(attentions, self.heatmaps_activation)

        if self.expand_size is not None:
            attentions = pack_interpolate_unpack(attentions,
                                                 self.expand_size,
                                                 self.expand_interpolation_mode)

        return attentions

class StableDiffusionXLDAAM(StableDiffusionDAAM):
    """Generic DAAMModule implementation for Stable Diffusion models. It is used to
    save the hidden states of the cross attention blocks and to build a callable
    DAAM function.

    Arguments
    ---------
    blocks : List[DAAMBlock]
        The list of DAAM blocks to use.
    tokenizer : Callable
        The tokenizer to use to encode the text.
    text_encoder : Callable
        The text encoder to use to encode the text.
    heatmaps_activation : str or Callable
        The activation function to apply to the heatmaps. If a string, it must be one of
        "relu", "sigmoid", "tanh" or None (identity function).
    heatmaps_aggregation : str or Callable
        The aggregation function to apply to the heatmaps. If a string, it must be one of
        "mean", "sum", "max" or None (identity function).
    expand_size : Tuple[int, int]
        The size to expand the heatmaps to. If None, the heatmaps are not expanded.
    expand_interpolation_mode : str
        The interpolation mode to use when expanding the heatmaps. Default: "bilinear"

    """

    def __init__(
        self,
        blocks: List["CrossAttentionDAAMBlock"],
        pipeline: Union["StableDiffusionPipeline", "StableDiffusionXLPipeline", "StableDiffusionImg2ImgPipeline", "StableDiffusionXLImg2ImgPipeline"] ,
        heatmaps_activation: Optional[
            Union[Literal["sigmoid", "tanh", "relu"], "Callable"]
        ] = None,
        heatmaps_aggregation: Union[Literal["mean", "sum"], "Callable"] = "mean",
        block_latent_size: Optional[Tuple[int, int]] = None,
        block_interpolation_mode: str = "bilinear",
        expand_size: Optional[Tuple[int, int]] = None,
        expand_interpolation_mode: str = "bilinear",
    ):
        super().__init__(
            blocks,
            pipeline,
            heatmaps_activation,
            heatmaps_aggregation,
            block_latent_size,
            block_interpolation_mode,
            expand_size,
            expand_interpolation_mode,
        )
        assert hasattr(pipeline, "tokenizer_2") and hasattr(pipeline, "text_encoder_2"), f"Pipeline must have tokenizer_2 and text_encoder_2"
        assert pipeline.tokenizer_2 is not None and pipeline.text_encoder_2 is not None, f"Pipeline must have tokenizer_2 and text_encoder_2 not None"
        self.tokenizer_2 = pipeline.tokenizer_2
        self.text_encoder_2 = pipeline.text_encoder_2

    @torch.no_grad()
    def encode_text(self, text: str, context_sentence: Optional[str] = None, remove_special_tokens: TYPE_CHECKING = True, padding=False) -> torch.Tensor:

        return full_encode_sdxl(self, text, context_sentence, remove_special_tokens, padding)