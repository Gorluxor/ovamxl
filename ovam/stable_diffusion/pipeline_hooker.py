from typing import TYPE_CHECKING, Optional, Tuple, Union

from ..base.store_hooker import StoreHiddenStatesHooker
from ..base.pipeline_hooker import PipelineHooker
from .block_hooker import CrossAttentionHooker
from .daam_module import StableDiffusionDAAM, DAAMModule, StableDiffusionXLDAAM
from .locator import UNetCrossAttentionLocator
from ..base.hooker import ObjectHooker, ModuleType

if TYPE_CHECKING:
    from diffusers import (
        StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
    )
    from ..utils.attention_ops import ActivationTypeVar, AggregationTypeVar


class StableDiffusionHooker(PipelineHooker):
    """DAAMHooker used to save the hidden states of the cross attention blocks
    during the fordward passes of the Stable Diffusion UNET.

    Arguments
    ---------
    pipeline: Union[StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline]
        Pipeline to be hooked
    restrict_block_index: Optional[Iterable[int]]
        Restrict the hooking to the blocks with the given indices. If None, all
        the blocks are hooked.
    locate_middle_block: bool, default=False
        If True, the middle block is located and hooked. This block is not
        hooked by default because its spatial size is too small and the
        attention maps are not very informative.

    Attributes
    ----------
    module: List[ObjectHooker]
        List of ObjectHooker for the cross attention blocks
    locator: UNetCrossAttentionLocator
        Locator of the cross attention blocks

    Note
    ----
    This class is based on the class DiffusionHeatMapHooker of the
    daam.trace module.
    """

    def __init__(
        self,
        pipeline: Union["StableDiffusionPipeline", "StableDiffusionImg2ImgPipeline", "StableDiffusionXLPipeline", "StableDiffusionXLImg2ImgPipeline"],
        locate_middle_block: bool = False,
        block_hooker_kwargs: dict = {},
        locator_kwargs: dict = {},
        daam_module_class: type = StableDiffusionDAAM, # added to support SDXL
        block_hooker_class: type = CrossAttentionHooker, # future proofing
        locator_hooker_class: type = UNetCrossAttentionLocator, # future proofing
        store_hidden_states_hooker_class: type = StoreHiddenStatesHooker, # future proofing
    ):
        self.store_hidden_states_hooker_class = store_hidden_states_hooker_class
        assert issubclass(store_hidden_states_hooker_class, ObjectHooker), f"{store_hidden_states_hooker_class} is not a subclass of ObjectHooker"
        assert issubclass(daam_module_class, DAAMModule), f"{daam_module_class} is not a subclass of DAAMModule"
        super().__init__(
            pipeline,
            locator=locator_hooker_class(
                locate_middle_block=locate_middle_block,
                **locator_kwargs,
            ),
            block_hooker_class=block_hooker_class,
            daam_module_class=daam_module_class,
            block_hooker_kwargs=block_hooker_kwargs,
        )

    def _register_extra_hooks(self):
        """Hook the encode prompt and forward functions along the forward pass"""
        self.register_hook(
            self.store_hidden_states_hooker_class(
                module=self.pipeline.image_processor,
                parent_trace=self,
                function_patched="postprocess",
            )
        )

    @property
    def cross_attention_hookers(self):
        """Returns the cross attention blockers"""
        return self.module[:-1]

    def get_ovam_callable(
        self,
        heads_epochs_activation: "ActivationTypeVar" = "token_softmax",  # "linear" for linear daam,
        heads_epochs_aggregation: "AggregationTypeVar" = "sum",
        heads_activation: "ActivationTypeVar" = "linear",
        heads_aggregation: "AggregationTypeVar" = "sum",
        block_interpolation_mode: str = "bilinear",
        blocks_activation: "ActivationTypeVar" = "linear",
        heatmaps_activation: Optional["ActivationTypeVar"] = None,
        heatmaps_aggregation: "AggregationTypeVar" = "sum",
        expand_size: Optional[Tuple[int, int]] = None,
        expand_interpolation_mode: str = "bilinear",
        block_kwargs: dict = {},
        module_kwargs: dict = {},
    ) -> Union["StableDiffusionDAAM", "StableDiffusionXLDAAM"]:
        """
        Buld a OVAM module with the current hidden states.
        This module can be evaluated to obtain the attention maps
        for any arbitrary text embedding (sentence) or token (word).

        Arguments
        ---------
        heads_epochs_activation: str or Callable, default="token_softmax"
            The activation function applied to the attentions of each attention
            head of each block of each epoch. By default the attention heads
            of the epoch are softmaxed in the token dimension. To execute the
            linear damm version set this parameter to "linear". Activation
            used in a tensor of shape (n_epochs, heads, n_tokens,
            latent_size / factor, latent_size / factor) where factor
            depends on the block.
        heads_epochs_aggregation: str or Callable, default="sum"
            The aggregation function applied to aggregate the attention
            heads across all epochs. By default the epochs are summed.
            Collapses the `n_epochs` dimension of a tensor with shape
            (n_epochs, heads, n_tokens, latent_size / factor, latent_size / factor)
        heads_activation : str or Callable, default="linear"
            The activation function to apply to each of the attention blocks
            after aggregate their epochs. By default the attention blocks are
            not activated. Recieves a tensor of shape (heads, n_tokens,
            latent_size / factor, latent_size / factor) where factor depends
            on the block.
        heads_aggregation : str or Callable, default="sum"
            Aggregation function applied to the attention heads of each block.
            By default the attention heads are summed. Collapses the `heads`
            dimension of a tensor with shape (heads, n_tokens, latent_size / factor,
            latent_size / factor).
        block_interpolation_mode : str, default="bilinear"
            The interpolation mode to use when expanding the attention maps
            of each block to normalize all sizes to the original latent size.
        blocks_activation : str or Callable, default="linear"
            The activation function to apply to each of the attention blocks
            after the aggregation of the attention heads. By default the
            attention blocks are not activated.
        """

        var_module_kwargs = {
            "heatmaps_activation": heatmaps_activation,
            "heatmaps_aggregation": heatmaps_aggregation,
            "expand_size": expand_size,
            "expand_interpolation_mode": expand_interpolation_mode,
            "block_interpolation_mode": block_interpolation_mode,
        }
        var_module_kwargs.update(module_kwargs)

        var_block_kwargs = {
            "heads_activation": heads_activation,
            "blocks_activation": blocks_activation,
            "heads_epochs_activation": heads_epochs_activation,
            "heads_aggregation": heads_aggregation,
            "heads_epochs_aggregation": heads_epochs_aggregation,
        }
        var_block_kwargs.update(block_kwargs)

        return super().daam(
            module_kwargs=var_module_kwargs,
            block_kwargs=var_block_kwargs,
        )
