"""
Implementation of the module locator for the cross-attention modules in
Stable Diffusion UNet (CrossAttn2DConditionModel).

Based on the original implementation from
What the DAAM:  Interpreting Stable Diffusion Using Cross Attention
(Tang et al., ACL 2023)

"""
import itertools
from typing import TYPE_CHECKING, Dict

from ..base.locator import ModuleLocator

if TYPE_CHECKING:
    from diffusers import StableDiffusionPipeline
    from diffusers.models.attention import CrossAttention

__all__ = ["UNetCrossAttentionLocator", "SlimeAttentionLocator"]

def get_attn_modules(model) -> list:
    """
    Retrieve all attention modules matching the naming convention.

    The regex captures modules with names like:
        up_blocks.<num>.attentions.<num>.transformer_blocks.<num>.attn<digit>
        down_blocks.<num>.attentions.<num>.transformer_blocks.<num>.attn<digit>
        mid_block.<num>.attentions.<num>.transformer_blocks.<num>.attn<digit>

    Returns:
        List of tuples (module_name, module)
    """
    regex_pattern = r"^(up_blocks|down_blocks|mid_block)\.\d\.attentions\.\d\.transformer_blocks\.\d\.attn[\d]$"
    return [(name, module) for name, module in model.named_modules() if re.match(regex_pattern, name)]

def format_block_name(module_name: str) -> str:
    """
    Compose a block name from the module name, following the original logic.

    For example, converts:
        "up_blocks.2.attentions.0.transformer_blocks.0.attn2"
    to:
        "up-2-attentions-1-transformer-1-attn2"

    Returns:
        The formatted block name as a string.
    """
    parts = module_name.split(".")
    # Determine prefix: convert "up_blocks" -> "up", "down_blocks" -> "down", "mid_block" -> "mid"
    if parts[0] == "up_blocks":
        prefix = f"up-{parts[1]}"
    elif parts[0] == "down_blocks":
        prefix = f"down-{parts[1]}"
    elif parts[0] == "mid_block":
        prefix = "mid"
    else:
        prefix = parts[0]
    # parts[2] is "attentions" and parts[3] is its index (0-indexed);
    # parts[4] is "transformer_blocks" and parts[5] is its index (0-indexed);
    # parts[6] is the attention type (e.g. "attn1" or "attn2").
    attn_index = int(parts[3]) + 1
    transformer_index = int(parts[5]) + 1
    attn_type = parts[6]
    return f"{prefix}-attentions-{attn_index}-transformer-{transformer_index}-{attn_type}"

class UNetCrossAttentionLocator(ModuleLocator["CrossAttention"]):
    """
    Locate cross-attention modules in a UNet2DConditionModel.

    Arguments
    ---------
    restrict: bool
        If not None, only locate the cross-attention modules with the given indices.
    locate_middle_block: bool, default=False
        If True, the middle block is located and hooked. This block is not
        hooked by default because its spatial size is too small and the
        attention maps are not very informative.
    locate_attn1: bool, default=False
        If True, locate the first attention module in each cross-attention block.
    locate_attn2: bool, default=True
        If True, locate the second attention module in each cross-attention block.

    Note
    ----
    This class is based on the class UNetCrossAttentionLocator of the
    daam.trace module. The unique difference is that this class allows to
    locate the first or second attention modules separately.

    """

    def __init__(
        self,
        locate_attn1: bool = False,
        locate_attn2: bool = True,
        locate_middle_block: bool = False,
    ):
        super().__init__()
        self.locate_attn1 = locate_attn1
        self.locate_attn2 = locate_attn2
        self.locate_middle_block = locate_middle_block

    # def locate(self, pipe: "StableDiffusionPipeline") -> Dict[str, "CrossAttention"]:
    #     """
    #     Locate cross-attention modules in a UNet2DConditionModel using the index
    #     from get_attn_modules and compose block names accordingly.

    #     Args:
    #         pipe (StableDiffusionPipeline): The pipeline with a UNet containing
    #             cross-attention modules.

    #     Returns:
    #         Dict[str, CrossAttention]: A dictionary mapping formatted block names
    #             to the corresponding cross-attention modules.
    #     """
    #     model = pipe.unet
    #     blocks = {}
    #     registered_idx = []
    #     attn_modules = get_attn_modules(model)

    #     for i, (module_name, module) in enumerate(attn_modules):
    #         # Skip the middle block if not requested
    #         if module_name.startswith("mid_block") and not self.locate_middle_block:
    #             print(f"Skipped {module_name} (middle block disabled)")
    #             continue

    #         # Compose the block name using the original naming logic.
    #         block_name = format_block_name(module_name)

    #         # Decide whether to register the module based on the attention type.
    #         if block_name.endswith("attn1"):
    #             if not self.locate_attn1:
    #                 print(f"Skipped {module_name} (attn1 not requested)")
    #                 continue
    #         elif block_name.endswith("attn2"):
    #             if not self.locate_attn2:
    #                 print(f"Skipped {module_name} (attn2 not requested)")
    #                 continue

    #         blocks[block_name] = module
    #         registered_idx.append(i)
    #         print(f"Successfully located {module_name} as {block_name} with index {i}")

    #     # Log registration status for each module based on its index.
    #     for i, (module_name, _) in enumerate(attn_modules):
    #         if i in registered_idx:
    #             print(f"[REG][{i}] {module_name}")
    #         else:
    #             print(f"[UNREG][{i}] {module_name}")

    #     return blocks

    # def locate(self, pipe: "StableDiffusionPipeline") -> Dict[str, "CrossAttention"]:
    #     """
    #     Locate cross-attention modules in a UNet2DConditionModel and print the exact
    #     indexes used, similar to SlimeAttentionLocator.

    #     Args:
    #         pipe (StableDiffusionPipeline): The pipeline with a UNet containing
    #             cross-attention modules.

    #     Returns:
    #         Dict[str, CrossAttention]: A dictionary mapping formatted module names
    #             to the corresponding cross-attention module.
    #     """
    #     model = pipe.unet
    #     blocks = {}
    #     registered_idx = []

    #     # Retrieve attention modules using the standalone function
    #     attn_modules = get_attn_modules(model)
    #     for i, (name, module) in enumerate(attn_modules):
    #         # Skip the middle block if not requested
    #         if name.startswith("mid_block") and not self.locate_middle_block:
    #             print(f"Skipped {name} (middle block disabled)")
    #             continue

    #         # Register modules based on the attention type flags
    #         if name.endswith("attn1"):
    #             if self.locate_attn1:
    #                 block_name = name.replace(".", "-")
    #                 blocks[block_name] = module
    #                 registered_idx.append(i)
    #                 print(f"Successfully located {name} as attn1 with index {i}")
    #             else:
    #                 print(f"Skipped {name} (attn1 not requested)")
    #         elif name.endswith("attn2"):
    #             if self.locate_attn2:
    #                 block_name = name.replace(".", "-")
    #                 blocks[block_name] = module
    #                 registered_idx.append(i)
    #                 print(f"Successfully located {name} as attn2 with index {i}")
    #             else:
    #                 print(f"Skipped {name} (attn2 not requested)")

    #     # Log registration status for each attention module
    #     for i, (name, _) in enumerate(attn_modules):
    #         if i in registered_idx:
    #             print(f"[REG][{i}] {name}")
    #         else:
    #             print(f"[UNREG][{i}] {name}")

    #     return blocks


    def locate(self, pipe: "StableDiffusionPipeline") -> Dict[str, "CrossAttention"]:
        """
        Locate cross-attention modules in a UNet2DConditionModel.

        Args:
            pipe (`StableDiffusionPipeline`): The pipe with unet containing
            the cross-attention modules in.

        Returns:
            `LisDict[str, CrossAttention]`: The cross-attention modules.
        """
        model = pipe.unet
        blocks = {}
        up_names = [f"up-{j}" for j in range(1, len(model.up_blocks) + 1)]
        down_names = [f"down-{j}" for j in range(1, len(model.down_blocks), +1)]
        all_layer = []

        def _correct(name):
            # need to correct the number by -1 
            left, right = name.split('-')
            left = left.replace('up', 'up_blocks').replace('down', 'down_blocks').replace('mid', 'mid_blocks')
            return f"{left}.{int(right)-1}"
            

        for unet_block, name in itertools.chain(
            zip(model.up_blocks, up_names),
            zip(model.down_blocks, down_names),
            zip([model.mid_block], ["mid"]) if self.locate_middle_block else [],
        ):
            if "CrossAttn" in unet_block.__class__.__name__:
                for i, spatial_transformer in enumerate(unet_block.attentions):
                    for j, transformer_block in enumerate(
                        spatial_transformer.transformer_blocks
                    ):
                        ln = f"{_correct(name)}.attentions.{i}.transformer_blocks.{j}"
                        if self.locate_attn1:
                            block_name = (
                                f"{name}-attentions-{i+1}-transformer-{j+1}-attn1"
                            )
                            blocks[block_name] = transformer_block.attn1
                            all_layer.append(f"{ln}.attn1")
                            #import ipdb; ipdb.set_trace()
                        if self.locate_attn2:
                            block_name = (
                                f"{name}-attentions-{i+1}-transformer-{j+1}-attn2"
                            )
                            blocks[block_name] = transformer_block.attn2
                            # import ipdb; ipdb.set_trace()
                            all_layer.append(f"{ln}.attn1")
        k = 0
        for i, (name, module) in enumerate(get_attn_modules(model)):
            #import ipdb; ipdb.set_trace()
            if name not in all_layer:
                print(f"[UNREG][{i:3d}] {name}")
            else:
                print(f"[ REG ][{i:3d}] {name}")
                k += 1
        print(f'\n\nNUM_OF_K={k}\n\n')

        return blocks

import re
class SlimeAttentionLocator(ModuleLocator["CrossAttention"]):
    # _layers = [
    #     'up_blocks.1.attentions.0.transformer_blocks.0.attn2',
    #     'up_blocks.1.attentions.1.transformer_blocks.0.attn2',
    #     'up_blocks.1.attentions.2.transformer_blocks.0.attn2',
    #     'up_blocks.2.attentions.0.transformer_blocks.0.attn2',
    #     'up_blocks.2.attentions.1.transformer_blocks.0.attn2',
    #     'up_blocks.3.attentions.0.transformer_blocks.0.attn1',
    #     'up_blocks.3.attentions.1.transformer_blocks.0.attn1',
    #     'up_blocks.3.attentions.2.transformer_blocks.0.attn1'
    # ] # SD2.1

    _layers = [
        'up_blocks.0.attentions.0.transformer_blocks.0.attn2',
        'up_blocks.0.attentions.0.transformer_blocks.1.attn2',
        'up_blocks.0.attentions.0.transformer_blocks.2.attn2',
        'up_blocks.0.attentions.0.transformer_blocks.3.attn2',
        'up_blocks.0.attentions.0.transformer_blocks.4.attn2',
        'up_blocks.0.attentions.0.transformer_blocks.5.attn2',
        'up_blocks.0.attentions.0.transformer_blocks.6.attn2',
        'up_blocks.0.attentions.0.transformer_blocks.7.attn2',
    ] # SDXL

    def __init__(
        self,
        locate_attn1: bool = False,
        locate_attn2: bool = True,
        locate_middle_block: bool = False,
    ):
        super().__init__()
        self.locate_attn1 = locate_attn1
        self.locate_attn2 = locate_attn2
        self.locate_middle_block = locate_middle_block

    def locate(self, pipe: "StableDiffusionPipeline") -> Dict[str, "CrossAttention"]:
        """
        Locate cross-attention modules in a UNet2DConditionModel.

        Args:
            pipe (`StableDiffusionPipeline`): The pipeline with UNet containing
            the cross-attention modules.

        Returns:
            `Dict[str, CrossAttention]`: The cross-attention modules.
        """
        model = pipe.unet
        blocks = {}
        once_error = False
        registered_idx = []
        k = 0
        for layer in self._layers:
            try:
                module, _i = self.get_module_by_name(model, layer)
                registered_idx.append(_i)
                k += 1
                blocks[layer.replace(".", '-')] = module
                print(f'Successfully located {layer} with {str(module)}')
            except AttributeError as e:
                once_error = True
                print(f"Error locating {layer}: {e}")
        
        print(f'\n\nNUM_OF_K={k}\n\n')

        

        for i, (name, module) in enumerate(get_attn_modules(model)):
            if i not in registered_idx:
                print(f"[UNREG][{i:3d}] {name}")
            else:
                print(f"[ REG ][{i:3d}] {name}")

        if len(self._layers) != k:
            import ipdb; ipdb.set_trace()

        if once_error:
            print("-"*80)
            for name, module in pipe.unet.named_modules():
                print(name)
            print("-"*80)
        return blocks

        #return [(name, module) for name, module in model.named_modules() if str(name)[:-1] == ".attn"]
    def get_module_by_name(self, model, module_name):
        # Use named_modules of the unet to locate the actual instance

        named_modules_of_attn = get_attn_modules(model)
        print(f"Named modules of attn: \n{named_modules_of_attn}\n\n")
        # for i, (name, module) in enumerate(model.named_modules()):
        for i, (name, module) in enumerate(named_modules_of_attn):
            if name == module_name:
                return module, i
        raise AttributeError(f"Module {module_name} not found in model")