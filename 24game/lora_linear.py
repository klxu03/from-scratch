import torch.nn as nn
import torch
import math
from contextlib import contextmanager
from transformers import Qwen3_5ForCausalLM
from types import MethodType

def apply_lora(model: Qwen3_5ForCausalLM):
    model.requires_grad_(False)
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and name != "lm_head":
            model.set_submodule(name, LoRALinear(module), strict=True)
    model.disable_lora = MethodType(disable_lora, model)
    return model

@contextmanager
def disable_lora(model: nn.Module):
    layers = [
        m for m in model.modules() if isinstance(m, LoRALinear)
    ]
    previous = [layer.enabled for layer in layers]
    
    try:
        for layer in layers:
            layer.enabled = False
        yield
    finally:
        for layer, enabled in zip(layers, previous):
            layer.enabled = enabled

from contextlib import nullcontext

class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 8, alpha: float = 4):
        super().__init__()
        self.scale = alpha / math.sqrt(r)
        self.enabled = True

        self.base = linear
        self.A = nn.Linear(linear.in_features, r, bias=False, device=linear.weight.device, dtype=torch.float32)
        self.B = nn.Linear(r, linear.out_features, bias=False, device=linear.weight.device, dtype=torch.float32)

        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original = self.base(x)

        if not self.enabled:
            return original
        
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if x.is_cuda
            else nullcontext()
        )

        with context:
            update = self.B(self.A(x.float()))

        return original + (self.scale * self.B(self.A(x.float()))).to(original.dtype)