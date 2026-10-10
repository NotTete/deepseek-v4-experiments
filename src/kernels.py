
from typing import override
from math import sqrt
import torch.nn as nn
import torch.nn.functional as F
import torch

class Linear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        compute_dtype: torch.dtype = torch.bfloat16, 
        weight_dtype: torch.dtype = torch.bfloat16, 
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.compute_dtype: torch.dtype = compute_dtype

        self.weight: nn.Parameter = nn.Parameter(torch.empty(
            out_dim, in_dim, 
            dtype=weight_dtype, 
            device=device
        ))

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype

        x = x.to(self.compute_dtype)
        w = self.weight.to(self.compute_dtype)

        x = F.linear(x, w)

        x = x.to(out_dtype)
        return x



class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        compute_dtype: torch.dtype = torch.bfloat16, 
        weight_dtype: torch.dtype = torch.bfloat16, 
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.eps: float = eps
        self.compute_dtype: torch.dtype = compute_dtype

        self.weight: nn.Parameter = nn.Parameter(torch.ones(
            dim,
            dtype=weight_dtype,
            device=device
        ))

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype

        x = x.to(self.compute_dtype)
        w = self.weight.to(self.compute_dtype)

        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        x = w * x

        x = x.to(out_dtype)
        return x



class Hadamard(nn.Module):
    def __init__(
        self,
        dim: int,
        compute_dtype: torch.dtype = torch.bfloat16, 
        device: str = "cpu"
    ) -> None:
        super().__init__()

        # Comprobar que es potencia de 2
        assert((dim & (dim - 1)) == 0 and dim > 0)

        self.compute_dtype: torch.dtype = compute_dtype
        self.dim: int = dim

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        original_shape = x.shape

        x = x.to(self.compute_dtype)

        h = 2
        while h <= self.dim:
            hf = h // 2
            x = x.view(*original_shape[:-1], self.dim // h, h)
            half_1, half_2 = x[..., :hf], x[..., hf:]
            x = torch.cat((half_1 + half_2, half_1 - half_2), dim=-1)
            h *= 2
        x = (x / sqrt(self.dim)).view(*original_shape)

        x = x.to(out_dtype)
        return x



class RoPE(nn.Module):
    def __init__(
        self,
        dim: int,
        compute_dtype: torch.dtype = torch.bfloat16, 
        inverse: bool = False,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        # Comprobar que es divisible por 2
        assert(dim % 2 == 0)

        self.dim: int = dim
        self.compute_dtype: torch.dtype = compute_dtype
        self.inverse: bool = inverse

    @override
    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        # La dimensión debe coincider con lo indicado
        assert(x.shape[-1] == self.dim)
        assert(freqs.shape[-1] == 2 and freqs.shape[-2] == self.dim // 2)
        assert(x.shape[-2] == freqs.shape[-3])

        original_shape = x.shape
        out_dtype = x.dtype

        x = x.to(self.compute_dtype)
        freqs = freqs.to(self.compute_dtype)
        x = x.view(*original_shape[:-1], self.dim // 2, 2)

        x0, x1 = x[..., 0], x[..., 1]
        cos, sin = freqs[..., 0], freqs[..., 1]

        if self.inverse:
            sin = -sin

        y0 = x0 * cos - x1 * sin
        y1 = x0 * sin + x1 * cos

        x[..., 0] = y0
        x[..., 1] = y1

        x = x.view(*original_shape)
        x = x.to(out_dtype)
        return x

class RegularAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        compute_dtype: torch.dtype = torch.bfloat16, 
        inverse: bool = False,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        # Comprobar que es divisible por 2
        assert(dim % 2 == 0)

        self.dim: int = dim
        self.compute_dtype: torch.dtype = compute_dtype
        self.inverse: bool = inverse

    @override
    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        # La dimensión debe coincider con lo indicado
        assert(x.shape[-1] == self.dim)
        assert(freqs.shape[-1] == 2 and freqs.shape[-2] == self.dim // 2)
        assert(x.shape[-2] == freqs.shape[-3])

        original_shape = x.shape
        out_dtype = x.dtype

        x = x.to(self.compute_dtype)
        freqs = freqs.to(self.compute_dtype)
        x = x.view(*original_shape[:-1], self.dim // 2, 2)

        x0, x1 = x[..., 0], x[..., 1]
        cos, sin = freqs[..., 0], freqs[..., 1]

        if self.inverse:
            sin = -sin

        y0 = x0 * cos - x1 * sin
        y1 = x0 * sin + x1 * cos

        x[..., 0] = y0
        x[..., 1] = y1

        x = x.view(*original_shape)
        x = x.to(out_dtype)
        return x


