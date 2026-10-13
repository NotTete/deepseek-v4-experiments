from typing import override
import math
from math import sqrt
import torch.nn as nn
import torch.nn.functional as F
import torch

def precompute_rope_freqs(
    dim: int,
    seqlen: int,
    original_seq_len: int = 0,
    base: float = 10000.0,
    factor: float = 1.0,
    beta_fast: int = 32,
    beta_slow: int = 1,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Precomputa los senos y cosenos necesarios para realizar
    el RoPE. Si la `original_seq_len == 0` no aplica YaRN.

    YaRN es una técnica usada para poder aumentar la longitud
    del contexto de un modelo a una mayor de la usada en entrenamiento.
    Esto permite mantener relaciones entre tokens en contexto de 1M aunque
    el modelo haya sido entrenado con 128K de contexto.

    RoPE Referencia: https://arxiv.org/abs/2104.09864
    YaRN Referencia: https://arxiv.org/abs/2309.00071
    """

    def find_correction_dim(num_rotations: int, dim: int, base: float, max_seq_len: int) -> float:
        """
        Dado un número de rotaciones(`num_rotations`) devuelve el indice de la dimension donde esa
        frecuencia completa `num_rotations` vueltas en una ventana de `max_seq_len`. Usado por YaRN.
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot: int, high_rot: int, dim: int, base: float, max_seq_len: int) -> tuple[int, int]:
        """
        Traduce las frecuencias de interés (`beta_fast`, `beta_slow`) en un rango
        de indices de dimensiones `[low, high]`. Usado por YaRN.

        Si dim_i >= dim(beta_fast), rotan con una alta frecuencia y se dejan intactas
        Si dim_i <= dim(beta_slow), rotan con una baja frecuencia y se multiplican para cubrir contextos largos
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))

        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(mn: float, mx: float, dim: int) -> torch.Tensor:
        """
        Devuelve un factor [0, 1] con el que hacer interpolacion lineal. Usado por YaRN.
        """
        if mn == mx:
            mx += 0.001
        return torch.clamp((torch.arange(dim, dtype=torch.float32) - mn) / (mx - mn), 0, 1)

    # Frecuencias base del RoPE: [dim // 2,]
    # base ^ -(dim_i/dim)
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    # Aplicamos YaRN a las frecuencias base
    if original_seq_len > 0:
        # Rango de dimensiones donde se hace la interpolacion
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)

        # smooth = 0 -> frecuencia interpolada (dividida por `factor`),
        # smooth = 1 -> frecuencia original sin tocar.
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)

        # Interpolación entre frecuencia escalada y frecuencia original
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    # Ángulos por los que hacer la rotación para cada elemento de la secuencia
    # [seqlen, dim // 2]
    angles = torch.outer(torch.arange(seqlen, dtype=torch.float32), freqs)

    # Coseno y Seno de los ángulos
    # [seqlen, dim // 2, 2]
    return torch.stack([angles.cos(), angles.sin()], dim=-1).to(device)


class Linear(nn.Module):
    """
    Capa lineal sin sesgo.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        compute_dtype: torch.dtype = torch.float16,
        weight_dtype: torch.dtype = torch.float16,
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
        # x  : [..., in_dim]
        # out: [..., out_dim]
        out_dtype = x.dtype

        x = x.to(self.compute_dtype)
        w = self.weight.to(self.compute_dtype)

        x = F.linear(x, w)

        x = x.to(out_dtype)
        return x


class RMSNorm(nn.Module):
    """
    Capa de normalización por Root Mean Square.
    Esta capa solo normaliza el segundo momento pero no la media.

    Referencia: https://arxiv.org/abs/1910.07467
    """
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6, # Constante de estabilidad numérica
        compute_dtype: torch.dtype = torch.float16,
        weight_dtype: torch.dtype = torch.float16,
        device: str = "cpu",
        weighted: bool = True # Existe un parámetro gamma de pesos aprendido
    ) -> None:
        super().__init__()

        self.eps: float = eps
        self.compute_dtype: torch.dtype = compute_dtype
        self.weighted: bool = weighted

        if self.weighted:
            self.weight: nn.Parameter = nn.Parameter(torch.ones(
                dim,
                dtype=weight_dtype,
                device=device
            ))

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x  : [..., dim]
        # out: [..., dim]
        out_dtype = x.dtype

        x = x.to(self.compute_dtype)

        # Media cuadrática (segundo momento) por la última dimensión
        var = x.square().mean(-1, keepdim=True)

        # Normalizamos: x / sqrt(var + eps)
        x = x * torch.rsqrt(var + self.eps)

        if self.weighted:
            w = self.weight.to(self.compute_dtype)
            x = w * x

        x = x.to(out_dtype)
        return x


class WeightedSum(nn.Module):
    """
    Suma ponderada mediante softmax, con un bias
    posicional aditivo distinto para cada slot.

    El número de `slots` y su disposición (solapados o no) 
    los decide quien lo usa.

    out = softmax(scores + ape, across slots) * values
    """
    def __init__(
        self,
        hidden_dim: int,
        slots: int,
        compute_dtype: torch.dtype = torch.float16,
        weight_dtype: torch.dtype = torch.float16,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        self.hidden_dim: int = hidden_dim
        self.slots: int = slots
        self.compute_dtype: torch.dtype = compute_dtype

        self.ape: nn.Parameter = nn.Parameter(torch.zeros(
            slots, hidden_dim, dtype=weight_dtype, device=device
        ))

    @override
    def forward(self, values: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        # values, scores: [..., slots, hidden_dim]
        # out           : [..., hidden_dim]
        out_dtype = values.dtype

        values = values.to(self.compute_dtype)
        gate = scores.to(self.compute_dtype) + self.ape.to(self.compute_dtype)

        out = (F.softmax(gate, dim=-2) * values).sum(dim=-2)   # [..., hidden_dim]

        out = out.to(out_dtype)
        return out


class Hadamard(nn.Module):
    """
    Transformada de Hadamard rápida (FWHT) por mariposas.
    `dim` debe ser una potencia de 2.

    Referencia: https://en.wikipedia.org/wiki/Hadamard_transform
    """
    def __init__(
        self,
        dim: int,
        compute_dtype: torch.dtype = torch.float16, 
        device: str = "cpu"
    ) -> None:
        super().__init__()

        # Comprobar que es potencia de 2
        assert((dim & (dim - 1)) == 0 and dim > 0)

        self.compute_dtype: torch.dtype = compute_dtype
        self.dim: int = dim

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x  : [..., dim]
        # out: [..., dim]
        out_dtype = x.dtype
        original_shape = x.shape

        x = x.to(self.compute_dtype)

        # h = tamaño de bloque, hf = mitad de bloque
        h = 2
        while h <= self.dim:
            hf = h // 2
            x = x.view(*original_shape[:-1], self.dim // h, h)
            half_1, half_2 = x[..., :hf], x[..., hf:]
            x = torch.cat((half_1 + half_2, half_1 - half_2), dim=-1)
            h *= 2

        # Normalización 1/sqrt(dim)
        x = (x / sqrt(self.dim)).view(*original_shape)

        x = x.to(out_dtype)
        return x



class RoPE(nn.Module):
    """
    Aplica (o invierte, si `inverse=True`) rotaciones posicionales RoPE
    sobre pares consecutivos de dimensiones.

    Invertir es multiplicar por el conjugado. Es decir, `sin` pasa a ser `-sin`.

    Referencia: https://arxiv.org/abs/2104.09864
    """
    def __init__(
        self,
        dim: int,
        compute_dtype: torch.dtype = torch.float16, 
        inverse: bool = False,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        # Comprobar que es divisible por 2
        assert(dim % 2 == 0)

        self.dim: int = dim
        self.compute_dtype: torch.dtype = compute_dtype
        self.inverse: bool = inverse   # Si True aplica la rotación inversa

    @override
    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        # x    : [..., dim]
        # freqs: [seqlen, dim // 2, 2]
        # out  : [..., dim]
        assert(x.shape[-1] == self.dim)
        assert(freqs.shape[-1] == 2 and freqs.shape[-2] == self.dim // 2)

        original_shape = x.shape
        out_dtype = x.dtype

        x = x.to(self.compute_dtype)
        freqs = freqs.to(self.compute_dtype)

        # Agrupamos la última dimensión en pares: [..., dim // 2, 2]
        x = x.view(*original_shape[:-1], self.dim // 2, 2)

        # Componentes del par a rotar y su coseno/seno
        x0, x1 = x[..., 0], x[..., 1]
        cos, sin = freqs[..., 0], freqs[..., 1]

        # Para la rotación inversa basta con negar el seno
        if self.inverse:
            sin = -sin

        # Rotación 2D estándar de cada par
        y0 = x0 * cos - x1 * sin
        y1 = x0 * sin + x1 * cos

        y = torch.stack((y0, y1), dim=-1).view(*original_shape)
        return y.to(out_dtype)

class SafeMultiQueryAttention(nn.Module):
    """
    Atención multi-query con sink de atención y donde las claves y los valores
    son los mismos y están compartidos a lo largo de las cabezas de la query.

    Se extraen los tokens de una ventana deslizante sin comprimir y entradas
    comprimidas de la KV Cache seleccionadas por token mediante top-k.

    El algoritmo implementado usa SafeSoftmax para evitar errores numéricos.
    Además esta implementación no es muy eficiente ya que usa demasiado indexing
    en vez de aprovechar la continuidad de la memoria.

    Idealmente en un framework con mayor control de memoria, se debería de implementar
    por medio del algoritmo de FlashAttention. Que reduce el número de pasadas a 1. Existen
    hasta 3 versiones o implementaciones principales de este algoritmo pero las mejoras están
    pensadas para las GPU Nvidia.

    Referencia SafeSoftmax: https://tianlong312.github.io/blog/post-Online-Softmax/
    Referencia FlashAttention: https://arxiv.org/abs/2205.14135
    Otra referencia: https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
    """
    def __init__(
        self,
        n_heads: int,
        compute_dtype: torch.dtype = torch.float16, 
        weight_dtype: torch.dtype = torch.float16, 
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.n_heads: int = n_heads
        self.compute_dtype: torch.dtype = compute_dtype

        self.attention_sink: nn.Parameter = nn.Parameter(torch.zeros(
            n_heads,
            dtype=weight_dtype,
            device=device
        ))

    @override
    def forward(
        self,
        q: torch.Tensor,
        window: torch.Tensor,
        compressed: torch.Tensor | None,
        topk_index: torch.Tensor | None,
        win_index: torch.Tensor,
    ) -> torch.Tensor:
        # q         : [S, n_heads, head_dim]
        # window    : [N, head_dim]  buffer de ventana sin comprimir. k = v.
        # compressed: [C, head_dim] o None  buffer de entradas comprimidas. k = v.
        # topk_index: [S, topk]  entradas de `compressed` elegidas por token,
        #             con -1 para relleno. None en capas puramente windowed.
        # win_index : [S, W]  indices (sobre `window`) de la ventana de cada
        #             query, con -1 para posiciones fuera de la secuencia.
        # out       : [S, n_heads, head_dim]
        out_dtype = q.dtype

        q = q.to(self.compute_dtype)
        window = window.to(self.compute_dtype)
        if compressed is not None:
            compressed = compressed.to(self.compute_dtype)

        scale = 1.0 / sqrt(q.shape[-1])

        q = q.transpose(0, 1)                                       # [n_heads, S, head_dim]

        # Rama ventana (deslizante).
        W = win_index.shape[-1]

        # Clampeamos para evitar problemas al indexar
        win_idx = win_index.clamp(min=0)

        # Máscara que devuelve los indices válidos.
        win_valid = win_index >= 0

        # Indices seleccionados
        window_sel = window[win_idx]                                 # [S, W, head_dim]

        # QxK^T * scale
        window_scores = torch.einsum("hsd,swd->hsw", q, window_sel) * scale  # [n_heads, S, W]

        # Maskeamos los scores de los índices inválidos
        window_scores = window_scores.masked_fill(~win_valid.unsqueeze(0), float("-inf"))


        # Rama comprimida: sólo las top-k entradas elegidas por el índice.
        if compressed is not None:
            # Misma técnica para evitar problemas al indexar que en window
            selected = compressed[topk_index.clamp(min=0)]          # [S, topk, head_dim]

            # Calculamos scores
            compressed_scores = torch.einsum("hsd,skd->hsk", q, selected) * scale  # [n_heads, S, topk]

            # Maskeamos los scores de los índices inválidos
            compressed_scores = compressed_scores.masked_fill(
                ~(topk_index >= 0).unsqueeze(0), float("-inf")
            )

            # Concatenamos los scores
            scores = torch.cat([window_scores, compressed_scores], dim=-1)  # [n_heads, S, W + topk]
        else:
            selected = None
            scores = window_scores
        
        # Columna del sink un logit por cabeza, replicado en cada query.
        sink = self.attention_sink.to(self.compute_dtype)           # [n_heads]
        sink = sink.view(-1, 1, 1).expand(-1, q.shape[1], 1)        # [n_heads, S, 1]
        scores = torch.cat([scores, sink], dim=-1)                  # [n_heads, S, W + topk + 1]

        # Softmax estable: restar el máximo de cada fila antes de exponenciar.
        scores = scores - scores.amax(dim=-1, keepdim=True)
        attn = torch.exp(scores)
        attn = attn / attn.sum(dim=-1, keepdim=True)

        # El sink tiene valor cero: se descarta su columna. v = k = window/selected.
        out = torch.einsum("hsw,swd->hsd", attn[..., :W], window_sel)
        if selected is not None:
            out = out + torch.einsum("hsk,skd->hsd", attn[..., W:-1], selected)

        # [n_heads, S, head_dim] -> [S, n_heads, head_dim].
        out = out.transpose(0, 1)
        out = out.to(out_dtype)
        return out


class CheapMultiQueryAttention(nn.Module):
    """
    Atención multi-query barata: sustituye el softmax por una ReLU y una
    ponderación por token, evitando la normalización entre posiciones.
    """
    def __init__(
        self,
        compute_dtype: torch.dtype = torch.float16,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.compute_dtype: torch.dtype = compute_dtype

    @override
    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        # q       : [S, n_heads, head_dim]
        # kv      : [T, head_dim]
        # weights : [S, n_heads]
        # out     : [S, T]
        out_dtype = q.dtype

        q = q.to(self.compute_dtype)
        kv = kv.to(self.compute_dtype)
        weights = weights.to(self.compute_dtype)

        # Similitudes query-key
        score = torch.einsum("shd,td->sht", q, kv)                  # [S, n_heads, T]
        # ReLU + ponderación por cabeza y suma sobre las cabezas
        score = (score.relu() * weights.unsqueeze(-1)).sum(dim=1)   # [S, T]
        return score.to(out_dtype)

