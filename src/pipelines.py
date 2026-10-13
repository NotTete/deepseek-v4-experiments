from typing import override
import torch.nn as nn
import torch
from kernels import (
    Linear,
    RMSNorm,
    RoPE,
    WeightedSum,
    Hadamard,
    SafeMultiQueryAttention,
    CheapMultiQueryAttention,
    precompute_rope_freqs,
)


def get_window_topk_idxs(window_size: int, seqlen: int, start_pos: int) -> torch.Tensor:
    """
    Genera un tensor con los indices de ventana deslizante necesarios dado una
    longitud de secuencia, una longitud de ventana y una posición inicial

    Devuelve Indices [seqlen, window_size].
    """
    W = window_size
    i = torch.arange(seqlen).view(-1, 1)          # [S, 1]
    j = torch.arange(W).view(1, -1)               # [1, W]
    abs_pos = start_pos + i - (W - 1) + j         # [S, W]  = p - W + 1 + j
    slots = abs_pos % W                           # [S, W]  slot circular
    return torch.where(abs_pos >= 0, slots, torch.full_like(slots, -1))


def get_compress_topk_idxs(ratio: int, seqlen: int, start_pos: int, offset: int = 0) -> torch.Tensor:
    """Indices [seqlen, n_comp] sobre el buffer comprimido. -1 = sin atender.

    Para la query i (posición absoluta p = start_pos + i) son visibles las
    ventanas completas j < (p + 1) // ratio. Vale para cualquier chunk/prefill.
    """
    n_max = (start_pos + seqlen) // ratio
    if n_max == 0:
        return torch.full((seqlen, 0), -1, dtype=torch.long)
    i = torch.arange(seqlen).view(-1, 1)
    j = torch.arange(n_max).view(1, -1)
    valid = j < (start_pos + i + 1) // ratio
    matrix = torch.where(valid, j + offset, torch.full_like(j, -1))
    return matrix.to(torch.long)


class MultiQueryProyection(nn.Module):
    """
    Proyección de la query en dos pasos (estilo LoRA):
    Linear -> RMSNorm -> Linear -> reshape -> RMSNorm(sin peso) -> RoPE.

    Devuelve además la proyección intermedia (`shared_proyection`), que el
    `Indexer` reutiliza como su propia query.
    """
    def __init__(
        self,
        input_dim: int = 4096,
        intermediate_dim: int = 1024,
        hidden_dim: int = 512,
        n_heads: int = 64,
        rope_dim: int = 64,
        compute_dtype: torch.dtype = torch.float16,
        device: str = "cpu",
    ):
        super().__init__()

        self.input_dim: int = input_dim
        self.intermediate_dim: int = intermediate_dim
        self.hidden_dim: int = hidden_dim
        self.n_heads: int = n_heads
        self.rope_dim: int = rope_dim
        self.compute_dtype: torch.dtype = compute_dtype

        self.linear_1: Linear = Linear(self.input_dim, self.intermediate_dim, compute_dtype=compute_dtype, device=device)
        self.norm_1: RMSNorm = RMSNorm(self.intermediate_dim, compute_dtype=compute_dtype, device=device)
        self.linear_2: Linear = Linear(self.intermediate_dim, self.n_heads * self.hidden_dim, compute_dtype=compute_dtype, device=device)
        self.norm_2: RMSNorm = RMSNorm(self.hidden_dim, compute_dtype=compute_dtype, device=device, weighted=False)
        self.rope: RoPE = RoPE(self.rope_dim, compute_dtype=compute_dtype, device=device)

    @override
    def forward(self, seq_pos: int, x: torch.Tensor, freqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x    : [S, input_dim]
        # freqs: [S, rope_dim // 2, 2]
        # out  : shared [S, intermediate_dim], query [S, n_heads, hidden_dim]
        x = self.linear_1(x)
        x = self.norm_1(x)

        # Proyección usada por el indexer: [S, intermediate_dim]
        shared_proyection = x

        x = self.linear_2(x)
        x = x.reshape(-1, self.n_heads, self.hidden_dim)
        x = self.norm_2(x)

        # RoPE en las últimas `rope_dim` dimensiones (por cabeza).
        nope_dim = self.hidden_dim - self.rope_dim
        x = x.clone()
        x[..., nope_dim:] = self.rope(x[..., nope_dim:], freqs.unsqueeze(-3))

        query_proyection = x
        return shared_proyection, query_proyection


class WindowKV(nn.Module):
    """
    KV de la ventana deslizante (sin comprimir):
    Linear -> RMSNorm -> RoPE.
    """
    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 512,
        rope_dim: int = 64,
        compute_dtype: torch.dtype = torch.float16,
        device: str = "cpu",
    ):
        super().__init__()

        self.input_dim: int = input_dim
        self.hidden_dim: int = hidden_dim
        self.rope_dim: int = rope_dim
        self.compute_dtype: torch.dtype = compute_dtype

        self.linear: Linear = Linear(self.input_dim, self.hidden_dim, compute_dtype=compute_dtype, device=device)
        self.norm: RMSNorm = RMSNorm(self.hidden_dim, compute_dtype=compute_dtype, device=device)
        self.rope: RoPE = RoPE(self.rope_dim, compute_dtype=compute_dtype, device=device)

    @override
    def forward(self, seq_pos: int, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        # x    : [S, input_dim]
        # freqs: [S, rope_dim // 2, 2]
        # out  : [S, hidden_dim]
        x = self.linear(x)
        x = self.norm(x)

        nope_dim = self.hidden_dim - self.rope_dim
        x = x.clone()
        x[..., nope_dim:] = self.rope(x[..., nope_dim:], freqs)

        return x


class TokenCompressor(nn.Module):
    """
    Compresor de tokens.

    Proyecta la entrada a `coff*head_dim` (valor y score), agrupa en ventanas de
    `ratio` tokens, las combina con `WeightedSum` (+ `ape`), normaliza, aplica RoPE
    y, opcionalmente, Hadamard. Devuelve [S/ratio, hidden_dim] y escribe la KV cache.
    """
    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 512,
        rope_dim: int = 64,
        ratio: int = 4,
        overlap: bool = True,
        rotate: bool = False,  # Hadamard para mejorar cuantización (indexer).
        compute_dtype: torch.dtype = torch.float16,
        device: str = "cpu",
    ):
        super().__init__()

        self.input_dim: int = input_dim
        self.hidden_dim: int = hidden_dim
        self.rope_dim: int = rope_dim
        self.ratio: int = ratio
        self.overlap: bool = overlap
        self.rotate: bool = rotate
        self.compute_dtype: torch.dtype = compute_dtype

        self.coff: int = 2 if overlap else 1
        self.width: int = self.coff * hidden_dim       # columnas de wkv / wgate
        self.slots: int = self.coff * ratio            # slots tras overlap_transform

        self.linear_value = Linear(self.input_dim, self.width, compute_dtype=compute_dtype, device=device)
        self.linear_score = Linear(self.input_dim, self.width, compute_dtype=compute_dtype, device=device)

        self.weighted_sum: WeightedSum = WeightedSum(
            self.hidden_dim, self.slots,
            compute_dtype=compute_dtype, device=device,
        )

        self.norm: RMSNorm = RMSNorm(self.hidden_dim, compute_dtype=compute_dtype, device=device)
        self.rope: RoPE = RoPE(self.rope_dim, compute_dtype=compute_dtype, device=device)
        self.hadamard: Hadamard | None = Hadamard(self.hidden_dim, compute_dtype=compute_dtype, device=device) if rotate else None

        # Buffers que guardan tokens sin comprimir hasta que puedan ser comprimidos
        self.register_buffer(
            "value_state",
            torch.zeros(self.slots, self.width, dtype=compute_dtype, device=device),
            persistent=False,
        )
        self.register_buffer(
            "score_state",
            torch.full((self.slots, self.width), float("-inf"), dtype=compute_dtype, device=device),
            persistent=False,
        )

    def reset(self) -> None:
        self.value_state.zero_()
        self.score_state.fill_(float("-inf"))

    @override
    def forward(
        self,
        seq_pos: int,
        x: torch.Tensor,
        freqs: torch.Tensor,
        kv_cache: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        # x        : [S, input_dim]
        # freqs    : tabla completa [max_seq, rope/2, 2]
        # kv_cache : cache comprimida
        # out      : [n_win, hidden_dim] o None si no se cerró ninguna ventana.
        #
        # Streaming unificado: válido para "prefill" y "decode"

        ratio, d, width = self.ratio, self.hidden_dim, self.width
        S = x.shape[0]

        # Proyecciones lineales del score y value
        values = self.linear_value(x).to(self.compute_dtype)   # [S, coff*head_dim]
        scores = self.linear_score(x).to(self.compute_dtype)   # [S, coff*head_dim]

        # El estado guarda los tokens que aún no completan una ventana de `ratio`.
        # `filled` son los tokens sobrantes de la llamada anterior. 
        # Se ponen delante de los nuevos y se consumen ventanas completas de izquierda a derecha
        # El resto (`rem`) queda en el estado para la próxima llamada.
        # 
        # Con overlap el estado tiene `2*ratio` slots: los `ratio` primeros son la
        # ventana anterior (mitad A) y los `ratio` siguientes el hueco de la actual.
        filled = seq_pos % ratio
        off = ratio if self.overlap else 0
        if filled:
            stream_v = torch.cat([self.value_state[off: off + filled], values], dim=0)
            stream_s = torch.cat([self.score_state[off: off + filled], scores], dim=0)
        else:
            stream_v, stream_s = values, scores

        # Número de ventanas completas y tokens que sobran al final.
        n_win, rem = divmod(stream_v.shape[0], ratio)
        out = None
        if n_win:
            # Vistas por ventana: [n_win*ratio, width] -> [n_win, ratio, width].
            win_v = stream_v[: n_win * ratio].reshape(n_win, ratio, width)
            win_s = stream_s[: n_win * ratio].reshape(n_win, ratio, width)
            if self.overlap:
                # Cada ventana se solapa con la anterior: slots =
                # [mitad A de la ventana previa, mitad B de la actual].
                # La "previa" de la ventana 0 es la guardada en el estado.
                prev_v = torch.cat([self.value_state[:ratio].unsqueeze(0), win_v[:-1]], dim=0)
                prev_s = torch.cat([self.score_state[:ratio].unsqueeze(0), win_s[:-1]], dim=0)
                slots_v = torch.cat([prev_v[..., :d], win_v[..., d:]], dim=1)   # [n_win, 2r, d]
                slots_s = torch.cat([prev_s[..., :d], win_s[..., d:]], dim=1)
                if seq_pos - filled == 0:
                    # Arranque de secuencia: sólo la ventana 0 carece de previa.
                    # value=0 y score=-inf => el softmax la ignora.
                    slots_v[0, :ratio] = 0.0
                    slots_s[0, :ratio] = float("-inf")
            else:
                # Sin overlap: cada ventana son exactamente sus `ratio` tokens.
                slots_v, slots_s = win_v, win_s

            # Calculamos los tokens comprimidos
            out = self.weighted_sum(slots_v, slots_s)   # [n_win, d]

        # Actualizamos el estado con los tokens sobrantes
        if self.overlap:
            # Guardamos la última ventana completa como "previa" de la próxima
            # llamada y reiniciamos el hueco de la ventana actual.
            if n_win:
                self.value_state[:ratio] = stream_v[(n_win - 1) * ratio: n_win * ratio]
                self.score_state[:ratio] = stream_s[(n_win - 1) * ratio: n_win * ratio]
            self.value_state[ratio:] = 0.0
            self.score_state[ratio:] = float("-inf")
            if rem:
                # Los `rem` tokens sobrantes ocupan el inicio del hueco actual.
                self.value_state[ratio: ratio + rem] = stream_v[n_win * ratio:]
                self.score_state[ratio: ratio + rem] = stream_s[n_win * ratio:]
        else:
            # Sin overlap basta con guardar los `rem` tokens sobrantes al inicio.
            self.value_state[:] = 0.0
            self.score_state[:] = float("-inf")
            if rem:
                self.value_state[:rem] = stream_v[n_win * ratio:]
                self.score_state[:rem] = stream_s[n_win * ratio:]

        # Si no hemos comprimido ningún token no devolvemos nada
        if out is None:
            return None

        # RoPE en la posición de inicio de cada ventana comprimida
        window_starts = (seq_pos - filled) + torch.arange(n_win, device=out.device) * ratio
        freqs_win = freqs[window_starts]
        out = self.norm(out)
        out = out.clone()
        out[..., d - self.rope_dim:] = self.rope(out[..., d - self.rope_dim:], freqs_win)

        # Si aplicamos rotación usamos Hadamard. Se usa dentro del Indexer
        if self.hadamard is not None:
            out = self.hadamard(out)

        # Escribimos los resultados en la KV Cache
        if kv_cache is not None:
            base = (seq_pos - filled) // ratio
            kv_cache[base: base + n_win] = out

        return out


class IndexerTokenCompressor(TokenCompressor):
    """Token Level Compressor del indexer: head_dim = index_head_dim (128),
    overlap (ratio==4) y Hadamard antes de la cuantización fp4."""

    def __init__(
        self,
        input_dim: int = 4096,
        index_head_dim: int = 128,
        rope_dim: int = 64,
        ratio: int = 4,
        compute_dtype: torch.dtype = torch.float16,
        device: str = "cpu",
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=index_head_dim,
            rope_dim=rope_dim,
            ratio=ratio,
            overlap=(ratio == 4),
            rotate=True,
            compute_dtype=compute_dtype,
            device=device,
        )

class Indexer(nn.Module):
    """
    Indexer a nivel de token (sólo CSA, ratio==4).

    Proyecta la query compartida a `head_dim`, aplica RoPE y Hadamard, comprime
    la entrada con `IndexerTokenCompressor` y puntúa con `cheap_attention` contra
    la Indexer KV Cache para elegir los top-`index_topk` índices comprimidos.
    """
    def __init__(
        self,
        input_dim: int = 4096,
        q_lora_rank: int = 1024,
        n_heads: int = 64,
        head_dim: int = 128,
        rope_dim: int = 64,
        index_topk: int = 512,
        ratio: int = 4,
        max_seq_len: int = 4096,
        compute_dtype: torch.dtype = torch.float16,
        device: str = "cpu",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.index_topk = index_topk
        self.ratio = ratio
        self.compute_dtype = compute_dtype
        self.softmax_scale = head_dim ** -0.5

        self.wq_b: Linear = Linear(q_lora_rank, n_heads * head_dim, compute_dtype=compute_dtype, device=device)
        self.weights_proj: Linear = Linear(input_dim, n_heads, compute_dtype=compute_dtype, device=device)
        self.rope: RoPE = RoPE(rope_dim, compute_dtype=compute_dtype, device=device)
        self.hadamard: Hadamard = Hadamard(head_dim, compute_dtype=compute_dtype, device=device)
        self.cheap_attention: CheapMultiQueryAttention = CheapMultiQueryAttention(
            compute_dtype=compute_dtype, device=device
        )
        self.compressor: IndexerTokenCompressor = IndexerTokenCompressor(
            input_dim, head_dim, rope_dim, ratio, compute_dtype=compute_dtype, device=device
        )

    @override
    def forward(
        self,
        seq_pos: int,
        x: torch.Tensor,
        shared_projection: torch.Tensor,
        freqs: torch.Tensor,
        kv_cache: torch.Tensor,
    ) -> torch.Tensor:
        # x                 : [S, input_dim]
        # shared_projection : [S, q_lora_rank]  (salida de MultiQueryProyection)
        # freqs             : tabla completa [max_seq, rope/2, 2]
        # kv_cache          : Indexer KV Cache
        # out               : [S, k]  índices top-k (-1 = sin atender), k = min(index_topk, n).
        S = x.shape[0]
        freqs_now = freqs[seq_pos: seq_pos + S]                              # [S, rope_dim // 2, 2]

        q = self.wq_b(shared_projection).reshape(-1, self.n_heads, self.head_dim)  # [S, n_heads, head_dim]
        q = q.clone()

        # RoPE sobre las últimas `rope_dim` dims; `freqs_now` se difunde por cabeza.
        q[..., -self.rope_dim:] = self.rope(
            q[..., -self.rope_dim:],                    # [S, n_heads, rope_dim]
            freqs_now.unsqueeze(-3),                    # [S, 1, rope_dim // 2, 2]
        )
        q = self.hadamard(q)                                                    # [S, n_heads, head_dim]

        # Compresión escribe en Indexer KV Cache
        self.compressor(seq_pos, x, freqs, kv_cache=kv_cache)

        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)  # [S, n_heads]
        end_pos = seq_pos + S
        kv = kv_cache[: end_pos // self.ratio]                                 # [T, head_dim]
        index_score = self.cheap_attention(q, kv, weights)                     # [S, T]

        n = end_pos // self.ratio
        # Sólo son visibles las ventanas j < (p + 1) // ratio, con p la posición
        # absoluta de cada query. Vale para prefill y para chunks > 1 en decode.
        valid_count = (seq_pos + torch.arange(S, device=index_score.device).view(-1, 1) + 1) // self.ratio  # [S, 1]
        j = torch.arange(n, device=index_score.device).view(1, -1)            # [1, n]
        index_score = index_score.masked_fill(j >= valid_count, float("-inf"))  # [S, T]

        k = min(self.index_topk, n)
        topk_idxs = index_score.topk(k, dim=-1)[1]                             # [S, k]
        topk_idxs = torch.where(
            topk_idxs >= valid_count, torch.full_like(topk_idxs, -1), topk_idxs
        )                                                                      # [S, k]
        return topk_idxs.int()

    def reset(self) -> None:
        # La KV cache la posee Attention; aquí sólo el estado interno del compresor.
        self.compressor.reset()


class SharedKeyValueMultiQueryAttention(nn.Module):
    """
    Atención multi-query con K = V compartidas (ventana + comprimido).

    SafeSoftmax -> reverse RoPE -> proyección de salida agrupada (LoRA en dos
    pasos `wo_a` / `wo_b`) -> [S, out_dim].
    """
    def __init__(
        self,
        n_heads: int = 64,
        head_dim: int = 512,
        rope_dim: int = 64,
        o_groups: int = 8,
        o_lora_rank: int = 1024,
        out_dim: int = 4096,
        compute_dtype: torch.dtype = torch.float16,
        device: str = "cpu",
    ):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.o_groups = o_groups
        self.o_lora_rank = o_lora_rank
        self.compute_dtype = compute_dtype

        self.attention: SafeMultiQueryAttention = SafeMultiQueryAttention(n_heads, compute_dtype=compute_dtype, device=device)
        self.rope: RoPE = RoPE(rope_dim, compute_dtype=compute_dtype, inverse=True, device=device)
        # wo_a agrupa la salida: [o_groups*o_lora_rank, (n_heads*head_dim)/o_groups]
        self.wo_a: Linear = Linear(
            n_heads * head_dim // o_groups, o_groups * o_lora_rank,
            compute_dtype=compute_dtype, device=device,
        )
        self.wo_b: Linear = Linear(o_groups * o_lora_rank, out_dim, compute_dtype=compute_dtype, device=device)

    @override
    def forward(
        self,
        q: torch.Tensor,
        window: torch.Tensor,
        compressed: torch.Tensor | None,
        topk_index: torch.Tensor | None,
        win_index: torch.Tensor,
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        # q         : [S, n_heads, head_dim]
        # window    : [N, head_dim]
        # compressed: [C, head_dim] o None
        # topk_index: [S, topk] o None
        # win_index : [S, W]
        # freqs     : [S, rope_dim // 2, 2]
        # out       : [S, out_dim]
        o = self.attention(q, window, compressed, topk_index, win_index)   # [S, n_heads, head_dim]

        # Reverse RoPE sobre las últimas `rope_dim` dims (por cabeza).
        o = o.clone()
        o[..., -self.rope_dim:] = self.rope(o[..., -self.rope_dim:], freqs.unsqueeze(-3))

        S = o.shape[0]
        o = o.reshape(S, self.o_groups, -1)                                # [S, 8, 4096]
        wo_a = self.wo_a.weight.to(o.dtype).view(self.o_groups, self.o_lora_rank, -1)  # [8, 1024, 4096]
        o = torch.einsum("sgd,grd->sgr", o, wo_a)                          # [S, 8, 1024]
        return self.wo_b(o.flatten(1))                                     # [S, 4096]


class Attention(nn.Module):
    """
    Bloque de atención completo. El modo lo decide `ratio`:
      * ratio == 4   -> CSA: Compressor + Indexer (sparse top-k).
      * ratio == 128 -> HCA: Compressor (todo el KV comprimido).
      * ratio == 0   -> Window puro (sin compresión).

    Posee las KV caches (window / compress / indexer) y las freqs RoPE.
    """
    def __init__(
        self,
        dim: int = 4096,
        q_lora_rank: int = 1024,
        n_heads: int = 64,
        head_dim: int = 512,
        rope_dim: int = 64,
        window_size: int = 128,
        ratio: int = 4,
        index_n_heads: int = 64,
        index_head_dim: int = 128,
        index_topk: int = 512,
        o_groups: int = 8,
        o_lora_rank: int = 1024,
        max_seq_len: int = 4096,
        compress_rope_theta: float = 160000.0,
        rope_theta: float = 10000.0,
        rope_factor: float = 16.0,
        original_seq_len: int = 65536,
        beta_fast: int = 32,
        beta_slow: int = 1,
        compute_dtype: torch.dtype = torch.float16,
        device: str = "cpu",
    ):
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.window_size = window_size
        self.ratio = ratio
        self.max_seq_len = max_seq_len
        self.compute_dtype = compute_dtype

        self.mqa_proj: MultiQueryProyection = MultiQueryProyection(
            dim, q_lora_rank, head_dim, n_heads, rope_dim, compute_dtype=compute_dtype, device=device
        )
        self.window_kv: WindowKV = WindowKV(dim, head_dim, rope_dim, compute_dtype=compute_dtype, device=device)

        # self.ratio == 4   ==> CSA.    Compressor + Indexer
        # self.ratio == 128 ==> HCA.    Compressor
        # self.ratio == 0   ==> Window. No Compressor
        if self.ratio:
            self.compressor: TokenCompressor | None = TokenCompressor(
                dim, head_dim, rope_dim, ratio=ratio, overlap=(ratio == 4),
                rotate=False, compute_dtype=compute_dtype, device=device,
            )
            if ratio == 4:
                self.indexer: Indexer | None = Indexer(
                    dim, q_lora_rank, index_n_heads, index_head_dim, rope_dim,
                    index_topk, ratio, max_seq_len, compute_dtype=compute_dtype, device=device,
                )
            else:
                self.indexer = None
        else:
            self.compressor = None
            self.indexer = None

        self.output: SharedKeyValueMultiQueryAttention = SharedKeyValueMultiQueryAttention(
            n_heads, head_dim, rope_dim, o_groups, o_lora_rank, dim,
            compute_dtype=compute_dtype, device=device,
        )

        # Window buffer
        self.register_buffer(
            "window_cache",
            torch.zeros(window_size, head_dim, dtype=compute_dtype, device=device),
            persistent=False,
        )
        if self.ratio:
            # Global KV Cache
            self.register_buffer(
                "compress_cache",
                torch.zeros(max_seq_len // ratio, head_dim, dtype=compute_dtype, device=device),
                persistent=False,
            )
            if self.indexer is not None:
                # Indexer KV Cache (también propiedad de Attention)
                self.register_buffer(
                    "indexer_cache",
                    torch.zeros(max_seq_len // ratio, index_head_dim, dtype=compute_dtype, device=device),
                    persistent=False,
                )

        # Frecuencias RoPE compartidas por todos
        if self.ratio:
            freqs = precompute_rope_freqs(
                rope_dim, max_seq_len, original_seq_len, compress_rope_theta,
                rope_factor, beta_fast, beta_slow, device=device,
            )
        else:
            freqs = precompute_rope_freqs(
                rope_dim, max_seq_len, 0, rope_theta, 1.0, beta_fast, beta_slow, device=device,
            )
        self.register_buffer("freqs", freqs, persistent=False)

    def _write_window(self, kv: torch.Tensor, start_pos: int) -> None:
        # Buffer circular: solo importan los últimos W tokens.
        W = self.window_size
        S = kv.shape[0]
        if S > W:
            kv = kv[-W:]
            start_pos = start_pos + S - W
            S = W
        pos = (torch.arange(S, device=kv.device) + start_pos) % W
        self.window_cache[pos] = kv

    def _window_view(self, start_pos: int, S: int, kv: torch.Tensor):
        """Buffer lineal de ventana (previos + chunk actual) e índices [S, W],
        con -1 en posiciones anteriores al inicio de la secuencia."""
        W = self.window_size
        prev_len = min(W - 1, start_pos)
        if prev_len > 0:
            pos = torch.arange(start_pos - prev_len, start_pos, device=kv.device)
            window = torch.cat([self.window_cache[pos % W], kv], dim=0)
        else:
            window = kv
        base = start_pos - prev_len
        i = torch.arange(S, device=kv.device).view(-1, 1)
        j = torch.arange(W, device=kv.device).view(1, -1)
        abs_pos = start_pos + i - (W - 1) + j
        win_index = abs_pos - base
        win_index = torch.where(abs_pos >= 0, win_index, torch.full_like(win_index, -1))
        return window, win_index

    def reset(self) -> None:
        """Reinicia buffers/estados (nueva secuencia, start_pos=0)."""
        self.window_cache.zero_()
        if self.ratio:
            self.compress_cache.zero_()
            self.compressor.reset()
            if self.indexer is not None:
                self.indexer_cache.zero_()
                self.indexer.compressor.reset()

    @override
    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        # x  : [S, dim]
        # out: [S, dim]
        S = x.shape[0]
        x = x.to(self.compute_dtype)
        freqs = self.freqs
        freqs_now = freqs[start_pos: start_pos + S]

        # Query + Indexer shared projection
        shared, q = self.mqa_proj(start_pos, x, freqs_now)          # [S,1024], [S,64,512]

        # Window KV
        kv = self.window_kv(start_pos, x, freqs_now)                # [S,512]
        window, win_index = self._window_view(start_pos, S, kv)
        self._write_window(kv, start_pos)

        # KV comprimido + indice
        compressed = None
        compress_index = None
        if self.ratio:
            self.compressor(start_pos, x, freqs, kv_cache=self.compress_cache)
            n_comp = (start_pos + S) // self.ratio
            compressed = self.compress_cache[:n_comp]
            if compressed.shape[0] == 0:
                compressed, compress_index = None, None
            else:
                if self.indexer is not None:
                    compress_index = self.indexer(start_pos, x, shared, freqs,
                                                  self.indexer_cache)
                else:
                    compress_index = get_compress_topk_idxs(self.ratio, S, start_pos)

        # Atención
        return self.output(q, window, compressed, compress_index, win_index, freqs_now)
