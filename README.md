# Experimentos con DeepSeek V4

Entorno autocontenido para la **atención de DeepSeek-V4-Flash**. El objetivo es
probar piezas de la Compressed Sparse Attention de DeepSeek-V4-Flash.


## Instalación

El entorno se gestiona con `uv`. Aunque está devenv para poder correr todo en NixOS.


## Uso


**1. Descarga los tensores de atención de las capas CSA** 

```bash
python scripts/download_csa_layers.py \
    --output-dir data/deepseek-v4-flash-csa \
    --layers 2,4,6            # opcional; por defecto = todas las capas CSA
```

**2. Convierte a un único dtype y formatea a la nueva implementación**

```bash
python scripts/convert_attention_weights.py \
    --input-dir  data/deepseek-v4-flash-csa \
    --output-dir data/deepseek-v4-flash-csa-converted \
    --dtype bf16
```

**3. Ejecuta un target e inspecciona el JSON:**

```bash
python scripts/run.py module/indexer 10 2 3 1 3 4 2 --range 0.4 \
    --size real --dtype bf16 \
    --weights data/deepseek-v4-flash-csa-converted/layer_0002.safetensors \
    --output idx
```

## Casos de pruebas (`scripts/run.py`)

```bash
python scripts/run.py <target> <chunks...> [opciones]
```

- **`<target>`** — `pipeline`, `module/<nombre>`, `kernel/<nombre>`, `<nombre>`
- **`<chunks...>`** — corre el modelo con secuencias de tokens de la longitud especificada: 10 2 3 1 3 4 2.

### Targets

| Target | Pieza |
|--------|-------|
| `pipeline` / `module/attention` | Capa de atención completa |
| `module/mqa_proj` | Multi-Query Projection |
| `module/window_kv` | Window KV |
| `module/token_compressor` | Token Level Compressor (CSA/HCA) |
| `module/indexer_compressor` | Indexer Token Level Compressor |
| `module/indexer` | Indexer (Cheap MQA + top-k) |
| `module/sparse_attention` | Flash attention (ventana + gather + sink) |
| `module/output` | Reverse RoPE + reshape + `wo_a`/`wo_b` |
| `kernel/linear` | `Linear` |
| `kernel/rmsnorm` | `RMSNorm` |
| `kernel/weighted_sum` | `WeightedSum` (+ `ape`) |
| `kernel/hadamard` | `Hadamard` (FWHT) |
| `kernel/rope` | `RoPE` (+ `precompute_rope_freqs`) |
| `kernel/safe_attention` | `SafeMultiQueryAttention` |
| `kernel/cheap_attention` | `CheapMultiQueryAttention` |

### Opciones

| Opción | Significado |
|--------|-------------|
| `--range X` | rango de entrada simétrico `[-X, X]` al samplear tokens aleatorios |
| `--range A B` | rango de entrada `[A, B]` (default: `1` → `[-1, 1]`) |
| `-o, --output F` | prefijo de los JSON de salida (default `output`) |
| `--seed N` | semilla; **aleatoria por defecto** (se imprime la elegida) |
| `--weights FILE` | `.safetensors` de la capa a cargar; **obligatorio** en `pipeline`/`module/*` (las claves de una pieza se recortan por su prefijo) |
| `--random` | usa pesos sintéticos en vez de `--weights` |
| `--size tiny\|real` | preset de config (`tiny` para pruebas rápidas en CPU, `real` = dims de V4 flash) |
| `--layer-type csa\|hca\|window` | tipo de atención (ratio 4 / 128 / 0) |
| `--dtype fp16\|fp32\|bf16` | dtype de cómputo |
| `--device` | `cuda` si está disponible, si no `cpu` |
| `--ratio N` | sobreescribe el ratio del compresor |

### Salida

Cada ejecución escribe tres ficheros JSON legibles:

- **`<output>.in.json`** — la secuencia de entrada: un vector por token, más la
  config, el schedule, la semilla y el rango.
- **`<output>.out.json`** — la salida procesada **por token, paso a paso**
  (kind, longitud, posición de inicio, filas de salida).
- **`<output>.state.json`** — el estado de **todos los buffers** tras el run
  (shape, dtype, `finite` y valores).

### Ejemplos

```bash
# capa completa con los pesos reales
python scripts/run.py pipeline p12 p8 d1 d1 d4 --size real --dtype bf16 \
    --weights data/deepseek-v4-flash-csa-converted/layer_0002.safetensors --output run1

# una pieza, rango simétrico [-0.4, 0.4]
python scripts/run.py module/indexer 10 2 3 1 3 4 2 --range 0.4 --random --output idx

# dos conversaciones independientes (reset en medio)
python scripts/run.py pipeline p16 d4 r p10 d6 --random --output conv

# un kernel de bajo nivel (no usa pesos de capa)
python scripts/run.py kernel/rope 8 --dtype fp32 --output rope
```
