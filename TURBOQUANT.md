# ik_llama.cpp + TurboQuant KV Cache

Branch: 

This branch combines two independent projects:

- **[ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)** — MTP (Multi-Token Prediction) support, hybrid model architectures (Qwen3-MTP, Gemma4, etc.), superior speculative decoding
- **[TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)** — TurboQuant KV cache quantization:  types reducing KV memory by 4–8×, enabling 262k context in standard GPU VRAM

## What this enables

Running Qwen3.6-27B-MTP at 262k context with:
- **MTP speculative decoding** ( flag): ~20% generation speedup at zero quality cost
- **TurboQuant KV cache** (): 262k context fits in ~20GB VRAM instead of ~40GB+

## Usage

```bash
./build/bin/llama-server \
  -m Qwen3.6-27B-MTP-Q4_K_M.gguf \
  --cache-type-k turbo3 \
  --cache-type-v turbo3 \
  --ctx-size 262144 \
  --gpu-layers 99 \
  --reasoning on \
  --jinja \
  -mtp --draft-max 1 --draft-p-min 0.0 \
  --port 7211
```

## Build

```bash
mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . -j 16 --target llama-server
```

## What was ported from TurboQuant

| File | Purpose |
|---|---|
|  | Core type definitions, quantize/dequantize device functions |
|  | Walsh-Hadamard Transform CUDA kernel |
|  | Inner quantization kernels |
|  | Matrix-vector multiply for TurboQuant weight types |
|  | Flash attention template instances (15 K/V type combinations) |
|  | , ,  enum values (42–46) |
|  | ,  constants |
|  |  entries for turbo types |
|  | Turbo WHT op registration |

## Status

- [x] Compiles with CUDA
- [x] turbo2/3/4 KV cache wired into fattn-vec dispatch (K scoring + V dequant)
- [ ] End-to-end inference test with MTP + turbo3

The turbo type definitions, CUDA kernels, and template instances are all present.
The remaining work is wiring the turbo KV types into ik_llama.cpp's  dispatch path
(different from turboquant's  which was a single combined file).
