# vLLM-TAVA: Token-Adaptive VLM Agent Execution

**vLLM-TAVA** is a custom fork of the [vLLM](https://github.com/vllm-project/vllm) project, serving as the official codebase for the paper *Back to the Future: Load-Regulated VLM Agent Serving on a Single Machine*. 

---

## 🌟 Key Features

* **Model-Specific Token Pruning:** Implements specialized visual token pruning logic native to **LLaVA-NeXT** and **InternVL** architectures, significantly reducing KV cache pressure during extended agent sessions.
* **Request-Level Pruning Control:** Dynamically adjust the pruning ratio on a *per-request* basis. This allows agents to apply aggressive pruning for simple UI navigation and preserve high fidelity for complex image analysis within the same session.

## 🛠️ Installation

Because vLLM-TAVA modifies core routing, attention mechanisms, and introduces custom pruning kernels for LLaVA-NeXT/InternVL, it must be installed from source.

### Prerequisites
* OS: Linux (or Windows via WSL2)
* Python: 3.10
* GPU: Compute Capability 7.0 or higher (e.g., V100, T4, RTX20xx, A100, L4, H100)

### Build from Source

```bash
# 1. Clone the repository
git clone https://github.com/mirpri/vllm-TAVA.git
cd vllm-TAVA

# 2. Create and activate a virtual environment
conda create -n tava python=3.10 -y
conda activate tava

# 3. Install dependencies and build
pip install -e .
```

## Acknowledgements
This project is built upon the foundational work of the vLLM project. Our fork is based on tag v0.11.0. We thank the vLLM community for their continuous efforts in advancing efficient LLM serving.