

# LLEMA - LLM-guided Evolution for MAterials Design
Accelerating materials design via LLM-guided evolutionary search.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  
[![arXiv](https://img.shields.io/badge/arXiv-2510.22503-cs.LG-red.svg)](https://arxiv.org/abs/2510.22503)  
[![Hugging Face Paper](https://img.shields.io/badge/HuggingFace-Paper-yellow.svg)](https://huggingface.co/papers/2510.22503)

Official implementation of [“Accelerating Materials Design via LLM-Guided Evolutionary Search,” arXiv:2510.22503](https://arxiv.org/abs/2510.22503)

## 🧠 What is LLEMA?  

LLEMA is a unified framework that uses large language models (LLMs) + chemistry-informed evolutionary rules + surrogate predictors to discover novel, stable, synthesizable materials faster. It tackles the challenge of balancing conflicting objectives (e.g., bandgap vs. stability, conductivity vs. transparency) by combining reasoning, evolution and prediction.

---

## 🚀 Key Contributions  
- LLM-driven candidate generation under property constraints  
- Evolutionary memory loop with chemistry-informed operators  
- Multi-objective optimization using surrogate models  
- Benchmark suite of **14 materials discovery tasks** across electronics, energy and optics  
- Empirical results: higher hit rates, stronger Pareto fronts, broader diversity, out-performing CDVAE, G-SchNet, DiffCSP and LLMatDesign  

---

## 🔧 Getting Started

Requirements:
- Python 3.11+

Steps:
1) Clone this repository
```bash
git clone https://github.com/your-org/LLEMA.git
cd LLEMA/
```

2) Create and activate an environment

```bash
conda env create -f environment.yml  # creates env defined in file
conda activate llema
```

3) Install Python dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you plan to modify or run surrogate models locally, also clone their repos (see below). Otherwise, LLEMA can download the required pretrained weights on demand for ALIGNN.

---

## API Keys and Configuration

You must provide API keys before running the agent:

- OPENAI for LLM calls: `OPENAI_API_KEY`
- Materials Project for structure/property queries: `MATERIALS_PROJECT_API_KEY`

Recommended: Copy the example environment file and fill in the values.
```bash
cp env.example .env
# edit .env and set OPENAI_API_KEY and MATERIALS_PROJECT_API_KEY
```

Environment variables read by LLEMA (subset):
- `OPENAI_API_KEY` – used by the agent LLM interface
- `LLM_MODEL` – optional, defaults to `gpt-4o-mini`
- `MATERIALS_PROJECT_API_KEY` – used by property extraction utilities
- `SURROGATE_MODELS_DIR` – optional, defaults to `src/surrogate_models`

If not using a dotenv loader, you can also export them in your shell before running:
```bash
export OPENAI_API_KEY=...
export MATERIALS_PROJECT_API_KEY=...
```

---

## Surrogate Models

LLEMA integrates fast surrogate models to estimate materials properties during the search loop.

### ALIGNN (Atomistic Line Graph Neural Network)

- Pretrained models from JARVIS-DFT are automatically fetched when needed.
- Downloads are stored under `src/surrogate_models/alignn/alignn/` as `.zip` archives.
- For details on which archives are included and local customizations, see `src/surrogate_models/README.md`.

Clone (optional, for local development or customization):
```bash
cd src/surrogate_models
git clone https://github.com/usnistgov/alignn.git
```

### CGCNN (Crystal Graph Convolutional Neural Network)

- CGCNN can be used as an alternative or complementary surrogate.
- LLEMA includes minor output-format changes for clearer, property-specific CLI output.

Clone (optional, for local development or customization):
```bash
cd src/surrogate_models
git clone https://github.com/txie-93/cgcnn.git
```

See `src/surrogate_models/README.md` for more details on supported properties and output formats.

---

## ⚙️ Quick Start

Run the full benchmark suite via a bash script:
```bash
cd src
bash run_all_tasks.sh
```

## 📚 Citation

@article{Abhyankar2025LLEMA,
  title={Accelerating Materials Design via LLM-Guided Evolutionary Search},
  author={Abhyankar, Nikhil and Kabra, Sanchit and Desai, Saaketh and Reddy, Chandan K.},
  journal={arXiv preprint arXiv:2510.22503},
  year={2025}
}

## 📄 License

This repository is licensed under the MIT License.

## 📬 Contact Us
For any questions or issues, you are welcome to open an issue in this repo or contact us at nikhilsa@vt.edu and sanchit23@vt.edu.
