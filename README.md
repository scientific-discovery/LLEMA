

# LLEMA - LLM-guided Evolution for MAterials Design (ICLR 2026)
Accelerating materials design via LLM-guided evolutionary search.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![arXiv](https://img.shields.io/badge/arXiv-2510.22503-b31b1b.svg)](https://arxiv.org/abs/2510.22503) [![Hugging Face Paper](https://img.shields.io/badge/HuggingFace-Paper-yellow.svg)](https://huggingface.co/papers/2510.22503) [![Hugging Face Dataset](https://img.shields.io/badge/HuggingFace-Dataset-blue.svg)](https://huggingface.co/datasets/nikhilsa/LLEMABench)

Official implementation of [“Accelerating Materials Design via LLM-Guided Evolutionary Search”](https://arxiv.org/abs/2510.22503)

## 🧠 What is LLEMA?  

LLEMA is a unified framework that uses large language models (LLMs) + chemistry-informed evolutionary rules + surrogate predictors to discover novel, stable, synthesizable materials faster. It tackles the challenge of balancing conflicting objectives (e.g., bandgap vs. stability, conductivity vs. transparency) by combining reasoning, evolution, and prediction.

## 🚀 Key Contributions  
- LLM-driven candidate generation under property constraints  
- Evolutionary memory loop with chemistry-informed operators  
- Multi-objective optimization using surrogate models  
- Benchmark suite of **14 materials discovery tasks** across electronics, energy, aerospace, coatings, and optics  
- Empirical results: higher hit rates, stronger Pareto fronts, and broader diversity.

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

To run surrogate models locally, clone their repos (see below).

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

**Note:** The `src/agent/config.py` file contains run-specific information such as iteration limits, memory settings, and multi-island configuration parameters.

## Surrogate Models

LLEMA integrates fast surrogate models to estimate materials properties during the search loop.

### ALIGNN (Atomistic Line Graph Neural Network)

- Pretrained models from JARVIS-DFT are to be downloaded and stored under `src/surrogate_models/alignn/alignn/` as `.zip` archives.
- For details on which archives are included and local customizations, see `src/surrogate_models/README.md`.

```bash
cd src/surrogate_models
git clone https://github.com/usnistgov/alignn.git
```

### CGCNN (Crystal Graph Convolutional Neural Network)

- CGCNN can be used as an alternative or complementary surrogate.
- LLEMA includes minor output-format changes for clearer, property-specific CLI output.

```bash
cd src/surrogate_models
git clone https://github.com/txie-93/cgcnn.git
```

See `src/surrogate_models/README.md` for more details on supported properties and output formats.

## ⚙️ Quick Start

Run the full benchmark suite via a bash script:
```bash
cd src
bash run_all_tasks.sh
```

## 📊 Evaluation

LLEMA provides tools to evaluate CIF files for validity (property constraints) and stability analysis on the taks in LLEMABench. This section describes how to use these evaluation scripts.

### Validity Analysis

The `calculate_validity.py` script evaluates CIF files against task-specific property constraints to determine if they meet the requirements for a given materials discovery task.

**Usage:**
```bash
cd src
conda activate llema  # Ensure the mat_sci environment is activated
python calculate_validity.py --tasks <task_name> [options]
```

**Examples:**
```bash
# Evaluate CIF files for a specific task
python calculate_validity.py --tasks "Hard, Stiff Ceramics"

# Evaluate for all available tasks
python calculate_validity.py --tasks all
```

**Arguments:**
- `--tasks`: Task name(s) to evaluate. Use `"all"` to process all tasks, or specify one or more task names.
- `--cif-dir`: Directory containing CIF files to process (default: `example`)
- `--output-dir`: Output directory for results (default: auto-generated with timestamp in `validity_output/`)

**Output:**
- Results are saved in `validity_output/property_output_<timestamp>/` directory
- Each task generates a `results_<task_name>.jsonl` file containing:
  - Compound formula
  - Calculated property values (band gap, formation energy, bulk modulus, etc.)
  - Categorical constraint results (earth_abundant, non_toxic, etc.)
  - Successful and failed constraint checks
  - Materials API usage flag

### Stability Analysis

The `calculate_stability.py` script analyzes the thermodynamic stability of candidates from validity analysis results by calculating energy above hull and formation energy.

**Usage:**
```bash
cd src
conda activate llema  # Ensure the mat_sci environment is activated
python calculate_stability.py --task <task_name> [options]
```

**Arguments:**
- `--task` or `-t`: Specific task name to analyze (required)
- `--max-samples` or `-n`: Maximum number of samples to process per task
- `--quiet` or `-q`: Reduce output verbosity (only show summary statistics)
- `--output-dir`: Specific validity output directory to process (default: latest)

**Output:**
- Summary statistics are saved in `stability_output/stability_summary_<timestamp>.json`
- The JSON file contains:
  - Overall statistics: total candidates, valid/invalid counts, stability breakdown (stable/marginally stable/unstable/unknown)
  - Task-specific breakdown with detailed statistics
  - Energy above hull calculation success rates
  - Materials API and surrogate model usage statistics

**Note:** The stability analysis script automatically searches for results in `validity_output/property_output_*` directories and maps CIF files from the `example` directory (or the directory specified during validity analysis).

## 📚 Citation
```
@inproceedings{abhyankar2026llema,
        title={LLEMA: Accelerating Materials Design via {LLM}-Guided Evolutionary Search},
        author={Abhyankar, Nikhil and Kabra, Sanchit and Desai, Saaketh and Reddy, Chandan K},
        booktitle={The Fourteenth International Conference on Learning Representations (ICLR)},
        year={2026},
        url={https://openreview.net/forum?id=TIqzhBvCNB}
}
```
## 📄 License

This repository is licensed under the MIT License.

## 📬 Contact Us
For any questions or issues, you are welcome to open an issue in this repo or contact us at nikhilsa@vt.edu and sanchit23@vt.edu.
