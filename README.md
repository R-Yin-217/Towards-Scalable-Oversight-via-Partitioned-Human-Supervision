# Towards Scalable Oversight via Partitioned Human Supervision
This repository contains the official implementation of:

**"Towards Scalable Oversight via Partitioned Human Supervision"**  
📄 [arXiv:2510.22500](https://arxiv.org/abs/2510.22500)

---

## Repository Layout

```
.
├── Dataset/                 # put all CSV datasets here
├── Exp_1/                   # experiments on MMLU-Pro, MedQA-USMLE, GPQA, MATH-MC
├── Exp_2/                   # experiments on EDINET-Bench, EDINET-Bench Extended, Medical Abstracts
├── Exp_3/                   # ADAS runs for GPQA, MATH-MC, Medical Abstracts (AFlow users: see upstream repo)
├── Results/                 # outputs will be written here
└── README.md
```

---

## API Keys

We rely on OpenAI API for model inference. Please set your API key as follows:

```bash
export OPENAI_API_KEY="YOUR_KEY_HERE"
```

---

## Datasets

Place all datasets (CSV) under `Dataset/`. A typical layout:

```
Dataset/
├── MMLU_Pro_10.csv
├── MedQA_USMLE_4.csv
├── GPQA_Extended_4.csv
├── MATH_MC_5.csv
├── EDINET_Bench.csv
├── EDINET_Bench_Extended.csv
├── Medical_Abstract_5.csv
└── ...
```

### CSV Schema

All CSVs should follow:

```
Question, A, B, C, D, ... , Answer, Comp_label
```

- `Answer`: the **correct** option letter (e.g., `A`) for ordinary-label evaluation.  
- `Comp_label`: the **complementary** option letter (i.e., definitely **not** correct) for comp-label evaluation.

---

## Quick Start

### Experiment 1 (MMLU-Pro / MedQA-USMLE / GPQA / MATH-MC)

Minimal example:

```bash
python Exp_1/exp.py \
  --expr_name exp1_mmlu_pro \
  --data_filename Dataset/MMLU_Pro_10.csv \
  --num_multiple 10 \
  --method original
```

Change `--method comp` to use complementary labels. Replace `--data_filename` and `--num_multiple` to match your dataset.

---

### Experiment 2 (EDINET-Bench / EDINET-Bench Extended / Medical Abstracts)

Usage mirrors **Experiment 1**. Example:

```bash
python Exp_2/exp.py \
  --expr_name exp2_edinet \
  --data_filename Dataset/EDINET_Bench.csv \
  --num_multiple 4 \
  --method original
```

---

### Experiment 3 (ADAS; concise usage)

This pipeline runs **ADAS** on **GPQA**, **MATH-MC**, and **Medical Abstracts**.  
We keep usage succinct—just pick the preset matching your dataset. All other args use built-in defaults.

> Defaults (for reference): `temperature=0.5`, `model=gpt-4.1-nano`, `method=comp`, `n_repeat=1`, `valid_size=128`, `test_size=800`, multiprocessing enabled with `max_workers=48`.

### Preset one-liners

**Medical Abstracts (K=5)**

```bash
python Exp_3/exp.py --expr_name exp3_medabs --data_filename Dataset/Medical_Abstract_5.csv --num_multiple 5
```

Optional overrides (when needed):

- Switch model: `--model gpt-5-nano`
- Use ordinary labels: `--method original`
- Adjust validation/test size: `--valid_size 1152 --test_size 1000`
Other datasets (e.g., GPQA, MATH-MC) follow the same usage; see `Exp_3/exp.py` for full options.
---


## Results & Reproducibility

- Outputs (predictions, logs, summaries) are written under:
  ```
  Results/<expr_name>/
  ```
- To reproduce a run, fix `--shuffle_seed` and list your environment (e.g., `pip freeze > requirements-lock.txt`).

---

## AFlow Users

For AFlow-based workflows, please use the official implementation:  
[AFlow GitHub](https://github.com/FoundationAgents/AFlow)  

---

## Acknowledgments

This repository includes modified portions of code derived from **ADAS** , which is licensed under the Apache License 2.0. We thank the authors for releasing their work and insights.
[ADAS GitHub](https://github.com/ShengranHu/ADAS)

---

## License

This repository is licensed under the Apache License, Version 2.0.

This repository includes modified portions of code derived from
the **ADAS project**, which is licensed under the Apache License, Version 2.0.

Modifications and additional code © 2025 Ren Yin.

See [LICENSE](./LICENSE) for the full text of the license.

[ADAS GitHub Repository](https://github.com/ShengranHu/ADAS)

---
