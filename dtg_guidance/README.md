# DTG-Diff: Official Reproduction Toolkit

This repository provides the official implementation of:

DTG-Diff: Downstream Utility Guided Diffusion for Few-Shot Anomalous Time Series Generation

Accepted for publication in Knowledge-Based Systems.

DTG-Diff is a downstream-utility-guided diffusion framework for few-shot anomalous time-series generation. Instead of focusing solely on generation fidelity, DTG-Diff incorporates feedback from the downstream anomaly detector to guide the generation process toward anomalous samples that provide greater utility for subsequent anomaly detection.

This repository provides the core implementation and experimental pipeline described in the paper, including dataset processing, DTG-Diff training, downstream detector training, failure mining, downstream-utility guidance, evaluation, and TSTR/TSRTR protocols.

---

## Directory Structure

```text
dtg_guidance/
├── README.md
├── __init__.py
│
├── configs/                         # Dataset-specific configurations
│   ├── base.py                      # Shared configuration schema
│   ├── swat_config.py
│   ├── wadi_config.py
│   ├── psm_config.py
│   └── __init__.py
│
├── dataset_processing.py            # Dataset loading + windowing + masks + statistics
├── train_dtgdiff.py                 # DTG-Diff generator training
├── train_detector.py                # Downstream detector training
│
├── generator.py                     # DTG-Diff generator
├── diffusion.py                     # Gaussian diffusion utilities + sampling
│
├── failure_mining.py                # Detector failure / blind-spot mining
├── guidance.py                      # Discriminative + downstream task-gradient guidance
│
├── evaluation.py                    # Evaluation utilities
├── experiment_logging.py            # Experiment logging
├── visualization.py                 # Visualization utilities
│
├── pipeline_full.py                 # End-to-end reproduction pipeline
├── protocols.py                     # TSTR / TSRTR protocol runner
├── ablation.py                      # Ablation experiments
├── hyper_search.py                  # Hyperparameter search
├── multi_seed.py                    # Multi-seed experiments
└── batch_protocols.py               # Batch protocol execution
```

---

## One-Click Reproduction

For end-to-end reproduction, we recommend using `pipeline_full.py`.

Example:

```bash
python3 -m dtg_guidance.pipeline_full \
    --dataset SWaT \
    --protocol TSRTR \
    --save_visuals \
    --stats_plot
```

The pipeline integrates data processing, generator training, downstream detector training, anomalous sample generation, evaluation, experiment logging, and visualization.

---

## Outputs

A typical experiment produces the following outputs:

```text
outputs/
└── runs/
    └── <run_name>/
        ├── checkpoints/
        ├── vis/
        ├── split_stats.json
        └── summary.json

logs/
├── experiment.jsonl
└── experiment_meta.json
```

The exact output files may vary depending on the selected dataset, protocol, and experimental configuration.

---

## Release and Availability Notes

This repository is intended to support reproduction and further study of the methodology and experiments presented in the paper.

Resource availability follows the corresponding dataset licenses, usage policies, and release requirements. The current implementation focuses on the methodology and experimental settings used on the supported benchmark datasets.

Additional trained models, reference checkpoints, experimental artifacts, and supplementary resources may be provided progressively to facilitate further evaluation and reuse.

This repository will be updated as additional resources become available.

---

## Reproducibility

For consistent experimental reproduction:

- Use the dataset-specific configurations provided in `dtg_guidance/configs/`.
- Keep preprocessing and evaluation settings consistent across methods.
- Fix random seeds for multi-seed experiments.
- Use `pipeline_full.py` for end-to-end reproduction.
- Keep downstream detector settings consistent when comparing different generation methods.
- Report the mean and standard deviation over at least three random seeds.

---

## Citation

If you find this work or repository useful, please cite:

Ni Zhang, Hao Miao, Zefei Ning, and Li Wang.  
"DTG-Diff: Downstream Utility Guided Diffusion for Few-Shot Anomalous Time Series Generation."  
Knowledge-Based Systems, 2026.

The complete BibTeX entry will be updated once the final bibliographic information is available.
