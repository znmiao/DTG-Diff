# DTG-Diff Reproduction Toolkit (Paper-Grade)

This folder provides a toolkit for ‘’DTG‑Diff: Downstream Task‑Guided Few‑Shot Time‑Series Anomaly Generation‘’. 

This toolkit implements the core methods described in the paper. Certain pieces are omitted or stubbed for privacy and licensing reasons; see **“Release & Disclosure Notes”** below.

---

 Directory Structure

```
 dtg_guidance/
 ├── README.md                       
 ├── __init__.py                      
 ├── configs/                         # Dataset‑specific configs 
 │   ├── base.py                      # Shared config schema
 │   ├── swat_config.py
 │   ├── wadi_config.py
 │   ├── psm_config.py
 │   └── __init__.py
 │
 ├── dataset_processing.py            # Dataset loading + windowing + masks + stats
 ├── train_dtgdiff.py                 # Standalone generator training (base + adapt)
 ├── train_detector.py                # Standalone detector training
 │
 ├── generator.py                     # DTG‑Diff generator 
 ├── diffusion.py                     # Gaussian diffusion utilities + sampling
 │
 ├── failure_mining.py                # Blind‑spot mining
 ├── guidance.py                      # Dual guidance: discriminative + task‑gradient
 │
 ├── evaluation.py                   
 ├── experiment_logging.py            
 ├── visualization.py                
 │
 ├── pipeline_full.py         
 ├── protocols.py                     # TSTR / TSRTR protocol runner
 ├── ablation.py                      
 ├── hyper_search.py                 
 ├── multi_seed.py                    
 └── batch_protocols.py              

---



 One‑Click Reproduction (Recommended)

```
python3 -m dtg_guidance.pipeline_full \
  --dataset SWaT \
  --protocol TSRTR \
  --save_visuals \
  --stats_plot
```

Outputs are written to:
```
outputs/runs/<dataset>/<timestamp>/
  logs/experiment.jsonl
  logs/experiment_meta.json
  checkpoints/
  vis/
  split_stats.json
  summary.json
```

---

Release & Disclosure Notes (Why some pieces are not fully public)

We intentionally withhold a small subset of training infrastructure and pre‑trained artifacts in this repository to comply with:

1) Data usage agreements & privacy**: Several datasets used for validation are governed by restrictive licenses or internal data‑use policies. We therefore do not distribute certain pipeline components that encode dataset‑specific preprocessing rules beyond the public benchmarks.

2) Security considerations**: The blind‑spot‑guided generation module can synthesize adversarial failure cases for safety‑critical systems. We provide the method and reproducible experiments for public benchmarks, but we do not release specialized tooling that could be misused against operational systems.

3) Reproducibility audits**: All results reported in the paper were audited against internal baselines and compliance constraints. We are preparing a clean, policy‑compliant release of pre‑trained generator checkpoints and detector baselines.

Planned release (post‑publication):
- Pre‑trained DTG‑Diff generator weights (per dataset)
- Reference detector checkpoints (CutAddPaste baseline)
- Full training recipes and exact hyperparameter sweeps
- Exported synthetic sample archives for TSTR/TSRTR

We will update this README with download links and checksums once the release is cleared.

---

 Reproducibility Tips
- Always fix seeds for multi‑seed runs: `dtg_guidance/multi_seed.py`
- Prefer `pipeline_full.py` for end‑to‑end reproduction
- Report mean/std across ≥3 seeds for paper‑grade results

---

 Citing
If you use this toolkit, please cite the associated paper (citation text will be added after publication).
