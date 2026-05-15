# INSTA — Governance-Priced AI Liability Insurance

> **Team INSTA**
> Samsung Fire & Marine Insurance · Risk Management Competition 2026
> A dynamic premium model that prices AI governance directly into the rate.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![Model](https://img.shields.io/badge/LLM-Qwen2.5--7B--Instruct-orange)]()
[![Data](https://img.shields.io/badge/data-AIID%20CSETv1-green)]()
[![License](https://img.shields.io/badge/license-Research%20Use-lightgrey)]()

---

## 1. Overview

AI governance quality enters the premium formula as an actuarial input. The model takes 213 AIID incident × domain records labelled by Qwen2.5-7B-Instruct against the eight EU AI Act controls, plugs the resulting governance score into a Munich Re-calibrated frequency decay (γ = 1.314), and runs a 100,000-replicate compound-Poisson simulation. A firm that moves from Type A (no governance) to Type D (certified) pays 68.8% less premium on average across seven industries. The direction holds across the full Munich Re calibration band; the magnitude is reported with the base case.

### Headline numbers

| Metric                                            | Value                       |
| ------------------------------------------------- | --------------------------- |
| AIID raw incidents                                | 183 (→ 213 incident×domain) |
| Monte Carlo replicates (per scenario)             | 100,000                     |
| Scenario grid                                     | 4 governance types × 7 domains = 28 |
| Frequency-decay coefficient γ (Munich Re Cyber)   | 1.314                       |
| σ (log-normal severity, AIID 35-yr full)          | 4.351                       |
| σ (AIID 5-yr subset, cross-validation)            | 4.329 — Δ < 1%              |
| TVaR loading φ (Solvency II)                      | 0.15                        |
| Risk margin θ (Munich Re Cyber 2024)              | 0.25                        |
| Expense ratio e (FSS Korea + Lloyd's)             | 0.30                        |
| Type-D premium reduction (calibrated base)        | **68.8%**                   |
| Type-D premium reduction range (γ 0.357–2.629)    | 20% – 89%                   |

---

## 2. The four pillars

```
┌──────────────────────────────────┐   ┌──────────────────────────────────┐
│ Pillar 1                         │   │ Pillar 2                         │
│ 8-Control Failure Pattern        │   │ Governance ↔ Frequency           │
│  D Q T H M R C S                 │   │  λ_final = λ_global · e^(−γ·G)   │
└──────────────────────────────────┘   └──────────────────────────────────┘
┌──────────────────────────────────┐   ┌──────────────────────────────────┐
│ Pillar 3                         │   │ Pillar 4                         │
│ Dynamic Premium as Incentive     │   │ Four-Axis Risk Framework         │
│  π(G) = (E[L]+φ·TVaR)·(1+θ)/(1−e)│   │  Data × Algorithm × Train × Deploy│
└──────────────────────────────────┘   └──────────────────────────────────┘
```

| # | Pillar                  | Claim                                                          | Evidence                                                          |
|---|-------------------------|----------------------------------------------------------------|-------------------------------------------------------------------|
| 1 | Common Failure Pattern  | High-impact AI incidents share a governance failure signature. | AIID 213 × Qwen2.5-7B labelling → 8-control taxonomy              |
| 2 | Governance ↔ Frequency  | Governance has a significant relationship with incident λ.     | G<sub>ops</sub> from 213 records; γ from Munich Re Cyber 2024     |
| 3 | Dynamic Premium         | Governance-linked premium acts as an economic safety incentive.| 100K MC: 15.8% / 49.4% / **68.8%** for Type A → B / C / D         |
| 4 | Four-Axis Framework     | Risk geometry differs by Data × Algorithm × Train × Deploy.    | Industry (λ, μ) profiles diverge by orders of magnitude           |

---

## 3. Method

```
   AIID CSETv1  ──►  llm_eval.py     ──►  weights.csv  +  scores.csv
   (183 incidents)   (Qwen2.5-7B)         (8 controls / domain)
                                                  │
                                                  ▼
                                       simulation.py / .ipynb
                                       (100K Monte Carlo)
                                                  │
                                                  ▼
                                       Premium relativity matrix
                                       Sensitivity (σ, γ, φ)
                                       Per-capita premium (USD / KRW)
```

### Step 1. Frequency model · Poisson GLM

```math
N_i \sim \mathrm{Poisson}(\lambda_i), \qquad
\lambda_i \;=\; \lambda_{\mathrm{global}} \cdot e^{-\gamma \cdot G_{\mathrm{ops}}}
```

### Step 2. Severity model · log-normal (decay disabled)

```math
X_{ij} \sim \mathrm{Lognormal}\!\bigl(\log \mu_i - \tfrac{\sigma^2}{2},\;\sigma\bigr),
\qquad \mu_{\mathrm{eff}} = \mu_{\mathrm{industry}} \;(\text{no decay})
```

### Step 3. Aggregate loss → Solvency II premium

```math
L_i = \sum_{j=1}^{N_i} X_{ij},
\qquad
\pi(G) \;=\; \frac{E[L_i] \;+\; \phi \cdot \mathrm{TVaR}_{99}}{1 - e}\,(1 + \theta)
```

### Step 4. Governance score · four-axis composition

```math
G_{\mathrm{ops},D} \;=\;
   w_{\text{data},D}\,g_{\text{data}} +
   w_{\text{algo},D}\,g_{\text{algo}} +
   w_{\text{train},D}\,g_{\text{train}} +
   w_{\text{deploy},D}\,g_{\text{deploy}}
```

The per-domain weights `w_c(D)` come from the AIID label counts:

```math
w_c^{(D)} \;=\; q_c^{(D)} \,\Big/\, \sum_c q_c^{(D)}
```

where `q_c^(D)` is the fraction of AIID incidents in domain `D` that Qwen2.5-7B-Instruct flagged for control `c`.

---

## 4. Repository layout

```
.
├── README.md                              ← this file
├── llm_eval.py                            ← Stage A — LLM labelling pipeline
├── simulation.py                          ← Stage B — Monte Carlo (script form)
├── simulation.ipynb                       ← Stage B — Monte Carlo (notebook form)
├── aiid_taxonomy_CSETv1.csv               ← AIID raw taxonomy (input)
├── aiid_incident_scores.csv               ← per-incident G_ops (output)
└── aiid_domain_and_general_weights.csv    ← w_c per domain  (output)
```

| File                                  | Role                                                                       | Size      |
|---------------------------------------|----------------------------------------------------------------------------|-----------|
| `aiid_taxonomy_CSETv1.csv`            | AIID CSETv1 taxonomy raw export (input to `llm_eval.py`)                   | 858 KB    |
| `llm_eval.py`                         | Loads taxonomy, runs Qwen2.5-7B, derives `w_c` and `G_ops`                 | 444 LoC   |
| `simulation.py`                       | Headless Python script with the same logic as the notebook                 | 1,116 LoC |
| `simulation.ipynb`                    | Five sections: hyperparameters, scenarios, MC, plots, sensitivity          | 14 cells  |
| `aiid_incident_scores.csv`            | Per-incident binary flags + label set + `real_G_ops` deficiency            | 213 × 13  |
| `aiid_domain_and_general_weights.csv` | Row-normalised control weights for 7 domains + Others + General            | 9 × 9     |

`simulation.py` and `simulation.ipynb` carry the same hyperparameters, RNG seed (42), section structure, and outputs. The notebook is for reading the analysis alongside the charts; the script is for running it once headless.

### Domain × control weight matrix (`aiid_domain_and_general_weights.csv`)

|                        | D     | Q     | T     | H     | M     | R     | C     | S     |
|------------------------|-------|-------|-------|-------|-------|-------|-------|-------|
| Medical AI             | .129  | .151  | .140  | .108  | .129  | .129  | .140  | .075  |
| Autonomous Vehicle     | .070  | .172  | .172  | .130  | .130  | .135  | .126  | .065  |
| Robotics               | .070  | .163  | .174  | .105  | .128  | .128  | .151  | .081  |
| Financial Services     | .091  | .152  | .152  | .121  | .121  | .121  | .152  | .091  |
| Recruitment / Hiring   | .098  | .171  | .171  | .098  | .106  | .122  | .146  | .089  |
| Public Administration  | .129  | .152  | .152  | .114  | .083  | .129  | .144  | .098  |
| Education              | .134  | .134  | .146  | .146  | .110  | .110  | .134  | .085  |
| Others                 | .120  | .149  | .149  | .116  | .118  | .134  | .125  | .089  |
| **General (W)**        | .108  | .155  | .156  | .117  | .116  | .130  | .133  | .085  |

`D Data Governance · Q Risk & Quality · T Testing & Validation · H Human Oversight · M Monitoring & Logging · R Incident Response · C Compliance & Documentation · S Security & Robustness`

---

## 5. Reproduce locally

### 5.1 Environment

```bash
python -m venv .venv && source .venv/bin/activate

pip install torch transformers accelerate
pip install numpy pandas matplotlib seaborn scipy scikit-learn jupyter
```

GPU (CUDA or Apple MPS) speeds up Stage A. CPU works for both stages but Stage A becomes slow.

### 5.2 Stage A · LLM labelling

```bash
python llm_eval.py
```

- Qwen2.5-7B-Instruct downloads once (~15 GB).
- Outputs cache to `qwen_taxonomy_cache.json`; subsequent runs read the cache and finish in seconds.
- Final outputs: `aiid_domain_and_general_weights.csv`, `aiid_incident_scores.csv`.
- Sampling is deterministic: `temperature=0.1`, `do_sample=False`, `max_new_tokens=60`. Identical input → identical labels.

### 5.3 Stage B · Monte Carlo simulation

Two interchangeable entry points.

**Notebook** (inline charts, narrative)
```bash
jupyter notebook simulation.ipynb
```

Headless variant:
```bash
jupyter nbconvert --to notebook --execute simulation.ipynb --output simulation_run.ipynb
```

**Script** (headless, CI-friendly)
```bash
python simulation.py
```

Both paths produce the same premium matrix, sensitivity CSVs, and figure PNGs. The RNG seed is fixed at 42.

**Sections (shared between `.py` and `.ipynb`)**

| # | Section                       | Purpose                                                                          |
|---|-------------------------------|----------------------------------------------------------------------------------|
| 1 | Libraries & Hyperparameters   | Imports, constants (σ, γ, φ, θ, e), and SamsungGothicCondensedOTF font registration |
| 2 | Domain & Scenario Definition  | 7 AI domains × 4 governance types (A / B / C / D) with scenario-level G<sub>ops</sub> |
| 3 | Monte Carlo Simulation        | 100,000-replicate compound-Poisson → E[L], TVaR<sub>99</sub>, premium matrix     |
| 4 | Real Data Validation          | Closed-form premium re-applied to 183 actual AIID incidents                      |
| 5 | Sensitivity Analysis          | OAT perturbation of σ / γ / φ / θ / e + λ-source variant + CV                    |

### 5.4 Decisions baked into the model

- **Severity decay disabled.** `μ_eff = μ_industry`. Governance reduces incident frequency, not the loss size of a given incident, so adding a decay term on μ would not match the IBM 2024 data the parameter is sourced from.
- **Single labeller.** Qwen2.5-7B-Instruct alone, with no inter-annotator agreement. The closed eight-control vocabulary and deterministic decoding bound the subjectivity.
- **σ = 4.351.** Estimated on the full 35-year AIID set, cross-validated against the 5-year subset (σ = 4.329, Δ < 1%).
- **Direction robust, magnitude γ-dependent.** Moving γ across the Munich Re band (0.357 / 1.314 / 2.629) shifts the Type-D reduction to 20% / 69% / 89%. The sign of the effect does not change; the size does.

---

## 6. Key outputs

- Premium relativity heatmap (domain × type, normalised against Type A in each domain).
- Per-capita annual premium for Type D — Medical $0.17 / Financial $0.23 / Robotics $9.49 / AV $2.32 / Education $0.23 / Public Admin $0.02 / Recruitment $0.11. KRW conversion at ₩1,380 / USD.
- (λ, μ) risk-geometry plot: simulated profile vs AIID empirical proxy agree on rank order across seven industries.
- G<sub>ops</sub> × premium scatter on the 183 real AIID records, confirming the exponential decay holds without re-tuning.
- Tornado + ±band sensitivity: CV(all 7) = 62.4%, CV(Grade-A only) = 24.6% across 13 OAT scenarios.

---

## 7. Regulatory alignment

| Domain                | Statutory concern                            | Insurance implication                        |
|-----------------------|----------------------------------------------|----------------------------------------------|
| Medical AI            | Patient safety, diagnostic accuracy          | Mandatory governance audit before issuance   |
| Autonomous Vehicle    | Road safety, liability allocation            | Multi-party policy (manufacturer + operator) |
| Robotics              | Workplace injury, occupational health        | Bundling with workers' compensation          |
| Financial Services    | Algorithmic bias, fair lending               | Compliance audit aligned with FSS rules      |
| Recruitment / Hiring  | Discrimination, fair employment              | Sliding premium on bias-test scores          |
| Public Administration | Citizen rights, due process                  | Government-backstop policy structure         |
| Education             | Student evaluation, equality of opportunity  | School-district group policy                 |

- Korea AI Basic Act (Act on the Development of Artificial Intelligence and Establishment of Trust): enacted December 2024, in force 22 January 2026.
- EU AI Act, Regulation (EU) 2024/1689, Articles 9–15 (Risk Mgmt · Data Governance · Logging · Human Oversight · Accuracy & Robustness · Cybersecurity).
- NIST AI RMF 1.0 (2023), ISO/IEC 42001:2023, Solvency II Directive 2009/138/EC, FSS Korea and EIOPA reporting templates.

---

## 8. References (selected)

**Actuarial & insurance**
- Goldburd, M., Khare, A., Tevet, D. *Generalized Linear Models for Insurance Rating* (CAS Monograph No. 5).
- Klugman, S. A., Panjer, H. H., Willmot, G. E. (2019). *Loss Models: From Data to Decisions*, 5th ed. Wiley.
- Frees, E. W. *Loss Data Analytics*, Ch. 5 — Aggregate Loss Models.
- Artzner, P., Delbaen, F., Eber, J.-M., Heath, D. (1999). Coherent Measures of Risk. *Mathematical Finance* 9(3), 203–228.
- McNeil, A. J., Frey, R., Embrechts, P. (2015). *Quantitative Risk Management*, revised ed. Princeton University Press.

**AI risk & insurance literature**
- Lior, A. (2022). Insuring AI: The Role of Insurance in AI Regulation. *Harvard Journal of Law & Technology*, 35(2).
- Hickok, M. (17 Dec 2025). *Why insurance companies should encourage solid AI risk management instead of excluding it.* OECD.AI Wonk.
- McGregor, S. (2021). Preventing Repeated Real-World AI Failures by Cataloging Incidents — The AI Incident Database. AAAI / IAAI-21, 35(17), 15458–15463. DOI 10.1609/aaai.v35i17.17817.
- Madry, A. et al. (2018). Towards Deep Learning Models Resistant to Adversarial Attacks. ICLR 2018.

**Data sources**
- AI Incident Database (incidentdatabase.ai) and CSETv1 taxonomy (Georgetown).
- Stanford HAI, *AI Index Report 2025* (2024 reported AI incidents = 233, +56% YoY).
- IBM Security & Ponemon Institute, *Cost of a Data Breach Report 2024* (industry severity μ).
- Munich Re, *Cyber Insurance: Risks and Trends 2024* (γ, θ calibration).

**Representative AI incidents**
- Moffatt v. Air Canada, 2024 BCCRT 149 — chatbot misinformation.
- Goseong agricultural-packaging plant industrial-robot fatality, 7 Nov 2023.
- Arup Group / Hong Kong USD 25.6 M deepfake CFO video-conference scam, Jan–May 2024.
- Cruise robotaxi pedestrian-dragging incident, San Francisco, 2 Oct 2023.
- Benavides Estate / Angulo v. Tesla, S.D. Fla. — fatal Autopilot crash; jury verdict 1 Aug 2025, USD 243 M (USD 43 M compensatory + USD 200 M punitive).

---

## 9. AI tools disclosure

| Step                     | Tool                                            | Human role                                                       |
|--------------------------|-------------------------------------------------|------------------------------------------------------------------|
| Incident labelling       | Qwen2.5-7B-Instruct (open-source, local)        | Prompt design, label validation, post-hoc review                 |
| Statistical inference    | NumPy compound-Poisson simulator, 100K replicates | Model specification, parameter sourcing, sensitivity analysis  |
| Calibration (γ, σ, μ, λ) | None — manual extraction from literature        | Munich Re, IBM, AIID, FSS, Solvency II, Klugman 2019             |
| Visualisation            | matplotlib / seaborn (deterministic)            | Chart selection, annotation, palette                             |
| Slide drafting           | Anthropic Claude (drafting assistant)           | Final copy, structure, fact-check against codebase               |

All quantitative results reproduce from `llm_eval.py` and `simulation.py` / `simulation.ipynb`. AI tools labelled data and accelerated computation; the underwriting framework itself has no AI in the loop.

---

## 10. Team & competition

- **Team — INSTA**
- **Competition — Samsung Fire & Marine Insurance Risk Management Competition 2026**
- **Topic — AI Governance Insurance Service**
- **Contact — open an issue on this repository**

> Insurance has long shaped safety standards: fire codes, seat belts, building inspections.
> Governance-priced AI insurance applies the same mechanism to the most consequential
> technology of this generation.

---

## 11. License & citation

Non-commercial research and competition review use is permitted. To cite this work:

```bibtex
@misc{insta2026,
  title  = {INSTA --- Governance-Priced AI Liability Insurance},
  author = {Team INSTA},
  year   = {2026},
  note   = {Samsung Fire \& Marine Insurance Risk Management Competition 2026},
  url    = {https://github.com/riskmanagemant-INSTA/code}
}
```
