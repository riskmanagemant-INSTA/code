"""
=============================================================================
AI Governance Taxonomy Labeling via LLM
=============================================================================
Purpose : Label AIID incident records with governance failure controls
          (D/Q/T/H/M/R/C/S) using a local Qwen2.5-7B-Instruct model,
          then derive domain-specific governance weights (w_c) used in
          the downstream insurance premium simulation.

Pipeline :
  [1] Load AIID taxonomy CSV  →  map domain  →  build rich context
  [2] LLM labels each incident with failed governance controls
  [3] Compute w_c = q_c / Σq_c  per domain and globally
  [4] Compute per-incident G_ops deficiency score
  [5] Export CSV for simulation notebook

Outputs  :
  aiid_domain_and_general_weights.csv   — w_c weights for simulation
  aiid_incident_scores.csv              — per-incident G_ops scores

References :
  NIST AI RMF (2023); EU AI Act (2024, Articles 9-20);
  ISO/IEC 42001 (2023); Partnership on AI — AIID Database
=============================================================================
"""

import os
import json
import re
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# 0. Configuration
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen2.5-7B-Instruct"
CACHE_FILE  = "qwen_taxonomy_cache.json"
SAMPLE_SIZE = 99999   # Set to e.g. 300 to analyse a subset

# Governance control codes and full names (NIST AI RMF / EU AI Act mapping)
CONTROLS = ['D', 'Q', 'T', 'H', 'M', 'R', 'C', 'S']
CONTROL_NAMES = {
    'D': 'Data Governance',
    'Q': 'Risk & Quality Management',
    'T': 'Testing & Validation',
    'H': 'Human Oversight',
    'M': 'Monitoring & Logging',
    'R': 'Incident Response',
    'C': 'Compliance & Documentation',
    'S': 'Security & Robustness',
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. Model Loading
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[0] Loading model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype="auto", device_map="auto"
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Data Loading and Preprocessing (CSET Taxonomy)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Loading data and building unified incident contexts...")
df = pd.read_csv("aiid_taxonomy_CSETv1.csv")


def map_to_standard_domain(row: pd.Series) -> str:
    """
    Map a raw AIID taxonomy row to one of seven standardised application
    domains based on keyword matching across multiple metadata fields.
    """
    sector = str(row.get('Sector of Deployment', '')).lower()
    task   = str(row.get('AI Task', '')).lower()
    title  = str(row.get('title', '')).lower()
    desc   = str(row.get('AI System Description', '')).lower()
    combined = f"{sector} {task} {title} {desc}"

    if any(k in combined for k in [
        'autonomous vehicle', 'self-driving', 'tesla', 'uber av',
        'car', 'traffic', 'cruise']):
        return "Autonomous Vehicle"

    if any(k in combined for k in [
        'medical', 'health', 'clinical', 'diagnosis',
        'hospital', 'patient', 'doctor']):
        return "Medical AI"

    if any(k in combined for k in [
        'finance', 'bank', 'insurance', 'credit',
        'stock', 'trading', 'loan']):
        return "Financial Services"

    if any(k in combined for k in [
        'hiring', 'recruitment', 'resume', 'employment',
        'job applicant', 'hr']):
        return "Recruitment / Hiring"

    if any(k in combined for k in [
        'robot', 'arm', 'manufacturing', 'factory', 'industrial']):
        return "Robotics"

    if any(k in combined for k in [
        'government', 'public sector', 'police', 'court',
        'administrative', 'surveillance', 'facial recognition']):
        return "Public Administration"

    if any(k in combined for k in [
        'education', 'student', 'school', 'grading',
        'university', 'exam']):
        return "Education"

    return "Others"


def create_comprehensive_context(row: pd.Series) -> str:
    """
    Concatenate all available taxonomy fields into a structured prompt
    context so the LLM can reason about technical, operational, and
    testing dimensions of each incident simultaneously.
    """
    def g(col):
        return str(row.get(col, 'Unknown'))

    return f"""
    [Technical Background]
    - AI System & Task  : {g('AI System')} / {g('AI Task')}
    - Data Inputs       : {g('Data Inputs')}
    - Tools / Methods   : {g('AI tools and methods')}
    - Autonomy Level    : {g('Autonomy Level')}

    [Operational Context]
    - Sector            : {g('Sector of Deployment')}
    - Public Sector     : {g('Public Sector Deployment')}
    - Quality Control   : {g('Quality Control')}

    [Testing History]
    - Producer (Controlled / Operational) :
        {g('Producer Test in Controlled Conditions')} /
        {g('Producer Test in Operational Conditions')}
    - User (Controlled / Operational) :
        {g('User Test in Controlled Conditions')} /
        {g('User Test in Operational Conditions')}

    [Incident Details]
    - Description  : {g('AI System Description')}
    - Harm Domain  : {g('Harm Domain')}
    - Technical Notes : {g('Notes (AI Functionality and Techniques)')}
    """


# Build incident list
INCIDENTS = []
for _, row in df.iterrows():
    INCIDENTS.append({
        "id":           int(row['incident_id']),
        "title":        str(row['title']),
        "domain":       map_to_standard_domain(row),
        "full_context": create_comprehensive_context(row),
    })

if len(INCIDENTS) > SAMPLE_SIZE:
    INCIDENTS = random.sample(INCIDENTS, SAMPLE_SIZE)

print(f"    Total incidents loaded: {len(INCIDENTS)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. LLM Labeling Function
# ─────────────────────────────────────────────────────────────────────────────
def label_with_hf(title: str, context: str, domain: str) -> list[str]:
    """
    Submit an incident to the local LLM and extract the set of governance
    controls whose absence contributed to the incident.

    Returns a deduplicated list of control codes (subset of CONTROLS).
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional AI Risk Auditor. "
                "Analyse the provided technical context to identify governance failures."
            ),
        },
        {
            "role": "user",
            "content": f"""
Analyse this AI incident in the [{domain}] domain to identify failed controls.

Title: {title}
{context}

Task:
Select ALL failed controls from the list below.

  D  Data Governance     — failure in data bias, privacy, or input management
  Q  Risk & Quality      — failure in overall quality control or risk assessment
  T  Testing             — failure or absence of testing (check Producer/User Test fields)
  H  Human Oversight     — failure in human control (check Autonomy Level)
  M  Monitoring          — failure to detect or log the issue in real-time
  R  Incident Response   — failure to respond or mitigate the incident effectively
  C  Compliance          — failure in documentation or regulatory alignment
  S  Security            — failure in robustness against attacks or technical errors

Return format:
Controls: [D, T, ...]
Reason: (one sentence)
""",
        },
    ]

    prompt  = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    outputs = pipe(
        prompt,
        max_new_tokens=60,
        do_sample=False,
        temperature=0.1,
        pad_token_id=tokenizer.eos_token_id,
    )
    res_text = outputs[0]['generated_text'].split("assistant\n")[-1]
    labels   = list(set(re.findall(r"[DQTHMRCS]", res_text.split("Controls:")[-1])))
    return labels

# ─────────────────────────────────────────────────────────────────────────────
# 4. Batch Labeling with Disk Cache
# ─────────────────────────────────────────────────────────────────────────────
try:
    with open(CACHE_FILE, "r") as f:
        cache = json.load(f)
except FileNotFoundError:
    cache = {}

print("\n[2] Running context-aware governance labeling...")
arc_rows = []

for i, inc in enumerate(INCIDENTS):
    key = str(inc['id'])
    if key in cache:
        labels = cache[key]
    else:
        labels = label_with_hf(inc['title'], inc['full_context'], inc['domain'])
        cache[key] = labels
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)

    row = {c: 1 if c in labels else 0 for c in CONTROLS}
    row.update({
        "id":     inc["id"],
        "domain": inc["domain"],
        "title":  inc["title"],
        "labels": ",".join(labels),
    })
    arc_rows.append(row)
    print(f"  {i+1:3d}/{len(INCIDENTS)} | {inc['title'][:42]:42s} → {labels}")

arc_df = pd.DataFrame(arc_rows)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Governance Weight Computation
# ─────────────────────────────────────────────────────────────────────────────
'''print("\n[3] Computing governance weights (w_c) ...")

# --- Domain-level weights ---------------------------------------------------
# q_c = fraction of incidents in domain d where control c was absent
domain_stats      = arc_df.groupby('domain')[CONTROLS].mean()
domain_weights_df = domain_stats.div(domain_stats.sum(axis=1), axis=0).fillna(0)

# --- Global weights (pooled across all domains) ----------------------------
global_stats   = arc_df[CONTROLS].mean()
global_weights = (global_stats / global_stats.sum()).fillna(0)

# --- Merge into a single dictionary ----------------------------------------
domain_weights_dict           = domain_weights_df.to_dict(orient='index')
domain_weights_dict['General'] = global_weights.to_dict()

print(f"\n  Domains analysed : {len(domain_weights_df)}")
print("\n  Global weights (w_c) :")

for c in sorted(global_weights, key=global_weights.get, reverse=True):
    bar = '█' * int(global_weights[c] * 50)
    print(f"    {c}  {CONTROL_NAMES[c]:<30s}: {global_weights[c]:.3f}  {bar}")
'''

# ─────────────────────────────────────────────────────────────────────────────
# 5. Governance Weight Computation
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Computing governance weights (w_c) ...")

# 디버깅용: 실제 라벨링된 데이터가 있는지 확인
total_labels_found = arc_df[CONTROLS].sum().sum()
print(f"    Total labels detected across all incidents: {total_labels_found}")

if total_labels_found == 0:
    print("    ⚠️ 경고: 라벨링된 데이터가 없습니다. 모든 가중치를 균등하게 배분합니다.")
    # 모든 컨트롤에 대해 균등 가중치 부여 (방어 코드)
    global_weights = pd.Series(1.0 / len(CONTROLS), index=CONTROLS)
    # 도메인 통계도 균등하게 설정
    domain_weights_df = pd.DataFrame(1.0 / len(CONTROLS), index=arc_df['domain'].unique(), columns=CONTROLS)
else:
    # --- 기존 로직 ---
    domain_stats = arc_df.groupby('domain')[CONTROLS].mean()
    # sum이 0인 경우를 대비해 fillna(0) 처리
    domain_weights_df = domain_stats.div(domain_stats.sum(axis=1), axis=0).fillna(1.0 / len(CONTROLS))

    global_stats = arc_df[CONTROLS].mean()
    denom = global_stats.sum()
    if denom == 0:
        global_weights = pd.Series(1.0 / len(CONTROLS), index=CONTROLS)
    else:
        global_weights = (global_stats / denom).fillna(0)

# --- 정렬 및 출력 (수정된 부분) ---
print(f"\n  Domains analysed : {len(domain_weights_df)}")
print("\n  Global weights (w_c) :")

# 안전하게 정렬 (NaN이 있어도 오류가 나지 않도록)
sorted_weights = global_weights.sort_values(ascending=False)

for c, v in sorted_weights.items():
    bar = '█' * int(v * 50)
    print(f"    {c}  {CONTROL_NAMES[c]:<30s}: {v:.3f}  {bar}")
    
# ─────────────────────────────────────────────────────────────────────────────
# 6. Paste-ready Output for Simulation Notebook
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("DOMAIN_WEIGHTS for simulation notebook (copy-paste ready):")
print("=" * 65 + "\n")
for dom, weights in domain_weights_df.items():
    formatted = {k: round(v, 3) for k, v in weights.items()}
    print(f"    '{dom}': {formatted},")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Visualisation — Domain × Control Heatmap
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, max(6, len(domain_weights_df) * 0.55)), dpi=200)

sns.heatmap(
    domain_weights_df,
    annot=True,
    fmt=".2f",
    cmap=sns.light_palette("#5BC2E7", as_cmap=True),
    linewidths=0.4,
    linecolor='white',
    annot_kws={"size": 9},
    ax=ax,
)
ax.set_title(
    "Data-Driven Governance Control Weights ($w_c$) by Domain\n"
    "Source: AIID Incidents labelled with Qwen2.5-7B-Instruct",
    fontsize=12,
    fontweight='bold',
    pad=14,
)
ax.set_xlabel("Governance Control", labelpad=10)
ax.set_ylabel("AI Application Domain", labelpad=10)

# Rename x-axis ticks to include full name
ax.set_xticklabels(
    [f"{c}\n{CONTROL_NAMES[c]}" for c in CONTROLS],
    rotation=0,
    fontsize=8,
)
plt.tight_layout()
plt.savefig("figures/governance_weights_heatmap.png", dpi=200, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 8. Per-incident G_ops (Governance Deficiency Score)
# ─────────────────────────────────────────────────────────────────────────────
def calculate_incident_g_ops(row: pd.Series) -> float:
    """
    G_ops_i = Σ_c  w_c(domain) × A_{i,c}

    A higher score indicates more governance controls were absent —
    i.e., a higher deficiency level (poorer governance).
    The score is used as a proxy rating factor in the premium model.
    """
    weights = domain_weights_df.get(row['domain'], global_weights.to_dict())
    return sum(row[c] * weights.get(c, 0) for c in CONTROLS)


arc_df['real_G_ops'] = arc_df.apply(calculate_incident_g_ops, axis=1)

print("\n[4] Per-incident G_ops computed — sample (top 5):")
print(arc_df[['id', 'domain', 'title', 'real_G_ops']].head().to_string(index=False))

# Distribution plot
fig, ax = plt.subplots(figsize=(11, 5), dpi=200)
palette = dict(zip(arc_df['domain'].unique(),
                   ["#5BC2E7", "#003CDC", "#FED63F", "#B4CF3C",
                    "#3AAD67", "#F39945", "#F2A4BC", "#F96D69"]))
sns.histplot(
    data=arc_df,
    x='real_G_ops',
    hue='domain',
    multiple="stack",
    bins=20,
    palette=palette,
    ax=ax,
)
ax.set_title(
    "Distribution of Governance Deficiency Score ($G_{ops}$) by Domain\n"
    "Higher score = more governance controls absent = poorer governance",
    fontsize=11,
    fontweight='bold',
    pad=14,
)
ax.set_xlabel(r"Governance Deficiency Score ($G_{ops}$)", labelpad=10)
ax.set_ylabel("Number of Incidents", labelpad=10)
ax.grid(True, linestyle='--', alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("figures/governance_deficiency_distribution.png",
            dpi=200, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 9. Export Results
# ─────────────────────────────────────────────────────────────────────────────
all_weights_df = pd.concat([
    domain_weights_df,
    pd.DataFrame([global_weights], index=['General']),
])
all_weights_df.to_csv("aiid_domain_and_general_weights.csv")
arc_df.to_csv("aiid_incident_scores.csv", index=False)

print("\n[Done] Exported:")
print("  aiid_domain_and_general_weights.csv  — domain weights for simulation")
print("  aiid_incident_scores.csv             — per-incident G_ops scores")