"""ArchEHR-QA: paper-faithful two-stage RAG with a small language model.

Run this file from the accompanying Kaggle notebook.  Attach the extracted
ArchEHR-QA dataset as a Kaggle dataset and change DATA_ROOT below if needed.
"""
# %% [markdown]
# # ArchEHR-QA - BioBERT RAG + Qwen2.5-3B
#
# Reimplementation of Kadusabe et al. (BioNLP 2025): BioBERT bi-encoder
# retrieval -> fine-tuned BioBERT cross-encoder reranking -> grounded answer
# generation.  Qwen2.5-3B-Instruct in 4-bit is used as the requested SLM.
# The `evidence_trace` is an auditable alternative to exposing hidden chain of
# thought: it splits the final answer into claims and attaches reranked EHR
# sentences to every claim.  `confidence` is answer self-consistency, not a
# calibrated probability or clinical certainty.

# %%
!pip -q install -U "transformers>=4.46" "accelerate>=0.34" bitsandbytes sentence-transformers datasets scikit-learn lxml

# %%
import gc, json, os, random, re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from lxml import etree
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_recall_fscore_support
from sentence_transformers import SentenceTransformer
from transformers import (AutoModelForCausalLM, AutoModelForSequenceClassification,
                          AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup)
from torch.utils.data import DataLoader, Dataset

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cuda.matmul.allow_tf32 = True

# Attach the supplied directory in Kaggle and point this to it.
DATA_ROOT = Path("/kaggle/input/archehr-qa-a-dataset-for-addressing-patients-information-needs-related-to-clinical-course-of-hospitalization-1-3")
if not DATA_ROOT.exists():
    DATA_ROOT = Path("/kaggle/input/archehr-qa")  # convenient alternate slug
OUT = Path("/kaggle/working/archehr_sml_rag"); OUT.mkdir(parents=True, exist_ok=True)

# The paper reports (K,N)=(13,30), which contradicts its own definition:
# reranking cannot select 30 sentences from 13 candidates.  We retain these
# two reported counts in the valid order: retrieve K=30, then retain N=13.
RETRIEVE_K, EVIDENCE_N = 30, 13
MAX_LENGTH, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, PATIENCE = 512, 8, 10, 2e-5, .01, 2
CROSS_MODEL = "dmis-lab/biobert-base-cased-v1.2"
BI_MODEL = "pritamdeka/S-BioBert-snli-multinli-stsb"
GEN_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert DATA_ROOT.exists(), f"Dataset not found: {DATA_ROOT}. Attach it and update DATA_ROOT."
print("GPU(s):", torch.cuda.device_count(), "| data:", DATA_ROOT)

# %%
@dataclass
class Case:
    case_id: str
    specialty: str
    narrative: str
    patient_question: str
    clinician_question: str
    sentences: list[dict]
    answer: str = ""
    labels: dict | None = None

def clean(x): return re.sub(r"\s+", " ", x or "").strip()

def parse_cases(split: str, with_key: bool) -> list[Case]:
    root = etree.parse(str(DATA_ROOT / split / "archehr-qa.xml")).getroot()
    key = {}
    if with_key:
        key = {str(x["case_id"]): x for x in json.loads((DATA_ROOT / split / "archehr-qa_key.json").read_text())}
    cases = []
    for node in root.findall("case"):
        cid = node.get("id")
        sentence_nodes = node.findall("./note_excerpt_sentences/sentence")
        sentences = [{"id": s.get("id"), "text": clean(s.text)} for s in sentence_nodes]
        ann = key.get(cid, {})
        labels = {str(x["sentence_id"]): x["relevance"] for x in ann.get("answers", [])} or None
        cases.append(Case(cid, clean(node.findtext("clinical_specialty")), clean(node.findtext("patient_narrative")),
                          clean(node.findtext("patient_question")), clean(node.findtext("clinician_question")),
                          sentences, clean(ann.get("clinician_answer", "")), labels))
    return cases

dev_cases, test_cases = parse_cases("dev", True), parse_cases("test", False)
assert len(dev_cases) == 20 and all(c.labels for c in dev_cases)
print(f"Loaded {len(dev_cases)} labelled development and {len(test_cases)} test cases")

def pairs_from(cases):
    rows = []
    for c in cases:
        for s in c.sentences:
            # Paper formulation: essential=1, supplementary/not-relevant=0.
            rows.append({"case_id": c.case_id, "question": c.patient_question,
                         "sentence": s["text"], "label": int(c.labels[s["id"]] == "essential")})
    return pd.DataFrame(rows)

pairs = pairs_from(dev_cases)
splitter = GroupShuffleSplit(n_splits=1, test_size=.20, random_state=SEED)
train_i, val_i = next(splitter.split(pairs, groups=pairs.case_id))
train_df, val_df = pairs.iloc[train_i].reset_index(drop=True), pairs.iloc[val_i].reset_index(drop=True)
print(train_df.label.value_counts().to_dict(), "validation cases:", val_df.case_id.unique())

# %%
class PairDataset(Dataset):
    def __init__(self, frame, tokenizer):
        self.frame, self.tok = frame.reset_index(drop=True), tokenizer
    def __len__(self): return len(self.frame)
    def __getitem__(self, i):
        r = self.frame.iloc[i]
        x = self.tok(r.question, r.sentence, truncation=True, max_length=MAX_LENGTH,
                     padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in x.items()} | {"labels": torch.tensor(float(r.label))}

cross_tok = AutoTokenizer.from_pretrained(CROSS_MODEL)
cross = AutoModelForSequenceClassification.from_pretrained(CROSS_MODEL, num_labels=1).to(DEVICE)
train_loader = DataLoader(PairDataset(train_df, cross_tok), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(PairDataset(val_df, cross_tok), batch_size=BATCH_SIZE, pin_memory=True)
optim = torch.optim.AdamW(cross.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = get_linear_schedule_with_warmup(optim, 0, EPOCHS * len(train_loader))
loss_fn = torch.nn.BCEWithLogitsLoss()  # exact binary cross-entropy objective in the paper

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); ys, ps = [], []
    for b in loader:
        y = b.pop("labels").numpy(); b = {k:v.to(DEVICE) for k,v in b.items()}
        p = (torch.sigmoid(model(**b).logits.squeeze(-1)) >= .5).cpu().numpy()
        ys.extend(y); ps.extend(p)
    return precision_recall_fscore_support(ys, ps, average="binary", zero_division=0)

best_f1, stale = -1, 0
for epoch in range(EPOCHS):
    cross.train()
    for b in train_loader:
        y = b.pop("labels").to(DEVICE); b = {k:v.to(DEVICE) for k,v in b.items()}
        loss = loss_fn(cross(**b).logits.squeeze(-1), y)
        optim.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(cross.parameters(), 1.0)
        optim.step(); sched.step()
    p, r, f1, _ = evaluate(cross, val_loader)
    print(f"epoch={epoch+1}: P={p:.3f}, R={r:.3f}, F1={f1:.3f}")
    if f1 > best_f1:
        best_f1, stale = f1, 0; cross.save_pretrained(OUT / "cross_encoder"); cross_tok.save_pretrained(OUT / "cross_encoder")
    else:
        stale += 1
        if stale >= PATIENCE: print("Early stop"); break
cross = AutoModelForSequenceClassification.from_pretrained(OUT / "cross_encoder").to(DEVICE).eval()

# %%
bi = SentenceTransformer(BI_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def reranker_scores(question: str, sentences: list[dict], batch=16):
    scores = []
    for start in range(0, len(sentences), batch):
        chunk = sentences[start:start+batch]
        x = cross_tok([question]*len(chunk), [s["text"] for s in chunk], padding=True,
                      truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(DEVICE)
        scores.extend(torch.sigmoid(cross(**x).logits.squeeze(-1)).cpu().tolist())
    return scores

def retrieve(question: str, sentences: list[dict]):
    """Paper's dense BioBERT retrieval followed by cross-encoder reranking."""
    if not sentences: return []
    texts = [s["text"] for s in sentences]
    emb = bi.encode([question] + texts, normalize_embeddings=True, convert_to_numpy=True,
                    show_progress_bar=False)
    candidate_ix = np.argsort(emb[1:] @ emb[0])[::-1][:min(RETRIEVE_K, len(sentences))]
    candidates = [dict(sentences[i]) for i in candidate_ix]
    for s, score in zip(candidates, reranker_scores(question, candidates)):
        s["relevance_score"] = float(score)
    return sorted(candidates, key=lambda x: x["relevance_score"], reverse=True)[:min(EVIDENCE_N, len(candidates))]

def evidence_block(evidence):
    return "\n".join(f"[{s['id']}] {s['text']}" for s in evidence)

def few_shots(cases, n=2):
    shots = []
    for c in cases[:n]:
        gold = [s for s in c.sentences if c.labels[s["id"]] == "essential"]
        shots.append(f"QUESTION: {c.patient_question}\nEVIDENCE:\n{evidence_block(gold)}\nANSWER: {c.answer}")
    return "\n\n".join(shots)

train_case_ids = set(train_df.case_id)
SHOT_TEXT = few_shots([c for c in dev_cases if c.case_id in train_case_ids])

# %%
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                           bnb_4bit_compute_dtype=torch.float16)
gen_tok = AutoTokenizer.from_pretrained(GEN_MODEL); gen_tok.pad_token = gen_tok.eos_token
gen = AutoModelForCausalLM.from_pretrained(GEN_MODEL, quantization_config=quant, device_map="auto",
                                           torch_dtype=torch.float16).eval()

def prompt_for(c: Case, evidence):
    system = ("You answer patient questions using only supplied clinical-note evidence. "
              "Do not add medical facts. Write 65-75 words. Every factual sentence must cite one or more "
              "supporting source IDs as [id]. If evidence is insufficient, say so briefly.")
    user = f"Examples:\n{SHOT_TEXT}\n\nPATIENT NARRATIVE: {c.narrative}\nPATIENT QUESTION: {c.patient_question}\nCLINICIAN QUESTION: {c.clinician_question}\nEVIDENCE:\n{evidence_block(evidence)}\nANSWER:"
    return gen_tok.apply_chat_template([{"role":"system","content":system}, {"role":"user","content":user}],
                                       tokenize=False, add_generation_prompt=True)

def word_count(answer): return len(re.findall(r"\b\w+\b", re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", "", answer)))

@torch.no_grad()
def sample_answer(c, evidence, seed):
    torch.manual_seed(seed)
    x = gen_tok(prompt_for(c, evidence), return_tensors="pt").to(gen.device)
    out = gen.generate(**x, do_sample=True, temperature=.70, top_p=.90, max_new_tokens=200,
                       pad_token_id=gen_tok.eos_token_id)
    return gen_tok.decode(out[0, x.input_ids.shape[1]:], skip_special_tokens=True).strip()

def generate_candidates(c, evidence, n=5, retries=10):
    answers = []
    for seed in range(SEED, SEED + n):
        candidate = ""
        for retry in range(retries):  # paper's 65-75 word retry rule
            candidate = sample_answer(c, evidence, seed + retry*1000)
            if 65 <= word_count(candidate) <= 75: break
        answers.append(candidate)
    return answers

def self_consistency(candidates):
    if len(candidates) < 2: return 1.0
    v = bi.encode(candidates, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    sim = v @ v.T
    return float(np.clip((sim.sum() - len(v)) / (len(v)*(len(v)-1)), 0, 1)), sim

def claim_trace(answer, evidence, top_k=2):
    """Claim -> important note sentences -> combined evidence trace (no hidden CoT)."""
    claims = [clean(x) for x in re.split(r"(?<=[.!?])\s+", answer) if clean(x)]
    trace = []
    for claim in claims:
        scored = [dict(s) for s in evidence]
        for s, score in zip(scored, reranker_scores(claim, scored)): s["claim_score"] = float(score)
        supporting = sorted(scored, key=lambda x:x["claim_score"], reverse=True)[:top_k]
        trace.append({"claim": claim, "supporting_sentence_ids": [s["id"] for s in supporting],
                      "supporting_sentences": [s["text"] for s in supporting]})
    return trace

def answer_case(c):
    evidence = retrieve(c.patient_question, c.sentences)
    candidates = generate_candidates(c, evidence)
    confidence, similarity_matrix = self_consistency(candidates)
    # Medoid: the candidate that agrees most with the other independently sampled answers.
    best = int(np.argmax(similarity_matrix.mean(axis=1)))
    answer = candidates[best]
    return {"case_id": c.case_id, "answer": answer, "confidence_self_consistency": confidence,
            "candidate_answers": candidates, "evidence": evidence, "evidence_trace": claim_trace(answer, evidence)}

# %%
# Run one case first as a smoke test; set RUN_TEST=True for the official test predictions.
demo = answer_case(dev_cases[0])
display(pd.DataFrame([{k: demo[k] for k in ["case_id", "answer", "confidence_self_consistency"]}]))
display(pd.DataFrame(demo["evidence_trace"]))

RUN_TEST = False  # change to True after inspecting the demo
if RUN_TEST:
    results = [answer_case(c) for c in test_cases]
    pd.DataFrame([{"case_id": r["case_id"], "answer": r["answer"],
                   "confidence_self_consistency": r["confidence_self_consistency"],
                   "evidence_trace": json.dumps(r["evidence_trace"])} for r in results]).to_csv(OUT / "test_predictions.csv", index=False)
    (OUT / "test_predictions_full.json").write_text(json.dumps(results, indent=2))
    print("Saved:", OUT / "test_predictions.csv")
