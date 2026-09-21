"""
Data loading and 5-fold grouped cross-validation partitioning for ArchEHR-QA.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None
    _HAS_NUMPY = False

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    pd = None
    _HAS_PANDAS = False

try:
    from sklearn.model_selection import GroupKFold
    _HAS_SKLEARN = True
except ImportError:
    GroupKFold = None
    _HAS_SKLEARN = False

import xml.etree.ElementTree as ET

def clean_text(text: Optional[str]) -> str:
    """Normalize whitespace and strip leading/trailing spaces."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()

@dataclass
class Case:
    case_id: str
    clinical_specialty: str
    patient_narrative: str
    patient_question: str
    clinician_question: str
    sentences: List[Dict[str, str]] = field(default_factory=list)
    clinician_answer: str = ""
    labels: Optional[Dict[str, str]] = None  # {sentence_id: "essential"|"supplementary"|"not-relevant"}

    @property
    def essential_sentence_ids(self) -> List[str]:
        if not self.labels:
            return []
        return [sid for sid, rel in self.labels.items() if rel == "essential"]

    @property
    def lenient_sentence_ids(self) -> List[str]:
        if not self.labels:
            return []
        return [sid for sid, rel in self.labels.items() if rel in {"essential", "supplementary"}]

    @property
    def sentence_map(self) -> Dict[str, str]:
        return {s["id"]: s["text"] for s in self.sentences}


def parse_cases_from_xml(split_dir: Any, with_key: bool = True) -> List[Case]:
    """
    Parse cases from archehr-qa.xml and optional archehr-qa_key.json.
    Works with standard library xml.etree.ElementTree. Accepts str or Path.
    """
    split_dir = Path(split_dir)
    xml_path = split_dir / "archehr-qa.xml"
    key_path = split_dir / "archehr-qa_key.json"

    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found at {xml_path}")

    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    key_dict: Dict[str, Dict[str, Any]] = {}
    if with_key and key_path.exists():
        with open(key_path, "r", encoding="utf-8") as f:
            raw_key = json.load(f)
            key_dict = {str(item["case_id"]): item for item in raw_key}

    cases: List[Case] = []
    for case_node in root.findall(".//case"):
        cid = case_node.get("id") or ""
        specialty = clean_text(case_node.findtext("clinical_specialty"))
        narrative = clean_text(case_node.findtext("patient_narrative"))

        # Patient question
        pat_q_node = case_node.find("patient_question")
        if pat_q_node is not None:
            phrases = [clean_text(p.text) for p in pat_q_node.findall("phrase") if p.text]
            pat_q = " ".join(phrases) if phrases else clean_text(pat_q_node.text)
        else:
            pat_q = ""

        clin_q = clean_text(case_node.findtext("clinician_question"))

        # Sentences
        sentences: List[Dict[str, str]] = []
        for s_node in case_node.findall("./note_excerpt_sentences/sentence"):
            sid = s_node.get("id") or ""
            stext = clean_text(s_node.text)
            sentences.append({"id": sid, "text": stext})

        # Key annotations
        c_ans = ""
        labels = None
        if cid in key_dict:
            c_ans = clean_text(key_dict[cid].get("clinician_answer", ""))
            if "answers" in key_dict[cid]:
                labels = {
                    str(ans["sentence_id"]): ans["relevance"]
                    for ans in key_dict[cid]["answers"]
                }

        cases.append(Case(
            case_id=cid,
            clinical_specialty=specialty,
            patient_narrative=narrative,
            patient_question=pat_q,
            clinician_question=clin_q,
            sentences=sentences,
            clinician_answer=c_ans,
            labels=labels
        ))

    return cases


def get_5fold_cv_splits(cases: List[Case]) -> List[Tuple[List[Case], List[Case]]]:
    """
    Partition development cases into 5 non-overlapping folds grouped by case_id.
    Guarantees that each case appears in the validation split exactly once.
    Uses sklearn GroupKFold when available; falls back to pure python grouping otherwise.
    """
    if _HAS_SKLEARN and _HAS_NUMPY:
        case_ids = np.array([c.case_id for c in cases])
        gkf = GroupKFold(n_splits=5)
        splits = []
        X = np.arange(len(cases))
        for train_idx, val_idx in gkf.split(X, groups=case_ids):
            train_cases = [cases[i] for i in train_idx]
            val_cases = [cases[i] for i in val_idx]
            splits.append((train_cases, val_cases))
        return splits

    # Pure Python deterministic GroupKFold fallback
    unique_case_ids = sorted(list({c.case_id for c in cases}))
    n_splits = 5
    splits = []
    # Distribute unique case IDs into 5 folds evenly
    val_groups = [[] for _ in range(n_splits)]
    for i, cid in enumerate(unique_case_ids):
        val_groups[i % n_splits].append(cid)

    for fold_val_ids in val_groups:
        val_set = set(fold_val_ids)
        train_cases = [c for c in cases if c.case_id not in val_set]
        val_cases = [c for c in cases if c.case_id in val_set]
        splits.append((train_cases, val_cases))

    return splits


def cases_to_sentence_dataframe(cases: List[Case]) -> Any:
    """
    Convert cases to a DataFrame (or list of dicts if pandas unavailable)
    with graded relevance labels:
      2 = essential
      1 = supplementary
      0 = not-relevant / other
    """
    rows = []
    relevance_map = {
        "essential": 2,
        "supplementary": 1,
        "not-relevant": 0
    }
    for c in cases:
        if not c.labels:
            continue
        for s in c.sentences:
            label_str = c.labels.get(s["id"], "not-relevant")
            graded_label = relevance_map.get(label_str, 0)
            rows.append({
                "case_id": c.case_id,
                "clinician_question": c.clinician_question,
                "patient_question": c.patient_question,
                "narrative": c.patient_narrative,
                "sentence_id": s["id"],
                "sentence_text": s["text"],
                "label_str": label_str,
                "label_graded": graded_label,
                "is_essential": int(graded_label == 2),
                "is_lenient": int(graded_label >= 1)
            })
    if _HAS_PANDAS:
        return pd.DataFrame(rows)
    return rows
