"""
Configuration parameters, model identifiers, and hyperparameters for ArchEHR-QA SC-Cal.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False

@dataclass
class PipelineConfig:
    # Random seed
    seed: int = 42

    # Data paths (auto-detects Kaggle, Colab, or local workspace)
    data_root: Path = field(default_factory=lambda: Path(
        os.environ.get("ARCHEHR_DATA_DIR", "")
    ))
    output_dir: Path = field(default_factory=lambda: Path("./outputs"))

    # Context & Generation constraints
    k_context: int = 10                  # Number of top reranked sentences to retain
    r_rollouts: int = 10                 # Number of stochastic rollouts for SC-Cal
    temperature: float = 0.70            # Sampling temperature for rollouts
    top_p: float = 0.90                  # Nucleus sampling probability
    max_answer_words: int = 75           # Strict official word ceiling
    min_answer_words: int = 40           # Minimum word floor for tuning objective

    # Verification & Attribution thresholds
    theta_ungrounded: float = 0.30       # NLI threshold below which a claim is pruned
    theta_cite: float = 0.50             # NLI threshold for retaining a citation

    # Cross-encoder fine-tuning
    ranker_epochs: int = 5
    ranker_batch_size: int = 16
    ranker_lr: float = 2e-5
    ranker_weight_decay: float = 0.01
    ranker_patience: int = 2

    # Target SLMs (<= 4B)
    slm_models: Dict[str, str] = field(default_factory=lambda: {
        "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
        "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
        "phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct",
    })

    # Target LLMs (7B - 8B)
    llm_models: Dict[str, str] = field(default_factory=lambda: {
        "mistral-7b-dpo": "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
        "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
        "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    })

    # Sentence Embedding Model for SC-Cal Consensus
    embedding_model: str = "pritamdeka/S-BioBert-snli-multinli-stsb"
    embedding_fallback: str = "BAAI/bge-small-en-v1.5"

    # Cross-Encoder Ranker Base
    ranker_base_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ranker_biomedical_base: str = "ncbi/MedCPT-Cross-Encoder"

    # Decoupled NLI Models
    pruner_nli_model: str = "cross-encoder/nli-deberta-v3-small"
    labeler_nli_model: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

    # Hardware & Runtime
    device: str = "cuda" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu"
    # Note: Use float16 on Kaggle T4 GPUs as T4 lacks native bfloat16 hardware acceleration
    compute_dtype: Any = field(default_factory=lambda: (torch.float16 if (_HAS_TORCH and torch.cuda.is_available()) else (torch.float32 if _HAS_TORCH else "float16")))

    def __post_init__(self):
        base_dir = Path(__file__).resolve().parent.parent
        if not (self.data_root / "dev" / "archehr-qa.xml").exists():
            # Check Kaggle input recursively first
            kaggle_input = Path("/kaggle/input")
            found = False
            if kaggle_input.exists():
                for xml_file in kaggle_input.glob("**/dev/archehr-qa.xml"):
                    self.data_root = xml_file.parent.parent.resolve()
                    found = True
                    break
            if not found:
                for base in [base_dir, Path("."), Path("..")]:
                    for xml_file in base.glob("**/dev/archehr-qa.xml"):
                        self.data_root = xml_file.parent.parent.resolve()
                        found = True
                        break
        self.output_dir = (base_dir / "outputs") if not self.output_dir.is_absolute() else self.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
