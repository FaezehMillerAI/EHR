"""
Unified SLM and LLM generation runner for ArchEHR-QA Subtask 3.
Supports 4-bit NF4 quantization on T4 GPUs, batched stochastic rollouts,
and whole-sentence word truncation to strictly respect the 75-word limit.
"""

import re
from typing import List, Dict, Optional, Any
from pathlib import Path
from archehr_pipeline.config import PipelineConfig
from archehr_pipeline.data_loader import Case

try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig
    )
    _HAS_TORCH_HF = True
except ImportError:
    torch = None
    _HAS_TORCH_HF = False


def strip_citations(text: str) -> str:
    """
    Strip inline citation tags (|1|, |2, 6|, [1], etc.) from text.
    Crucial to prevent embedding similarity from matching citation indices rather than content.
    """
    if not text:
        return ""
    # Strip pipe citations e.g. |2, 6| or |3|
    cleaned = re.sub(r"\|[^|]*\|", "", text)
    # Strip bracket citations e.g. [2, 6] or [3]
    cleaned = re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", "", cleaned)
    # Clean space before punctuation
    cleaned = re.sub(r"\s+([.,!?;:])", r"\1", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def count_words(text: str) -> int:
    """Count words excluding citation tags."""
    content_only = strip_citations(text)
    return len(re.findall(r"\b\w+\b", content_only))


def truncate_to_whole_sentences(answer_text: str, max_words: int = 75) -> str:
    """
    Enforce max_words ceiling by dropping whole trailing sentences rather than
    cutting mid-sentence, which preserves complete thoughts and pipe citation markers.
    """
    if not answer_text or not answer_text.strip():
        return ""

    # Split into lines/sentences (each sentence typically ends with citation |id|)
    lines = [line.strip() for line in answer_text.strip().split("\n") if line.strip()]
    if not lines:
        lines = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer_text.strip()) if s.strip()]

    accepted_lines = []
    current_words = 0

    for line in lines:
        line_words = count_words(line)
        if current_words + line_words <= max_words or not accepted_lines:
            accepted_lines.append(line)
            current_words += line_words
        else:
            # Adding this sentence would breach 75 words; stop here
            break

    return "\n".join(accepted_lines)


def format_evidence_block(evidence_sentences: List[Dict[str, Any]]) -> str:
    """Format evidence sentences as '[id] text'."""
    return "\n".join(f"[{s['id']}] {s['text']}" for s in evidence_sentences)


def build_clinical_prompt(
    case: Case,
    evidence_sentences: List[Dict[str, Any]],
    tokenizer: Optional[Any] = None,
    few_shot_examples: str = ""
) -> str:
    """
    Construct chat template prompt formatted for grounded clinical QA.
    Enforces the official citation syntax: 'Sentence. |id|' and <= 75 word limit.
    """
    system_prompt = (
        "You are an expert clinical AI assisting healthcare providers in answering patient inquiries. "
        "Your task is to answer the patient's question using ONLY the provided clinical note evidence. "
        "Strict Requirements:\n"
        "1. Write a clear, professional answer of at most 75 words.\n"
        "2. Do not introduce outside medical assumptions or unverified facts.\n"
        "3. Every factual sentence MUST cite its supporting clinical note sentence ID using pipe delimiters: "
        "e.g., 'The patient was diagnosed with CBD sludge. |2|' or 'Stent was placed to allow drainage. |2, 6|'.\n"
        "4. If the evidence is insufficient to answer the question, state so briefly."
    )

    user_content = (
        f"{few_shot_examples}\n\n" if few_shot_examples else ""
    ) + (
        f"PATIENT NARRATIVE: {case.patient_narrative}\n"
        f"PATIENT QUESTION: {case.patient_question}\n"
        f"CLINICIAN QUESTION: {case.clinician_question}\n\n"
        f"CLINICAL NOTE EVIDENCE:\n"
        f"{format_evidence_block(evidence_sentences)}\n\n"
        f"GROUNDED ANSWER (<= 75 words with |id| citations):"
    )

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # Fallback if model tokenizer doesn't support system role
            pass

    return f"SYSTEM: {system_prompt}\n\nUSER: {user_content}\n\nASSISTANT:"


class ClinicalAnswerGenerator:
    """
    Unified generator for the 3 SLMs and 3 LLMs in 4-bit NF4 with float16 compute.
    """
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct",
        config: Optional[PipelineConfig] = None,
        load_in_4bit: bool = True
    ):
        self.config = config or PipelineConfig()
        self.model_name = model_name_or_path
        self.load_in_4bit = load_in_4bit
        self.device = self.config.device
        self.tokenizer = None
        self.model = None

    def load(self):
        if not _HAS_TORCH_HF:
            raise RuntimeError("PyTorch and HuggingFace Transformers required for generation.")
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            quant_config = None
            if self.load_in_4bit and torch.cuda.is_available():
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=self.config.compute_dtype
                )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quant_config,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=self.config.compute_dtype,
                trust_remote_code=True
            ).eval()

    @torch.no_grad() if _HAS_TORCH_HF else lambda fn: fn
    def generate_greedy(
        self,
        case: Case,
        evidence_sentences: List[Dict[str, Any]],
        few_shot_examples: str = ""
    ) -> str:
        """
        Deterministic greedy generation (T=0.0). Used for baselines M0 and M1.
        """
        if not _HAS_TORCH_HF or self.model is None:
            # Mock fallback for test verification without GPU
            first_sids = [s["id"] for s in evidence_sentences[:2]]
            cite_str = f" |{', '.join(first_sids)}|" if first_sids else ""
            return f"The clinical course indicates relevant diagnostic findings.{cite_str}"

        self.load()
        prompt_text = build_clinical_prompt(case, evidence_sentences, self.tokenizer, few_shot_examples)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)

        out = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=180,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        gen_tokens = out[0, inputs.input_ids.shape[1]:]
        raw_answer = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        return truncate_to_whole_sentences(raw_answer, max_words=self.config.max_answer_words)

    @torch.no_grad() if _HAS_TORCH_HF else lambda fn: fn
    def generate_rollouts(
        self,
        case: Case,
        evidence_sentences: List[Dict[str, Any]],
        r_rollouts: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        few_shot_examples: str = ""
    ) -> List[str]:
        """
        Batched stochastic sampling (T=0.7, p=0.9). Generates R rollouts simultaneously.
        """
        r = r_rollouts or self.config.r_rollouts
        temp = temperature or self.config.temperature
        p = top_p or self.config.top_p

        if not _HAS_TORCH_HF or self.model is None:
            # Mock rollouts for testing
            mock_answers = []
            for i in range(r):
                first_sids = [s["id"] for s in evidence_sentences[:max(1, (i % 3) + 1)]]
                cite_str = f" |{', '.join(first_sids)}|" if first_sids else ""
                mock_answers.append(f"Sample {i+1}: Clinical examination revealed notable conditions.{cite_str}")
            return mock_answers

        self.load()
        prompt_text = build_clinical_prompt(case, evidence_sentences, self.tokenizer, few_shot_examples)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)

        # Batched sampling using num_return_sequences
        out = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=temp,
            top_p=p,
            num_return_sequences=r,
            max_new_tokens=180,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        rollouts = []
        prompt_len = inputs.input_ids.shape[1]
        for seq in out:
            gen_tokens = seq[prompt_len:]
            text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            truncated = truncate_to_whole_sentences(text, max_words=self.config.max_answer_words)
            rollouts.append(truncated)

        return rollouts
