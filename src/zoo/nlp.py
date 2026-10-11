"""Natural Language Processing model adapters: DistilBERT, BERT-base."""

import logging
from typing import Any, Dict
from src.zoo.base import BaseTorchAdapter

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class DistilBERTAdapter(BaseTorchAdapter):
    """Adapter for DistilBERT-base-uncased."""

    name: str = "distilbert"
    family: str = "nlp"

    def load(self, device: str, precision: str = "fp32") -> Any:
        dev = "cuda" if "cuda" in device.lower() or "t4" in device.lower() else "cpu"
        try:
            from transformers import DistilBertConfig, DistilBertModel
            config = DistilBertConfig()
            model = DistilBertModel(config)
        except Exception:
            model = nn.Sequential(
                nn.Embedding(1000, 128),
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True),
                    num_layers=2,
                ),
            )
        model = model.to(dev)
        if precision == "fp16" and dev == "cuda":
            model = model.half()
        model.eval()
        return model

    def make_input(self, batch_size: int, device: str = "cpu", precision: str = "fp32", seq_len: int = 128, **kwargs: Any) -> Any:
        dev = "cuda" if "cuda" in device.lower() or "t4" in device.lower() else "cpu"
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=dev, dtype=torch.long)
        if kwargs.get("include_mask", False):
            attention_mask = torch.ones(batch_size, seq_len, device=dev, dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        return {"input_ids": input_ids}


class BERTBaseAdapter(BaseTorchAdapter):
    """Adapter for BERT-base-uncased."""

    name: str = "bert_base"
    family: str = "nlp"

    def load(self, device: str, precision: str = "fp32") -> Any:
        dev = "cuda" if "cuda" in device.lower() or "t4" in device.lower() else "cpu"
        try:
            from transformers import BertConfig, BertModel
            config = BertConfig()
            model = BertModel(config)
        except Exception:
            model = nn.Sequential(
                nn.Embedding(1000, 256),
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True),
                    num_layers=4,
                ),
            )
        model = model.to(dev)
        if precision == "fp16" and dev == "cuda":
            model = model.half()
        model.eval()
        return model

    def make_input(self, batch_size: int, device: str = "cpu", precision: str = "fp32", seq_len: int = 128, **kwargs: Any) -> Any:
        dev = "cuda" if "cuda" in device.lower() or "t4" in device.lower() else "cpu"
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=dev, dtype=torch.long)
        if kwargs.get("include_mask", False):
            attention_mask = torch.ones(batch_size, seq_len, device=dev, dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        return {"input_ids": input_ids}

    def preprocess(self, raw_data: Any) -> Any:
        """Tokenize raw text inputs into input_ids tensor."""
        if isinstance(raw_data, str):
            raw_data = [raw_data]
        if isinstance(raw_data, (list, tuple)) and all(isinstance(s, str) for s in raw_data):
            try:
                from transformers import BertTokenizerFast
                tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", local_files_only=True)
                enc = tokenizer(list(raw_data), padding=True, truncation=True, return_tensors="pt")
                return {"input_ids": enc["input_ids"]}
            except Exception:
                tokens = [[abs(hash(w)) % 1000 + 1 for w in text.split()] or [1] for text in raw_data]
                max_len = max(len(t) for t in tokens)
                padded = [t + [0] * (max_len - len(t)) for t in tokens]
                return {"input_ids": torch.tensor(padded, dtype=torch.long)}
        return raw_data

    def postprocess(self, model_output: Any) -> Any:
        """Extract pooled representations or token logits."""
        if hasattr(model_output, "last_hidden_state"):
            return model_output.last_hidden_state.mean(dim=1)
        if hasattr(model_output, "logits"):
            return model_output.logits.argmax(dim=-1)
        if isinstance(model_output, (tuple, list)) and len(model_output) > 0:
            return model_output[0]
        return model_output
