"""Phone-recognizer stage: wav2vec2 CTC (espeak IPA) -> Switchboard 42-phone sequence with times."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phones.ipa import IpaMap

MODEL = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
STRIDE = 0.02           # wav2vec2 frame shift (s)


@dataclass
class RecPhone:
    ph: str             # Switchboard phone
    start: float
    end: float
    p: float            # mean CTC posterior of the frames that emitted it
    tok: str = ""       # the IPA token it came from

    def to_json(self) -> dict:
        return {"ph": self.ph, "start": round(self.start, 3), "end": round(self.end, 3), "p": round(self.p, 3), "tok": self.tok}


def collapse(ids: np.ndarray, probs: np.ndarray, vocab: list[str], t_off: float) -> list[dict]:
    """Greedy CTC: merge repeats, drop blanks (<pad>=0). Times from the frames of each run."""
    out, i, n = [], 0, len(ids)
    while i < n:
        j = i
        while j + 1 < n and ids[j + 1] == ids[i]:
            j += 1
        if ids[i] != 0:
            out.append({"tok": vocab[ids[i]], "start": round(t_off + i * STRIDE, 3),
                        "end": round(t_off + (j + 1) * STRIDE, 3), "p": round(float(probs[i:j + 1].mean()), 3)})
        i = j + 1
    return out


def to_swb(tokens: list[dict], ipa: IpaMap) -> list[RecPhone]:
    """Map IPA tokens to Switchboard phones; a token mapping to k phones shares its span evenly."""
    out: list[RecPhone] = []
    for t in tokens:
        phs = ipa.lookup(t["tok"])
        if not phs:
            continue
        step = (t["end"] - t["start"]) / len(phs)
        for k, ph in enumerate(phs):
            out.append(RecPhone(ph, t["start"] + k * step, t["start"] + (k + 1) * step, t["p"], t["tok"]))
    return out


class PhoneRecognizer:
    def __init__(self, ipa: IpaMap, model: str = MODEL, device: str | None = None) -> None:
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

        self.torch, self.ipa = torch, ipa
        self.dev = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(model)
        self.model = Wav2Vec2ForCTC.from_pretrained(model).to(self.dev).eval()
        vmap = json.loads(Path(hf_hub_download(model, "vocab.json")).read_text())
        self.vocab: list[str] = [""] * (max(vmap.values()) + 1)
        for tok, i in vmap.items():
            self.vocab[i] = tok

    def recognize(self, audio: np.ndarray, offset: float = 0.0) -> list[RecPhone]:
        torch = self.torch
        with torch.no_grad():
            inp = self.fe(audio, sampling_rate=16000, return_tensors="pt")
            probs = self.model(inp.input_values.to(self.dev)).logits[0].float().softmax(-1)
            ids = probs.argmax(-1).cpu().numpy()
            pmax = probs.max(-1).values.cpu().numpy()
        return to_swb(collapse(ids, pmax, self.vocab, offset), self.ipa)
