"""Model loading. Everything loads from TEAM_ID/weights/ (no network)."""
import os

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from . import label as L

_HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.environ.get("PRISM_WEIGHTS", os.path.join(_HERE, "..", "..", "weights"))


class ModelsUnavailable(RuntimeError):
    pass


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class MultiHead(nn.Module):
    """Same layout as LaViLa's VideoClassifierMultiHead: visual + fc_cls[verb 97, noun 300, action 3806]."""
    def __init__(self, visual, sizes):
        super().__init__()
        self.visual = visual
        self.fc_cls = nn.ModuleList([nn.Linear(visual.num_features, n) for n in sizes])

    def forward(self, x):                       # x: [B, 3, T, H, W], pixel values normalised
        e = self.visual(x)
        if isinstance(e, (list, tuple)):
            e = e[0]
        return [fc(e) for fc in self.fc_cls]


_MEAN = torch.tensor([108.3272985, 116.7460125, 104.09373615]).view(1, 3, 1, 1, 1)
_STD = torch.tensor([68.5005327, 66.6321579, 70.32316305]).view(1, 3, 1, 1, 1)


class ActionModel:
    def __init__(self):
        path = os.path.join(WEIGHTS, C.ACTION_CKPT)
        if not os.path.exists(path):
            raise ModelsUnavailable(f"missing {path} (run tools/fetch_weights.sh)")
        from third_party.lavila_timesformer import SpaceTimeTransformer
        sd = torch.load(path, map_location="cpu")
        sd = sd.get("state_dict", sd)
        sd = {k.replace("module.", ""): v.float() for k, v in sd.items()}
        sizes = [sd[f"fc_cls.{i}.weight"].shape[0] for i in range(3)]
        vis = SpaceTimeTransformer(num_frames=sd["visual.temporal_embed"].shape[1], time_init="zeros",
                                   attention_style="frozen-in-time", ln_pre=True, act_layer=QuickGELU,
                                   is_tanh_gating=False, drop_path_rate=0.0)
        vis.head = nn.Identity(); vis.pre_logits = nn.Identity(); vis.fc = nn.Identity()
        self.model = MultiHead(vis, sizes)
        res = self.model.load_state_dict(sd, strict=False)
        if res.missing_keys or res.unexpected_keys:
            raise RuntimeError(f"action checkpoint mismatch: missing {res.missing_keys[:5]} unexpected {res.unexpected_keys[:5]}")
        self.dev = device()
        self.model.to(self.dev).eval()
        self.mean, self.std = _MEAN.to(self.dev), _STD.to(self.dev)

    @torch.no_grad()
    def __call__(self, clips):
        """clips: uint8 array [B, T, H, W, 3] -> (verb probs [B, 97], noun probs [B, 300])"""
        x = torch.from_numpy(np.ascontiguousarray(clips)).to(self.dev).permute(0, 4, 1, 2, 3).float()
        x = (x - self.mean) / self.std
        with torch.autocast(self.dev.type, enabled=self.dev.type == "cuda"):
            out = self.model(x)
        return out[0].float().softmax(-1).cpu().numpy(), out[1].float().softmax(-1).cpu().numpy()


def _feat(x):
    return x if torch.is_tensor(x) else x.pooler_output


class Siglip:
    def __init__(self):
        path = os.path.join(WEIGHTS, C.SIGLIP_DIR)
        if not os.path.isdir(path):
            raise ModelsUnavailable(f"missing {path} (run tools/fetch_weights.sh)")
        from transformers import AutoModel, AutoProcessor
        self.dev = device()
        self.model = AutoModel.from_pretrained(path, local_files_only=True).to(self.dev).eval()
        self.proc = AutoProcessor.from_pretrained(path, local_files_only=True)
        self._text = None

    @torch.no_grad()
    def image_emb(self, frames):
        """frames: list of uint8 [H, W, 3] -> L2-normalised [N, D]"""
        inp = self.proc(images=list(frames), return_tensors="pt").to(self.dev)
        e = _feat(self.model.get_image_features(**inp)).float()
        return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()

    @torch.no_grad()
    def text_emb(self, prompts):
        inp = self.proc(text=list(prompts), padding="max_length", truncation=True, return_tensors="pt").to(self.dev)
        e = _feat(self.model.get_text_features(**inp)).float()
        return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()

    def text_side(self):
        """Class embeddings [15, D] (mean over prompts) and excitement directions [14, D] (pos - neg)."""
        if self._text is None:
            cls = np.stack([self.text_emb(L.CLASS_PROMPTS[c]).mean(0) for c in L.CLASSES])
            cls /= np.linalg.norm(cls, axis=1, keepdims=True)
            exc = []
            for c in L.SPORTS:
                pos, neg = L.excitement_prompts(c)
                exc.append(self.text_emb(pos).mean(0) - self.text_emb(neg).mean(0))
            scale = float(self.model.logit_scale.exp().item()) if hasattr(self.model, "logit_scale") else 100.0
            self._text = (cls, np.stack(exc), scale)
        return self._text
