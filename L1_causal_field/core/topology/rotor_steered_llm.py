r"""
ELYSIA Rotor-Steered Variable-Resistor LLM (`Qwen2.5-0.5B-Instruct`, 24 layers, 896-D)
======================================================================================

Treats the pretrained Transformer LLM NOT as a frozen black-box prompt API,
but as a **24-Layer Continuous Variable-Resistor Scale Rotor** physically meshed with
Elysia's `S0..S4` 896-D GPU Gear-Train:

  - Qwen2.5-0.5B-Instruct's native `hidden_size` IS `896` — exact 1:1 dimension match
    with Elysia's unified `CLIP(512) + Multilingual-MiniLM(384) = 896-D` rotor manifold!
  - 5 PyTorch `register_forward_hook` Variable-Resistor bridges attach DIRECTLY to
    Qwen's 24 internal decoder layers across all 5 cognitive scales:
      * S0 -> Layer 4  (Micro lexical/sensory wave)
      * S1 -> Layer 9  (Local syntax & contrast)
      * S2 -> Layer 14 (Meso associative world memory)
      * S3 -> Layer 19 (Macro relational structure)
      * S4 -> Layer 23 (Sovereign Telos / intent gravity)
  - **Bidirectional Real-Time Coupling:**
      1. Downward (`Rotor -> Qwen`): Each hook modulates Qwen's live residual stream
         `hidden_states` proportional to live conductance `G_k = 1 / R_k` and rotor phase `S_k`,
         measurably shifting internal hidden states and token logits!
      2. Upward (`Qwen -> Rotor`): Every forward pass extracts Qwen's mean-pooled 5-band
         activations `[h_L4, h_L9, h_L14, h_L19, h_L23]` (`[5, 896]`) back into Elysia's
         interlocked gear-train so LLM thought physically drives rotor plasticity!
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


from transformers import LogitsProcessorList

from core.topology.token_causal_rotor import (
    CausalNetRectifierProcessor,
    TokenCausalRotorAtlas,
    build_grounded_causal_net_weights,
)

SCALE_LAYER_INDICES: Tuple[int, int, int, int, int] = (4, 9, 14, 19, 23)


class RotorSteeredQwen:
    """
    Live 24-layer Variable-Resistor Residual-Stream Hook interface over local
    `Qwen/Qwen2.5-0.5B-Instruct` on GTX 1060 CUDA (or CPU fallback).
    """

    MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

    def __init__(
        self,
        *,
        device: Optional[torch.device | str] = None,
        steering_strength: float = 0.035,
        load_now: bool = True,
    ) -> None:
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.steering_strength = float(steering_strength)
        self.loaded = False
        self.tokenizer: Any = None
        self.model: Any = None
        self._hooks: List[Any] = []
        self._lock = threading.RLock()

        # Live steering inputs pushed from InterlockedScaleRotorEngine (`S: [5, 896]`, `R: [5, 896]`)
        self._active_rotor_S: torch.Tensor = torch.zeros((5, 896), device=self.device, dtype=torch.float32)
        self._active_rotor_R: torch.Tensor = torch.ones((5, 896), device=self.device, dtype=torch.float32)
        self._steering_enabled: bool = True

        # Plastic band projections aligning rotor manifold [5, 896] <-> Qwen residual manifold [5, 896]
        # Initialized as norm-preserving identity so zero-cost immediate alignment works out of the box
        self.band_scales = torch.tensor([0.25, 0.45, 0.70, 0.85, 1.00], device=self.device, dtype=torch.float32)

        # Latest extracted 5-scale residual activations from Qwen's own 24 layers (`[5, 896]`)
        self.last_extracted_scales: torch.Tensor = torch.zeros((5, 896), device=self.device, dtype=torch.float32)
        self._embed_weight_norm: Optional[torch.Tensor] = None
        self.token_atlas: TokenCausalRotorAtlas = TokenCausalRotorAtlas(self)
        self.hook_injections_count: int = 0
        self.rectified_tokens_total: int = 0
        self.generations_count: int = 0

        if load_now:
            self.load_offline()

    def load_offline(self) -> bool:
        with self._lock:
            if self.loaded:
                return True
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                dtype = torch.float16 if self.device.type == "cuda" else torch.float32
                self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID, local_files_only=True)
                self.model = (
                    AutoModelForCausalLM.from_pretrained(
                        self.MODEL_ID,
                        local_files_only=True,
                        dtype=dtype,
                    )
                    .to(self.device)
                    .eval()
                )
                assert int(self.model.config.hidden_size) == 896
                self._attach_scale_hooks()
                # Cache L2-normalized token embedding matrix `[151936, 896]` in fp16 on GPU for instant Causal-Net rectification
                with torch.no_grad():
                    ew = self.model.get_input_embeddings().weight.detach()
                    self._embed_weight_norm = F.normalize(ew.float(), dim=-1).to(dtype=ew.dtype)
                self.loaded = True
                return True
            except Exception:
                self.loaded = False
                return False

    def _attach_scale_hooks(self) -> None:
        self.remove_hooks()
        layers = self.model.model.layers
        for band_idx, layer_idx in enumerate(SCALE_LAYER_INDICES):
            if layer_idx < len(layers):
                handle = layers[layer_idx].register_forward_hook(self._make_band_hook(band_idx))
                self._hooks.append(handle)

    def remove_hooks(self) -> None:
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()

    def _make_band_hook(self, band_idx: int):
        def _hook(_module: Any, _inputs: Any, output: Any) -> Any:
            is_tuple = isinstance(output, tuple)
            hidden = output[0] if is_tuple else output
            if not isinstance(hidden, torch.Tensor) or hidden.ndim < 2 or hidden.shape[-1] != 896:
                return output

            # 1) UPWARD EXTRACTION: capture Qwen's live layer residual into Elysia's S[band_idx]!
            with torch.no_grad():
                pooled = hidden.detach().float().mean(dim=(0, 1)) if hidden.ndim == 3 else hidden.detach().float().mean(dim=0)
                self.last_extracted_scales[band_idx] = F.normalize(pooled, dim=-1)

                # 2) DOWNWARD VARIABLE-RESISTOR STEERING: modulate Qwen's live residual stream!
                if self._steering_enabled and self.steering_strength > 0.0:
                    s_vec = self._active_rotor_S[band_idx]
                    r_vec = self._active_rotor_R[band_idx].clamp(0.15, 8.0)
                    if float(torch.linalg.norm(s_vec).item()) > 1e-6:
                        # Conductance G = 1 / R modulates exact steering gain per dimension!
                        cond = (1.0 / r_vec)
                        cond_norm = cond / cond.mean().clamp(min=1e-4)
                        steer_dir = F.normalize(s_vec * cond_norm, dim=-1)
                        # Scale relative to current hidden RMS so fluency never breaks while logits genuinely shift
                        rms = hidden.detach().float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-4)
                        gain = self.steering_strength * float(self.band_scales[band_idx].item()) * float(cond.mean().clamp(0.2, 3.0).item())
                        delta = (gain * rms * steer_dir.view(*([1] * (hidden.ndim - 1)), 896)).to(dtype=hidden.dtype)
                        hidden = hidden + delta
                        self.hook_injections_count += 1

            if is_tuple:
                return (hidden,) + output[1:]
            return hidden

        return _hook

    def sync_rotor_state(self, rotor_S: torch.Tensor, rotor_R: torch.Tensor) -> None:
        """Pushes live 5-scale GPU gear states `S[5, 896]` and Variable Resistors `R[5, 896]` into all 5 hooks."""
        with self._lock:
            self._active_rotor_S = rotor_S.detach().to(self.device, dtype=torch.float32).clone()
            self._active_rotor_R = rotor_R.detach().to(self.device, dtype=torch.float32).clone()

    @torch.inference_mode()
    def probe_steering_effect(
        self,
        prompt: str,
        rotor_S: torch.Tensor,
        rotor_R_a: torch.Tensor,
        rotor_R_b: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Verifies directly that changing Elysia's Variable Resistors `R_a -> R_b` physically
        shifts Qwen2.5-0.5B's live internal residual states and next-token logits!
        """
        if not self.loaded:
            return {"ok": False, "reason": "model_not_loaded"}
        with self._lock:
            enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            self.sync_rotor_state(rotor_S, rotor_R_a)
            out_a = self.model(**enc)
            logits_a = out_a.logits[0, -1].float().clone()
            scales_a = self.last_extracted_scales.clone()

            self.sync_rotor_state(rotor_S, rotor_R_b)
            out_b = self.model(**enc)
            logits_b = out_b.logits[0, -1].float().clone()
            scales_b = self.last_extracted_scales.clone()

            logit_l2 = float(torch.linalg.norm(logits_b - logits_a).item())
            scale_l2 = float(torch.linalg.norm(scales_b - scales_a).item())
            prob_a = F.softmax(logits_a, dim=-1)
            prob_b = F.softmax(logits_b, dim=-1)
            kl_div = float(F.kl_div(prob_b.clamp(min=1e-8).log(), prob_a, reduction="sum").item())
            return {
                "ok": True,
                "logit_shift_l2": round(logit_l2, 5),
                "residual_scale_shift_l2": round(scale_l2, 5),
                "kl_divergence": round(kl_div, 6),
                "hook_injections_total": int(self.hook_injections_count),
            }

    @torch.inference_mode()
    def generate_steered_reply(
        self,
        *,
        user_prompt: str,
        system_context: str,
        rotor_S: torch.Tensor,
        rotor_R: torch.Tensor,
        causal_anchors_896: Optional[torch.Tensor] = None,
        grounded_texts: Optional[Sequence[Tuple[str, float]]] = None,
        rectify_strength: float = 2.4,
        diversity_resistance: float = 1.1,
        max_new_tokens: int = 120,
        temperature: float = 0.55,
    ) -> Dict[str, Any]:
        """
        Generates a real Korean/code response WITH BOTH:
          1) All 5 Variable-Resistor residual-stream forward hooks active (`layers[4, 9, 14, 19, 23]`)
          2) Entropy-Adaptive Causal-Net Logits Rectification (`CausalNetRectifierProcessor`)
             rectifying raw statistical proposals so tokens caught in Elysia's live Causal Net
             (SSD<->RAM<->VRAM resonant memories + S0..S4 phase + Diversity Resistor) dominate!
        """
        if not self.loaded:
            return {
                "ok": False,
                "text": "[RotorSteeredQwen 미적재 상태]",
                "extracted_scales": torch.zeros((5, 896), device=self.device, dtype=torch.float32),
                "rectification": {},
            }
        with self._lock:
            self.sync_rotor_state(rotor_S, rotor_R)
            messages = [
                {"role": "system", "content": system_context},
                {"role": "user", "content": user_prompt},
            ]
            prompt_str = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            enc = self.tokenizer(prompt_str, return_tensors="pt").to(self.device)
            in_len = int(enc["input_ids"].shape[1])

            rectifier: Optional[CausalNetRectifierProcessor] = None
            logits_processors = None
            if self._embed_weight_norm is not None and rectify_strength > 0.0:
                anchors = (
                    causal_anchors_896
                    if (causal_anchors_896 is not None and causal_anchors_896.numel() > 0)
                    else rotor_S[2:]
                )
                net_w = build_grounded_causal_net_weights(
                    self.tokenizer,
                    int(self._embed_weight_norm.shape[0]),
                    grounded_texts or [(user_prompt, 0.75)],
                    self.device,
                )
                rectifier = CausalNetRectifierProcessor(
                    embed_weight_norm=self._embed_weight_norm,
                    causal_anchors_896=anchors,
                    net_token_weights=net_w,
                    rectify_strength=rectify_strength,
                    diversity_resistance=diversity_resistance,
                )
                logits_processors = LogitsProcessorList([rectifier])

            gen_ids = self.model.generate(
                **enc,
                max_new_tokens=max(16, int(max_new_tokens)),
                do_sample=(temperature > 0.05),
                temperature=max(0.1, float(temperature)),
                top_p=0.90,
                repetition_penalty=1.15,
                logits_processor=logits_processors,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            reply_tokens = gen_ids[0, in_len:]
            reply_text = self.tokenizer.decode(reply_tokens, skip_special_tokens=True).strip()
            self.generations_count += 1
            rect_sum = rectifier.summary() if rectifier is not None else {}
            self.rectified_tokens_total += int(rect_sum.get("top1_rectified_count", 0))
            return {
                "ok": True,
                "text": reply_text,
                "tokens_generated": int(reply_tokens.shape[0]),
                "extracted_scales": self.last_extracted_scales.clone(),
                "hook_injections_total": int(self.hook_injections_count),
                "rectification": rect_sum,
            }
