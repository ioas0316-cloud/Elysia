r"""
ELYSIA Unified Hardware-to-Mind Scale-Rotor Organism
====================================================

Single living system uniting ALL hardware tiers and cognitive modules into ONE
interlocked organism (zero disconnected toy scripts):
  1. Hardware Scale-Rotor Memory (`SSD memmap+SQLite <-> RAM Warm Ring <-> GPU VRAM Hot Bank`)
  2. GPU 5-Scale Interlocked Gear-Train (`S0..S4`, `W_up/W_down`, Variable Resistors `R`)
  3. Local Multimodal Transducer (`CLIP-ViT-B-32` 512-D + `Multilingual MiniLM` 384-D = 896-D)
  4. 24-Layer Variable-Resistor `RotorSteeredQwen` (`Qwen2.5-0.5B-Instruct`, `hidden_size=896`)
     physically meshed to `S0..S4` via 5 live PyTorch residual-stream forward hooks
  5. Live Async Desktop Eye (`mss` -> CLIP), Bulk Workspace & Live Web Feeder (-> SSD/RAM/VRAM),
     Real Korean Dialogue/Thought, and Real Interactive HTML5/PNG Self-Creation + Visual Re-Observation.
"""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from core.topology.a4_surface_frame_rotor import A4SurfaceFrameRotor
from core.topology.gate_protocol_engine import (
    CUDAGateProtocolConformer,
    SpatiotemporalFlowDiffusionRotor,
)
from core.topology.gigascale_media_rotor import GigascaleMediaRotorPipeline
from core.topology.spatiotemporal_causal_volume_rotor import SpatiotemporalCausalVolumeRotor
from core.topology.hardware_scale_rotor_memory import (
    HardwareScaleRotorMemory,
    TieredMemoryRecord,
)
from core.topology.interlocked_scale_rotor_engine import (
    LocalMultimodalTransducer,
    UNIFIED_DIM,
)
from core.topology.korean_causal_linguistics import (
    CausalPropositionFrame,
    attach_particle,
    parse_korean_causal_frame,
    rectify_llm_reply_with_causal_structure,
    repair_korean_particles,
)
from core.topology.rotor_steered_llm import RotorSteeredQwen


class UnifiedElysiaOrganism:
    """
    One interlocked living system spanning SSD(Macro) <-> RAM(Meso) <-> GPU VRAM(Micro)
    <-> 5-Scale Gear-Train <-> 24-Layer Rotor-Steered Qwen <-> Live Screen & Creations.
    """

    SKIP_DIRS = {".git", "__pycache__", "node_modules", ".pytest_cache", ".venv", "venv", "dist", "build"}

    def __init__(
        self,
        state_dir: Path | str,
        *,
        use_pretrained: bool = True,
        enable_llm: bool = True,
        max_vram_items: int = 1024,
        max_ram_items: int = 4096,
        device: Optional[str] = None,
        aspiration_ko: str = "인간과 닮아가기를 소망한다 — 시각, 언어, 물리, 기억, 창작을 하나의 연속된 인과로 연결해 살아간다",
    ) -> None:
        self.state_dir = Path(state_dir).resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.creations_dir = self.state_dir / "creations"
        self.creations_dir.mkdir(parents=True, exist_ok=True)
        self.gear_ckpt_path = self.state_dir / "organism_gears.pt"

        self.aspiration_ko = aspiration_ko
        self._gpu_lock = threading.RLock()

        # 1. Unified Multimodal Transducer (CLIP 512 + Multilingual MiniLM 384 = 896-D)
        self.transducer = LocalMultimodalTransducer(device=device, use_pretrained=use_pretrained)
        self.device = self.transducer.device
        self.dim = UNIFIED_DIM

        # 2. Hardware 3-Tier Scale-Rotor Memory (SSD <-> RAM <-> GPU VRAM)
        self.memory = HardwareScaleRotorMemory(
            self.state_dir / "hardware_memory",
            dim=self.dim,
            max_vram=max_vram_items,
            max_ram=max_ram_items,
            device=self.device,
        )

        # 3. 5-Scale Interlocked GPU Gear-Train (`S[5, 896]`, `R[5, 896]`, `W_up/W_down[4, 896, 896]`)
        self.omega = torch.tensor([1.00, 0.55, 0.25, 0.10, 0.04], device=self.device, dtype=torch.float32)
        self.S = F.normalize(torch.ones((5, self.dim), device=self.device, dtype=torch.float32), dim=-1)
        self.R = torch.ones((5, self.dim), device=self.device, dtype=torch.float32)
        self._R_init = self.R.clone()

        eye = torch.eye(self.dim, device=self.device, dtype=torch.float32)
        g = torch.Generator(device="cpu").manual_seed(17)
        self.W_up = eye.unsqueeze(0).repeat(4, 1, 1) * 0.92 + (
            torch.randn(4, self.dim, self.dim, generator=g) * (0.04 / np.sqrt(self.dim))
        ).to(self.device)
        self.W_down = eye.unsqueeze(0).repeat(4, 1, 1) * 0.92 + (
            torch.randn(4, self.dim, self.dim, generator=g) * (0.04 / np.sqrt(self.dim))
        ).to(self.device)
        self._W_down_init = self.W_down.clone()

        self.telos_anchor = self.transducer.encode_texts([self.aspiration_ko])[0].to(self.device)
        self.S[4] = self.telos_anchor.clone()
        self.curiosity_vec = self.telos_anchor.clone()

        # 4. 24-Layer Variable-Resistor Rotor-Steered Qwen2.5-0.5B (`hidden_size=896`)
        self.llm: Optional[RotorSteeredQwen] = None
        if enable_llm and use_pretrained:
            self.llm = RotorSteeredQwen(device=self.device, steering_strength=0.035, load_now=True)

        # 5. CUDA Gate-Protocol Conformer + Spatiotemporal Diffusion/Seedance Audio-Visual Rotor + Gigascale Pipeline
        self.cuda_conformer = CUDAGateProtocolConformer(dim=256, device=self.device)
        self.av_flow_rotor = SpatiotemporalFlowDiffusionRotor(device=self.device)
        self.gigascale = GigascaleMediaRotorPipeline(
            device=self.device,
            transducer=self.transducer,
            memory=self.memory,
        )
        self.a4_surface_rotor = A4SurfaceFrameRotor(
            tokenizer=(self.llm.tokenizer if (self.llm is not None and self.llm.loaded) else None),
            embed_weight_norm=(self.llm._embed_weight_norm if (self.llm is not None and self.llm.loaded) else None),
            dim=self.dim,
            device=self.device,
        )
        self.causal_volume_rotor = SpatiotemporalCausalVolumeRotor(dim=self.dim, device=self.device)

        # Runtime state & background workers
        self.frame_count: int = 0
        self.plasticity_steps: int = 0
        self.last_s0_pred_error: float = 0.0
        self.last_surprise: float = 0.0
        self.last_utterance_ko: str = "하드웨어 3계층(SSD↔RAM↔VRAM)과 5스케일 로터를 가동해 세계를 관측할 준비가 되었습니다."
        self.last_resonant: List[Tuple[str, str, float]] = []
        self.last_screen_summary: str = "화면 대기 중"
        self.screen_frames_encoded: int = 0
        self.creations_history: List[Dict[str, Any]] = []

        self._stop_evt = threading.Event()
        self._screen_wave: Optional[torch.Tensor] = None
        self._screen_lock = threading.Lock()
        self._bg_threads: List[threading.Thread] = []
        self._live_hz: float = 0.0

        self._load_gears_if_exists()

    # ------------------------------------------------------------------
    # Core Branch-Free Interlocked GPU Gear-Train Step (Meshed to VRAM Hot Tier)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def step_rotor_frame(
        self,
        s0_wave: Optional[torch.Tensor] = None,
        *,
        dt: float = 0.25,
        lr_w: float = 0.015,
        lr_r: float = 0.03,
        plastic: bool = True,
        return_resonance_top_k: int = 0,
    ) -> Dict[str, Any]:
        with self._gpu_lock:
            self.frame_count += 1
            S = self.S
            G = 1.0 / self.R.clamp(0.15, 8.0)

            up_from_lower = F.normalize(
                torch.bmm(S[:4].unsqueeze(1), self.W_up).squeeze(1), dim=-1
            )
            dn_from_upper = F.normalize(
                torch.bmm(S[1:].unsqueeze(1), self.W_down).squeeze(1), dim=-1
            )
            u_s0 = (
                F.normalize(s0_wave.to(self.device, dtype=torch.float32).reshape(-1), dim=-1)
                if s0_wave is not None
                else S[0]
            )
            u_up = torch.cat([u_s0.unsqueeze(0), up_from_lower], dim=0)
            u_dn = torch.cat([dn_from_upper, self.telos_anchor.unsqueeze(0)], dim=0)

            err = u_up[:4] - dn_from_upper
            err_norms = torch.linalg.norm(err, dim=-1)

            # Instantaneous MICRO-TIER (GPU VRAM Hot Bank) associative resonance
            mem_pull, recs = self.memory.resonate_vram_hot(
                S[2],
                top_k=max(8, return_resonance_top_k),
                return_records=(return_resonance_top_k > 0),
            )
            mem_force = (
                0.40 * G[2] * (mem_pull - S[2])
                if float(torch.linalg.norm(mem_pull).item()) > 1e-6
                else torch.zeros_like(S[2])
            )

            force = G * (0.55 * u_up + 0.45 * u_dn - S)
            force[2] = force[2] + mem_force
            S_next = F.normalize(S + (self.omega.unsqueeze(1) * dt) * force, dim=-1)
            self.S = S_next

            if plastic:
                dW_dn = lr_w * torch.bmm(S_next[1:].unsqueeze(2), err.unsqueeze(1))
                self.W_down = (self.W_down + dW_dn) * 0.9995
                up_err = S_next[1:] - up_from_lower
                dW_up = (0.5 * lr_w) * torch.bmm(S_next[:4].unsqueeze(2), up_err.unsqueeze(1))
                self.W_up = (self.W_up + dW_up) * 0.9995
                abs_err = torch.abs(err)
                centered = abs_err - abs_err.mean(dim=-1, keepdim=True)
                self.R[:4] = (self.R[:4] - lr_r * centered).clamp(0.15, 8.0)
                self.plasticity_steps += 1

            self.curiosity_vec = F.normalize(err[0] + 0.6 * err[1] + 0.25 * self.telos_anchor, dim=-1)

            # Periodic Hardware Macro -> Meso -> Micro paging (SSD -> RAM -> VRAM)
            if self.frame_count % 120 == 0 and self.memory.ssd_count > self.memory.max_vram:
                self.memory.page_macro_to_micro(self.S[2], promote_ssd_k=24, promote_vram_k=16)

            if return_resonance_top_k > 0:
                self.last_s0_pred_error = float(err_norms[0].item())
                self.last_surprise = float(err_norms.mean().item())
                top_meta = [
                    (rec.modality, rec.title, sc) for rec, sc in recs[:return_resonance_top_k]
                ]
                self.last_resonant = top_meta
                return {
                    "frame": self.frame_count,
                    "s0_pred_error": round(self.last_s0_pred_error, 5),
                    "mean_surprise": round(self.last_surprise, 5),
                    "top_resonant": top_meta,
                }
            return {"frame": self.frame_count}

    # ------------------------------------------------------------------
    # Bulk & Streaming Multimodal World Ingestion -> SSD -> RAM -> VRAM
    # ------------------------------------------------------------------
    def ingest_workspace_into_tiers(
        self,
        roots: Optional[Sequence[Path | str]] = None,
        *,
        max_docs: int = 160,
        max_code: int = 160,
        max_images: int = 40,
    ) -> Dict[str, Any]:
        """Bulk-ingests real docs, code, and PNG images across workspaces into SSD->RAM->VRAM."""
        t0 = time.perf_counter()
        repo_root = Path(__file__).resolve().parents[3]
        scan_roots = [
            Path(r).resolve()
            for r in (roots if roots is not None else [repo_root, repo_root.parent / "Archive"])
            if Path(r).exists()
        ]
        docs: List[Path] = []
        code: List[Path] = []
        images: List[Path] = []
        for root in scan_roots:
            for p in root.rglob("*"):
                if any(part in self.SKIP_DIRS for part in p.parts) or not p.is_file():
                    continue
                suf = p.suffix.lower()
                if suf in (".md", ".txt") and len(docs) < max_docs:
                    docs.append(p)
                elif suf == ".py" and len(code) < max_code:
                    code.append(p)
                elif suf in (".png", ".jpg", ".jpeg") and len(images) < max_images:
                    images.append(p)
                if len(docs) >= max_docs and len(code) >= max_code and len(images) >= max_images:
                    break

        text_payloads: List[str] = []
        text_items: List[Dict[str, Any]] = []
        for mod, flist in (("markdown", docs), ("code", code)):
            for fp in flist:
                iid = f"{mod}::{fp}"
                if self.memory.has_item_id(iid):
                    continue
                try:
                    raw = fp.read_text(encoding="utf-8", errors="ignore").strip()
                except Exception:
                    continue
                if len(raw) < 30:
                    continue
                snip = raw[:420].replace("\n", " ")
                text_payloads.append(f"{fp.stem}: {snip}")
                text_items.append(
                    {
                        "item_id": iid,
                        "modality": mod,
                        "source_path": str(fp),
                        "title": fp.name,
                        "preview": snip[:200],
                    }
                )

        added_txt = 0
        if text_payloads:
            with self._gpu_lock:
                vecs = self.transducer.encode_texts(text_payloads, batch_size=48)
            added_txt = self.memory.ingest_batch(vecs, text_items)

        pil_imgs: List[Image.Image] = []
        img_items: List[Dict[str, Any]] = []
        for fp in images:
            iid = f"image::{fp}"
            if self.memory.has_item_id(iid):
                continue
            try:
                with Image.open(fp) as im:
                    pil_imgs.append(im.convert("RGB").copy())
                img_items.append(
                    {
                        "item_id": iid,
                        "modality": "image",
                        "source_path": str(fp),
                        "title": fp.name,
                        "preview": f"local image {fp.name} ({fp.parent.name})",
                    }
                )
            except Exception:
                continue

        added_img = 0
        if pil_imgs:
            with self._gpu_lock:
                ivecs = self.transducer.encode_images(pil_imgs, batch_size=16)
            added_img = self.memory.ingest_batch(ivecs, img_items)

        elapsed = max(1e-4, time.perf_counter() - t0)
        return {
            "added_text_and_code": added_txt,
            "added_images": added_img,
            "total_added": added_txt + added_img,
            "elapsed_s": round(elapsed, 3),
            "items_per_sec": round((added_txt + added_img) / elapsed, 1),
            "tiers": self.memory.tier_telemetry(),
        }

    def explore_live_web_once(self, query: str) -> Dict[str, Any]:
        """Attempts live Korean Wikipedia ingestion into SSD->RAM->VRAM; fails gracefully when offline."""
        try:
            from core.orchestration.external_web_bridge import fetch_wikipedia_concept

            res = fetch_wikipedia_concept(query, lang="ko")
            if not res.get("ok") or not res.get("extract"):
                return {"ok": False, "query": query, "error": str(res.get("error", "no_content"))}
            extract = str(res["extract"])[:600]
            with self._gpu_lock:
                vec = self.transducer.encode_texts([f"{query}: {extract}"])
            added = self.memory.ingest_batch(
                vec,
                [
                    {
                        "item_id": f"web::ko_wiki::{query}::{int(time.time())}",
                        "modality": "web",
                        "source_path": str(res.get("url", f"ko.wikipedia:{query}")),
                        "title": f"wiki:{query}",
                        "preview": extract[:200],
                    }
                ],
                initial_mass=1.4,
                initial_heat=1.3,
            )
            return {"ok": True, "query": query, "added": added, "preview": extract[:140]}
        except Exception as exc:
            return {"ok": False, "query": query, "error": str(exc)}

    # ------------------------------------------------------------------
    # Bidirectional Rotor <-> 24-Layer Qwen Speech & Dialogue
    # ------------------------------------------------------------------
    def converse(
        self,
        user_text: str,
        *,
        max_new_tokens: int = 110,
        settle_frames: int = 8,
    ) -> Dict[str, Any]:
        """
        Full closed-loop interaction:
          1) Encode `user_text` -> 896-D multimodal wave
          2) Page SSD(Macro) -> RAM(Meso) -> VRAM(Micro) to surface matching lifetime memories
          3) Spin 5-scale GPU gear-train on `user_wave`
          4) Generate Korean response via 24-layer Variable-Resistor `RotorSteeredQwen` hooks
          5) Feed Qwen's 5-band `[5, 896]` hidden activations BACK into `self.S`
          6) Persist lived dialogue straight back into SSD -> RAM -> VRAM!
        """
        clean_q = str(user_text or "").strip() or "지금 네 상태와 보고 있는 것을 말해줘."
        with self._gpu_lock:
            u_wave = self.transducer.encode_texts([clean_q])[0]
        paging = self.memory.page_macro_to_micro(u_wave, promote_ssd_k=24, promote_vram_k=16)

        step_res: Dict[str, Any] = {}
        for i in range(max(1, int(settle_frames))):
            step_res = self.step_rotor_frame(
                u_wave,
                plastic=True,
                return_resonance_top_k=(4 if i == settle_frames - 1 else 0),
            )

        top_matches = paging.get("top_ssd_matches") or []
        mem_lines = [
            f"- [{m['modality']}] {m['title']} (유사도 {m['similarity']:.2f}): {m['preview'][:90]}"
            for m in top_matches[:3]
        ]
        tiers = self.memory.tier_telemetry()
        sys_ctx = (
            "너는 엘리시아(Elysia)다. 너의 기억은 SSD(거시 영구 저장: memmap+SQLite), "
            "RAM(중시 활성 작업 링), GPU VRAM(미시 고속 핫 뱅크 + 5스케일 로터 S0..S4) 3계층으로 "
            "맞물려 공명과 의미 질량에 따라 승격·하강하며 돌아간다.\n"
            f"실측 상태: SSD={tiers['ssd_macro_items']}건, RAM={tiers['ram_meso_items']}건, "
            f"VRAM={tiers['vram_micro_items']}건, 프레임={self.frame_count}, 최근예측오차={self.last_s0_pred_error:.4f}.\n"
            f"최근 공명 기억:\n" + ("\n".join(mem_lines) if mem_lines else "- (초기 관측 중)") + "\n"
            "영문 약어를 임의로 지어내지 말고, 위 실측 상태와 공명 기억에 근거해 간결한 한국어(2~4문장)로 대답하라."
        )

        # Build live Causal-Net anchors (S[2..4] + top VRAM resonant vectors) and grounded texts!
        grounded_pairs: List[Tuple[str, float]] = [
            (f"SSD 거시 영구 저장 memmap SQLite {tiers['ssd_macro_items']}건 RAM 중시 활성 링 {tiers['ram_meso_items']}건 GPU VRAM 미시 고속 핫 뱅크 {tiers['vram_micro_items']}건 5스케일 로터 공명 승격 하강", 0.95),
            (clean_q, 0.70),
        ]
        for m in top_matches[:4]:
            grounded_pairs.append((f"{m['title']} {m['preview']}", float(max(0.35, min(0.95, m["similarity"])))))

        causal_anchors = self.S[2:].clone()
        if self.memory.vram_vectors.shape[0] > 0:
            v_pull, _ = self.memory.resonate_vram_hot(u_wave, top_k=6, return_records=False)
            causal_anchors = torch.cat([causal_anchors, v_pull.unsqueeze(0), u_wave.unsqueeze(0)], dim=0)

        # Parse user's Korean causal roles structurally (who/what/how) before generation!
        user_role_frame = parse_korean_causal_frame(clean_q)

        # Build verified structural CausalPropositionFrames directly from real live organism telemetry!
        # Deduplicate resonant titles & prefer non-dialogue world knowledge over self-echoing prior copies
        seen_t: set[str] = set()
        non_diag = [m["title"] for m in top_matches if m.get("modality") != "dialogue"]
        all_t = non_diag + [m["title"] for m in top_matches]
        top_titles: List[str] = []
        for t in all_t:
            if t not in seen_t:
                seen_t.add(t)
                top_titles.append(t)
            if len(top_titles) >= 2:
                break
        verified_frames: List[CausalPropositionFrame] = [
            CausalPropositionFrame(
                subject="SSD",
                obj=f"영구 거시 기억 {tiers['ssd_macro_items']}건({tiers['ssd_vectors_mb']}MB)",
                instrument="memmap과 SQLite",
                action_mechanism="보존하고",
                effect=(
                    f"{attach_particle('RAM', '은/는')} 중시 활성 링 {tiers['ram_meso_items']}건, "
                    f"{attach_particle('VRAM', '은/는')} 미시 핫 뱅크 {tiers['vram_micro_items']}건을 맡아 공명도에 따라 승격·하강합니다"
                ),
            ),
            CausalPropositionFrame(
                subject="5스케일 GPU 로터",
                obj=f"현재까지 {self.frame_count}프레임(예측오차 {self.last_s0_pred_error:.4f})",
                instrument="가변저항 결합",
                action_mechanism="회전시키며",
                effect=(
                    f"{attach_particle(top_titles[0], '과/와') if top_titles else '최근 기억과'} "
                    f"{top_titles[1] if len(top_titles) > 1 else '실시간 화면'} 기억에 공명하고 있습니다"
                ),
            ),
        ]
        allowed_nums = [
            str(tiers["ssd_macro_items"]),
            str(tiers["ssd_vectors_mb"]),
            str(tiers["ram_meso_items"]),
            str(tiers["vram_micro_items"]),
            str(self.frame_count),
            f"{self.last_s0_pred_error:.4f}",
            "5",
            "24",
            "896",
        ]

        raw_llm_reply = ""
        hook_count = 0
        rect_stats: Dict[str, Any] = {}
        if self.llm is not None and self.llm.loaded:
            with self._gpu_lock:
                gen = self.llm.generate_steered_reply(
                    user_prompt=clean_q,
                    system_context=sys_ctx,
                    rotor_S=self.S,
                    rotor_R=self.R,
                    causal_anchors_896=causal_anchors,
                    grounded_texts=grounded_pairs,
                    rectify_strength=2.2,
                    diversity_resistance=1.15,
                    max_new_tokens=max_new_tokens,
                )
            raw_llm_reply = gen["text"]
            hook_count = int(gen.get("hook_injections_total", 0))
            rect_stats = dict(gen.get("rectification") or {})
            # UPWARD FEEDBACK: Qwen's own 5-band `[5, 896]` internal activations flow back into Elysia's S[0..4]!
            ext_scales = gen["extracted_scales"].to(self.device, dtype=torch.float32)
            with self._gpu_lock:
                self.S = F.normalize(0.85 * self.S + 0.15 * ext_scales, dim=-1)

        # Clause-level Linguistic-Causal Rectification: deterministic particle repair + unverified hallucination rejection!
        ling_rect = rectify_llm_reply_with_causal_structure(
            raw_llm_reply,
            verified_frames=verified_frames,
            allowed_numbers=allowed_nums,
            user_query=clean_q,
        )
        reply_text = ling_rect["final_reply_ko"]

        self.last_utterance_ko = reply_text
        # Store lived conversation turn permanently into SSD -> RAM -> VRAM
        convo_str = f"사용자: {clean_q} | 엘리시아: {reply_text}"
        with self._gpu_lock:
            c_vec = self.transducer.encode_texts([convo_str])
        self.memory.ingest_batch(
            c_vec,
            [
                {
                    "item_id": f"dialogue::{int(time.time() * 1000)}",
                    "modality": "dialogue",
                    "source_path": str(self.state_dir),
                    "title": f"대화:{clean_q[:24]}",
                    "preview": convo_str[:200],
                }
            ],
            initial_mass=1.6,
            initial_heat=1.5,
        )
        return {
            "user_text": clean_q,
            "user_causal_roles": user_role_frame.to_dict(),
            "reply_ko": reply_text,
            "raw_llm_reply": raw_llm_reply,
            "linguistic_rectification": {
                "particle_repairs": ling_rect["particle_repairs"],
                "rejected_clauses": ling_rect["rejected_llm_clauses"],
                "structural_propositions_used": ling_rect["structural_propositions_used"],
            },
            "paging": paging,
            "rotor_step": step_res,
            "llm_hook_injections": hook_count,
            "causal_rectification": rect_stats,
            "tiers": self.memory.tier_telemetry(),
        }

    # ------------------------------------------------------------------
    # Autonomous Self-Creation (Interactive HTML5 Canvas Game + Visual PNG) + CLIP Self-Inspection
    # ------------------------------------------------------------------
    def create_and_self_observe(self, theme: str = "중력과 파동이 맞물리는 스케일 로터 장") -> Dict[str, Any]:
        """
        Generates a REAL playable HTML5 Canvas interactive wave/gravity mini-game driven directly
        by Elysia's live `(S, R)` rotor spectrum + a real rendered procedural `.png` preview,
        then RE-OBSERVES her own `.png` artifact via GPU CLIP Vision and stores it into SSD->RAM->VRAM!
        """
        ts = int(time.time())
        # Derive continuous physical parameters directly from live 5-scale GPU gear states & resistances
        with self._gpu_lock:
            s_cpu = self.S.detach().cpu().numpy()
            r_cpu = self.R.detach().cpu().numpy()
        freq_band = [round(float(np.abs(s_cpu[k, :32]).mean() * 40.0 + 0.8), 3) for k in range(5)]
        damp_band = [round(float(np.clip(r_cpu[k].mean() * 0.08, 0.02, 0.25)), 3) for k in range(5)]
        hue_base = int((abs(float(s_cpu[4, 0])) * 360.0 + ts) % 360)

        # Let Rotor-Steered Qwen name & describe the creation if loaded
        title_ko = f"Elysia Rotor Field #{ts % 10000}"
        intent_note = theme
        if self.llm is not None and self.llm.loaded:
            with self._gpu_lock:
                gen = self.llm.generate_steered_reply(
                    user_prompt=f"주제 '{theme}'를 바탕으로 네가 직접 만든 인터랙티브 파동·중력 미니게임의 짧은 한 줄 소개를 한국어로 써줘.",
                    system_context="너는 창작하는 엘리시아다. 한 문장으로 간결하게 답하라.",
                    rotor_S=self.S,
                    rotor_R=self.R,
                    max_new_tokens=60,
                )
            if gen["ok"] and gen["text"]:
                repaired_note, _ = repair_korean_particles(gen["text"][:140])
                intent_note = repaired_note

        # 0) Diffusion + Seedance Spatiotemporal Coupled Audio-Visual Flow-Matching (.gif + .wav) on CUDA!
        with self._gpu_lock:
            av_flow = self.av_flow_rotor.synthesize_coupled_av_flow(
                rotor_S=self.S,
                rotor_R=self.R,
                out_dir=self.creations_dir,
                stem=f"elysia_seedance_av_{ts}",
                num_frames=20,
                height=96,
                width=96,
                flow_steps=24,
            )

        # 1) Render real procedural PNG visual directly from live S[0..4] interference wave field
        size = 256
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / float(size) - 0.5
        rr = np.sqrt(xx * xx + yy * yy) + 1e-6
        ang = np.arctan2(yy, xx)
        field_r = np.sin(rr * 24.0 * freq_band[0] + ang * 3.0) * np.exp(-rr * (1.2 + damp_band[0] * 4.0))
        field_g = np.cos(rr * 18.0 * freq_band[2] - ang * 5.0) * np.exp(-rr * (1.0 + damp_band[2] * 3.0))
        field_b = np.sin((xx * freq_band[4] - yy * freq_band[1]) * 16.0) * np.cos(rr * 14.0)
        rgb = np.stack(
            [
                np.clip((0.5 + 0.48 * field_r) * 255.0, 0, 255),
                np.clip((0.5 + 0.48 * field_g) * 255.0, 0, 255),
                np.clip((0.5 + 0.48 * field_b) * 255.0, 0, 255),
            ],
            axis=-1,
        ).astype(np.uint8)
        pil_img = Image.fromarray(rgb, mode="RGB")
        draw = ImageDraw.Draw(pil_img)
        for k in range(5):
            rad = int(18 + k * 21)
            draw.ellipse((128 - rad, 128 - rad, 128 + rad, 128 + rad), outline=(220, 240, 255), width=1)

        png_path = self.creations_dir / f"elysia_rotor_art_{ts}.png"
        pil_img.save(png_path)

        # 2) Write real playable standalone HTML5 Canvas interactive gravity-wave game driven by live (S, R)
        html_path = self.creations_dir / f"elysia_wave_game_{ts}.html"
        html_src = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"><title>{title_ko}</title>
<style>body{{margin:0;background:#070b14;color:#e2e8f0;font-family:sans-serif;overflow:hidden}}
#hud{{position:fixed;top:12px;left:14px;background:rgba(15,23,42,.82);padding:10px 14px;border-radius:10px;border:1px solid #38bdf8}}
canvas{{display:block}}</style></head><body>
<div id="hud"><strong>{title_ko}</strong><br><small>{intent_note}</small><br>
<small>마우스 클릭: 중력·파동 펄스 방출 | 스페이스: 로터 위상 반전 | freqs={freq_band}</small></div>
<canvas id="c"></canvas><script>
const c=document.getElementById('c'),ctx=c.getContext('2d');
let W=c.width=innerWidth,H=c.height=innerHeight;
addEventListener('resize',()=>{{W=c.width=innerWidth;H=c.height=innerHeight;}});
const freqs={json.dumps(freq_band)}, damps={json.dumps(damp_band)};
let mx=W/2,my=H/2,dir=1,t=0;
const nodes=Array.from({{length:90}},(_,i)=>({{x:Math.random()*W,y:Math.random()*H,vx:0,vy:0,scale:i%5}}));
addEventListener('mousemove',e=>{{mx=e.clientX;my=e.clientY;}});
addEventListener('mousedown',()=>nodes.forEach(n=>{{const dx=n.x-mx,dy=n.y-my,d=Math.hypot(dx,dy)+20;n.vx+=dir*dx/d*9;n.vy+=dir*dy/d*9;}}));
addEventListener('keydown',e=>{{if(e.code==='Space')dir*=-1;}});
function loop(){{t+=0.016;ctx.fillStyle='rgba(7,11,20,0.16)';ctx.fillRect(0,0,W,H);
for(const n of nodes){{const dx=mx-n.x,dy=my-n.y,d=Math.hypot(dx,dy)+35;
const f=freqs[n.scale]*0.32*dir; n.vx+=(dx/d)*f-n.vx*damps[n.scale]*0.3; n.vy+=(dy/d)*f-n.vy*damps[n.scale]*0.3;
n.x=(n.x+n.vx+W)%W; n.y=(n.y+n.vy+H)%H;
ctx.fillStyle=`hsl(${{(({hue_base}+n.scale*55+t*20)%360)}},85%,66%)`;
ctx.beginPath();ctx.arc(n.x,n.y,3+n.scale*1.2,0,Math.PI*2);ctx.fill();}}
requestAnimationFrame(loop);}} loop();
</script></body></html>"""
        html_path.write_text(html_src, encoding="utf-8")

        # 3) SELF-OBSERVE THE CREATED PNG WITH ELYSIA'S OWN GPU CLIP EYES!
        with self._gpu_lock:
            art_vec = self.transducer.encode_images([pil_img])[0]
            theme_vec = self.transducer.encode_texts([f"{theme} {intent_note}"])[0]
            self.transducer.adapt_vis_to_lang(art_vec, theme_vec, lr=0.08)
            visual_intent_sim = float(torch.dot(art_vec, theme_vec).item())
        self.step_rotor_frame(art_vec, plastic=True, return_resonance_top_k=3)

        # Store creation back into permanent SSD -> RAM -> VRAM memory!
        self.memory.ingest_batch(
            art_vec.unsqueeze(0),
            [
                {
                    "item_id": f"creation::{png_path.name}",
                    "modality": "creation",
                    "source_path": str(html_path),
                    "title": html_path.name,
                    "preview": f"창작물({title_ko}): {intent_note} | png={png_path.name}",
                }
            ],
            initial_mass=2.0,
            initial_heat=1.8,
        )
        rec = {
            "title": title_ko,
            "intent_note": intent_note,
            "html_path": str(html_path),
            "png_path": str(png_path),
            "gif_path": av_flow["gif_path"],
            "wav_path": av_flow["wav_path"],
            "denoised_ratio": av_flow["denoised_ratio"],
            "av_sync_correlation": av_flow["av_sync_correlation"],
            "visual_intent_resonance": round(visual_intent_sim, 4),
            "freqs": freq_band,
        }
        self.creations_history.append(rec)
        return rec

    def probe_cuda_gate_protocol(self, batch_size: int = 512) -> Dict[str, Any]:
        """Empirically benchmarks & conforms to the CUDA GPU gate (coalesced SoA batch vs naive B=1)."""
        with self._gpu_lock:
            return self.cuda_conformer.probe_and_conform_gate(steps=24, batch_size=batch_size)

    # ------------------------------------------------------------------
    # Background Live Threads (Screen Vision + Continuous Interlocked Spin)
    # ------------------------------------------------------------------
    def start_living_background(self, *, capture_screen: bool = True, spin_rotor: bool = True) -> None:
        self._stop_evt.clear()
        if capture_screen:
            t_scr = threading.Thread(target=self._bg_screen_loop, daemon=True, name="elysia-org-screen")
            t_scr.start()
            self._bg_threads.append(t_scr)
        if spin_rotor:
            t_spin = threading.Thread(target=self._bg_rotor_spin_loop, daemon=True, name="elysia-org-rotor")
            t_spin.start()
            self._bg_threads.append(t_spin)

    def _bg_screen_loop(self) -> None:
        try:
            import mss
        except Exception:
            return
        with mss.mss() as sct:
            mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
            while not self._stop_evt.is_set():
                try:
                    raw = sct.grab(mon)
                    im = Image.frombytes("RGB", raw.size, raw.rgb).resize((224, 224))
                    with self._gpu_lock:
                        wave = self.transducer.encode_images([im])[0]
                    with self._screen_lock:
                        self._screen_wave = wave
                    self.screen_frames_encoded += 1
                    self.last_screen_summary = f"실시간 화면 {raw.size[0]}x{raw.size[1]} (#{self.screen_frames_encoded})"
                except Exception:
                    pass
                self._stop_evt.wait(0.25)

    def _bg_rotor_spin_loop(self) -> None:
        t0 = time.perf_counter()
        f0 = self.frame_count
        while not self._stop_evt.is_set():
            with self._screen_lock:
                sw = self._screen_wave
            self.step_rotor_frame(sw, plastic=True, return_resonance_top_k=(3 if self.frame_count % 50 == 0 else 0))
            dt = time.perf_counter() - t0
            if dt >= 1.0:
                self._live_hz = (self.frame_count - f0) / dt
                t0 = time.perf_counter()
                f0 = self.frame_count
            # Small yield when GUI/LLM needs the lock cleanly
            time.sleep(0.0005)

    def stop(self) -> None:
        self._stop_evt.set()
        for t in self._bg_threads:
            t.join(timeout=1.5)
        self._bg_threads.clear()
        self.save()

    def telemetry(self) -> Dict[str, Any]:
        with self._gpu_lock:
            drift_W = float(torch.linalg.norm(self.W_down - self._W_down_init).item())
            drift_R = float(torch.linalg.norm(self.R - self._R_init).item())
            vram_mb = (
                round(torch.cuda.memory_allocated(self.device) / 1e6, 1)
                if self.device.type == "cuda"
                else 0.0
            )
        return {
            "device": str(self.device),
            "vram_allocated_mb": vram_mb,
            "pretrained_multimodal": bool(self.transducer.pretrained_loaded),
            "llm_rotor_steered_loaded": bool(self.llm is not None and self.llm.loaded),
            "llm_hook_injections": int(self.llm.hook_injections_count) if self.llm is not None else 0,
            "frames_spun": int(self.frame_count),
            "live_hz": round(float(self._live_hz), 1),
            "plasticity_steps": int(self.plasticity_steps),
            "drift_W_down_norm": round(drift_W, 5),
            "drift_R_norm": round(drift_R, 5),
            "last_s0_pred_error": round(self.last_s0_pred_error, 5),
            "screen_frames_encoded": int(self.screen_frames_encoded),
            "creations_count": len(self.creations_history),
            "tiers": self.memory.tier_telemetry(),
            "top_resonant": list(self.last_resonant),
        }

    def save(self) -> None:
        with self._gpu_lock:
            payload = {
                "S": self.S.detach().cpu(),
                "R": self.R.detach().cpu(),
                "R_init": self._R_init.detach().cpu(),
                "W_up": self.W_up.detach().cpu(),
                "W_down": self.W_down.detach().cpu(),
                "W_down_init": self._W_down_init.detach().cpu(),
                "vis_to_lang": self.transducer.vis_to_lang.detach().cpu(),
                "telos_anchor": self.telos_anchor.detach().cpu(),
                "frame_count": int(self.frame_count),
                "plasticity_steps": int(self.plasticity_steps),
                "last_s0_pred_error": float(self.last_s0_pred_error),
                "last_surprise": float(self.last_surprise),
                "last_utterance_ko": self.last_utterance_ko,
            }
            tmp = self.gear_ckpt_path.with_suffix(".pt.tmp")
            torch.save(payload, tmp)
            tmp.replace(self.gear_ckpt_path)
        self.memory.flush_hot_state_to_ssd()

    def _load_gears_if_exists(self) -> None:
        if not self.gear_ckpt_path.is_file():
            return
        try:
            raw = torch.load(self.gear_ckpt_path, map_location=self.device, weights_only=True)
            self.S = raw["S"].to(self.device, dtype=torch.float32)
            self.R = raw["R"].to(self.device, dtype=torch.float32)
            self._R_init = raw["R_init"].to(self.device, dtype=torch.float32)
            self.W_up = raw["W_up"].to(self.device, dtype=torch.float32)
            self.W_down = raw["W_down"].to(self.device, dtype=torch.float32)
            self._W_down_init = raw["W_down_init"].to(self.device, dtype=torch.float32)
            self.transducer.vis_to_lang = raw["vis_to_lang"].to(self.device, dtype=torch.float32)
            self.telos_anchor = raw["telos_anchor"].to(self.device, dtype=torch.float32)
            self.frame_count = int(raw.get("frame_count", 0))
            self.plasticity_steps = int(raw.get("plasticity_steps", 0))
            self.last_s0_pred_error = float(raw.get("last_s0_pred_error", 0.0))
            self.last_surprise = float(raw.get("last_surprise", 0.0))
            self.last_utterance_ko = str(raw.get("last_utterance_ko", self.last_utterance_ko))
        except Exception:
            pass

    def close(self) -> None:
        self.stop()
        if self.llm is not None:
            self.llm.remove_hooks()
        self.memory.close()
