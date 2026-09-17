"""
Unit tests for Execution Manifold, Recalibration, Physical Grounding,
Retrocausal Reinterpretation, and Self-Referential Meta Engine.
"""

import pytest
import torch
import torch.nn.functional as F
from core.physics.execution_manifold import (
    AssemblyTraceVectorizer,
    MemoryCognitiveEngine,
    GroundedCognitiveEngine,
    retrocausal_reinterpretation_step,
    SelfReferentialMetaEngine
)
from core.physics.execution_phase_lock_op import execution_phase_lock


def test_assembly_trace_vectorizer():
    vectorizer = AssemblyTraceVectorizer(d_m=64, d_phi=16, d_s=48)
    macro_target = torch.randn(48)
    frame = {
        'RIP': 0x401000,
        'RSP': 0x7fff0000,
        'RAX': 0x1234,
        'RBX': 0x5678,
        'RCX': 0x9abc,
        'RDX': 0xdef0,
        'mem_addr': 0x7fff0008,
        'mem_write': 1
    }
    tensor = vectorizer.process_frame(frame, macro_target)
    assert tensor.shape == (128,)
    assert not torch.isnan(tensor).any()


def test_memory_cognitive_engine_recalibration():
    engine = MemoryCognitiveEngine(d_m=64, d_phi=16, d_s=48)
    Z_exec = torch.randn(2, 128, requires_grad=True)
    target_s = torch.randn(2, 48)

    Z_next, E_cross = engine(Z_exec, target_s)
    assert Z_next.shape == (2, 128)
    assert E_cross.shape == (2,)
    assert E_cross.mean().item() > 0.0


def test_grounded_cognitive_engine():
    engine = GroundedCognitiveEngine(d_m=64, d_v=32, d_a=16, d_mech=16, d_phi=16, d_s=48)
    z_m = torch.randn(2, 64)
    x_v = torch.randn(2, 32)
    x_a = torch.randn(2, 16)
    x_mech = torch.randn(2, 16)
    phi = torch.randn(2, 16)
    target_s = torch.randn(2, 48)

    steered, e_grounded = engine(z_m, x_v, x_a, x_mech, phi, target_s)
    assert steered.shape == (2, 144)  # 64+32+16+16+16 = 144
    assert e_grounded.shape == (2,)


def test_retrocausal_reinterpretation():
    proj_op = torch.nn.Linear(128, 48)
    T_steps = 5
    batch_size = 2
    z_trajectory = torch.randn(T_steps, batch_size, 128)
    target_s = torch.randn(batch_size, 48)

    precursor_masks = retrocausal_reinterpretation_step(
        z_trajectory, target_s, proj_op, theta_causal=0.01
    )
    assert precursor_masks.shape == (T_steps - 1, batch_size)


def test_self_referential_meta_engine():
    meta_engine = SelfReferentialMetaEngine(d_m=64, d_v=32, d_a=16, d_mech=16, d_phi=16, d_s=48)
    target_s = torch.randn(2, 48)
    optimizer = torch.optim.AdamW(meta_engine.parameters(), lr=1e-3)

    trajectory_inputs = [
        (torch.randn(2, 64), torch.randn(2, 32), torch.randn(2, 16), torch.randn(2, 16), torch.randn(2, 16))
        for _ in range(4)
    ]

    z_traj, total_drift = meta_engine.forward_trajectory(trajectory_inputs, target_s)
    assert z_traj.shape == (4, 2, 144)

    meta_loss_val = meta_engine.meta_update(z_traj, target_s, optimizer)
    assert isinstance(meta_loss_val, float)


def test_execution_phase_lock_op():
    z_in = torch.randn(3, 128)
    target_phase = torch.tensor([0.0, 1.0, 3.14])
    z_out = execution_phase_lock(z_in, target_phase, learning_rate=0.05)
    assert z_out.shape == (3, 128)
