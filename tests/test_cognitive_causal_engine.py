import torch
import numpy as np
import pytest
import os
from synaptic_architecture.cognitive_node_engine import CognitiveNodeEngine
from synaptic_architecture.causal_wave_streaming import CausalVideoDecoder, ContiguousCausalStreamBuffer
from synaptic_architecture.cst_container import CSTContainerHandler


def test_cognitive_node_engine_forward():
    num_nodes = 20
    dim = 32
    num_basis = 4
    top_k = 5
    engine = CognitiveNodeEngine(num_nodes=num_nodes, dim=dim, num_basis=num_basis, top_k=top_k)

    batch_size = 4
    x = torch.randn(batch_size, dim)
    c = torch.randn(batch_size, num_basis)

    out, topk_idx, norm_w, loss = engine(x, c)

    assert out.shape == (batch_size, dim)
    assert topk_idx.shape == (batch_size, top_k)
    assert norm_w.shape == (batch_size, top_k)
    assert loss.dim() == 0  # Scalar loss


def test_cognitive_node_split_and_prune():
    num_nodes = 10
    dim = 16
    engine = CognitiveNodeEngine(num_nodes=num_nodes, dim=dim, split_threshold=0.1, prune_threshold=0.9)

    # Force position gradient moving average high to trigger split
    engine.grad_mu_sq_avg.fill_(0.5)

    stats = engine.apply_split_and_prune()

    assert stats["split"] > 0 or stats["pruned"] > 0
    assert engine.current_num_nodes != num_nodes


def test_causal_video_decoder_stream():
    num_nodes = 64
    dim = 32
    decoder = CausalVideoDecoder(num_nodes=num_nodes, dim=dim)

    # Decode I-Frame
    i_state, i_header = decoder.decode_i_frame(0)
    assert i_state.shape == (num_nodes, dim)
    assert i_header.frame_type == "I-FRAME"

    # Decode P-Frame
    delta = torch.randn(num_nodes, dim) * 0.1
    p_state, p_header = decoder.decode_p_frame(delta, timestamp=0.1)
    assert p_state.shape == (num_nodes, dim)
    assert p_header.frame_type == "P-FRAME"

    # Render Latent Output
    rendered = decoder.render_output()
    assert rendered.shape == (dim,)


def test_cst_container_serialization(tmp_path):
    filepath = str(tmp_path / "stream_test.cst")
    handler = CSTContainerHandler(filepath)

    num_nodes = 12
    dim = 16
    topo_w = np.random.randn(num_nodes, num_nodes).astype(np.float32)
    topo_r = np.sign(np.random.randn(num_nodes, num_nodes)).astype(np.float32)

    i_frames = [np.random.randn(num_nodes, dim).astype(np.float32)]
    p_frames = [
        (0.1, np.random.randn(num_nodes, dim).astype(np.float32)),
        (0.2, np.random.randn(num_nodes, dim).astype(np.float32))
    ]

    handler.write_container(num_nodes, dim, topo_w, topo_r, i_frames, p_frames)

    res = handler.read_container()
    assert res["num_nodes"] == num_nodes
    assert res["dim"] == dim
    assert len(res["i_frames"]) == 1
    assert len(res["p_frame_deltas"]) == 2
    assert np.allclose(res["topo_weight"], topo_w)
