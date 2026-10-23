import torch
import torch.nn as nn


def quaternion_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Hamilton Product of 4D Quaternions q = [w, x, y, z].

    q1, q2: Tensors with shape [..., 4]
    """
    w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
    w2, x2, y2, z2 = torch.unbind(q2, dim=-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)


class ComplexPhasePCLayer(nn.Module):
    """
    Complex Phase Oscillator Predictive Coding Layer.

    Uses complex phasors z = A * exp(i * phi) in C^d.
    Local inference relies on complex conjugate and Hermitian Transpose (.mH)
    to back-rotate top-down error waves without global autograd graph traversal.
    """
    def __init__(self, in_features: int, out_features: int, gamma: float = 0.05):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gamma = gamma

        # Complex weights W = Real + i*Imag (rotations and scaling in complex space)
        real_W = torch.randn(out_features, in_features) * 0.1
        imag_W = torch.randn(out_features, in_features) * 0.1
        self.W = nn.Parameter(torch.complex(real_W, imag_W))

        # Dynamic complex state memory
        self.z = None  # Complex representation z = A * exp(i * phi)
        self.e = None  # Complex phase-mismatch error

    def reset_state(self, batch_size: int, device: torch.device):
        """Initialize complex state oscillators."""
        real_z = torch.randn(batch_size, self.out_features, device=device) * 0.1
        imag_z = torch.randn(batch_size, self.out_features, device=device) * 0.1
        self.z = torch.complex(real_z, imag_z)
        self.e = torch.zeros(batch_size, self.in_features, dtype=torch.complex64, device=device)

    def forward_predict(self) -> torch.complex64:
        """Top-down complex rotation prediction z_hat = polar(tanh(|z@W|), angle(z@W))."""
        raw_pred = torch.matmul(self.z, self.W)
        amp = torch.tanh(torch.abs(raw_pred))
        phase = torch.angle(raw_pred)
        return torch.polar(amp, phase)

    def update_state(self, bottom_input: torch.complex64, top_error: torch.complex64 = None):
        """Inference phase: Kuramoto-style local complex phase error convergence."""
        with torch.no_grad():
            z_hat = self.forward_predict()
            self.e = bottom_input - z_hat

            # Bottom-up signal via Hermitian Transpose (W.mH)
            bottom_up_signal = torch.matmul(self.e, self.W.mH)

            if top_error is not None:
                grad_z = bottom_up_signal - top_error
            else:
                grad_z = bottom_up_signal

            self.z += self.gamma * grad_z

    def update_weights(self, lr: float = 0.01):
        """Learning phase: Complex Hebbian local weight rotation update dW = z^H @ e."""
        with torch.no_grad():
            grad_W = torch.matmul(self.z.mH, self.e)
            self.W += lr * grad_W


class QuaternionPhasePCLayer(nn.Module):
    """
    4D Quaternion Phase Oscillator Predictive Coding Layer.

    Represents states as quaternions q in H^d [w, x, y, z].
    Enables 3D spatial rotation dynamics and local torque error matching.
    """
    def __init__(self, in_features: int, out_features: int, gamma: float = 0.05):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gamma = gamma

        # Quaternion weight tensor shape: [out_features, in_features, 4]
        self.W = nn.Parameter(torch.randn(out_features, in_features, 4) * 0.1)

        self.q = None  # Quaternion representation [batch, out_features, 4]
        self.e = None  # Quaternion error [batch, in_features, 4]

    def reset_state(self, batch_size: int, device: torch.device):
        q_raw = torch.randn(batch_size, self.out_features, 4, device=device)
        self.q = q_raw / torch.norm(q_raw, dim=-1, keepdim=True)
        self.e = torch.zeros(batch_size, self.in_features, 4, device=device)

    def forward_predict(self) -> torch.Tensor:
        """Top-down quaternion rotation prediction using Hamilton product."""
        # self.q: [B, Out, 4], self.W: [Out, In, 4]
        B = self.q.size(0)
        q_expand = self.q.unsqueeze(2).expand(B, self.out_features, self.in_features, 4)
        W_expand = self.W.unsqueeze(0).expand(B, self.out_features, self.in_features, 4)

        q_rotated = quaternion_mul(q_expand, W_expand)
        # Sum across out_features dimension to get top-down prediction for each in_feature
        q_pred = torch.sum(q_rotated, dim=1)  # [B, In, 4]
        norm = torch.norm(q_pred, dim=-1, keepdim=True) + 1e-8
        return q_pred / norm

    def update_state(self, bottom_input: torch.Tensor, top_error: torch.Tensor = None):
        """Inference phase: Local quaternion torque mismatch minimization."""
        with torch.no_grad():
            q_hat = self.forward_predict()
            self.e = bottom_input - q_hat  # [B, In, 4]

            # Inverse rotation matrix transform for bottom-up error propagation
            # Conjugate W: [Out, In, 4] -> [Out, In, (w, -x, -y, -z)]
            W_conj = self.W.clone()
            W_conj[..., 1:] = -W_conj[..., 1:]

            B = self.e.size(0)
            e_expand = self.e.unsqueeze(1).expand(B, self.out_features, self.in_features, 4)
            W_conj_expand = W_conj.unsqueeze(0).expand(B, self.out_features, self.in_features, 4)

            err_rotated = quaternion_mul(e_expand, W_conj_expand)
            bottom_up_signal = torch.sum(err_rotated, dim=2)  # [B, Out, 4]

            if top_error is not None:
                grad_q = bottom_up_signal - top_error
            else:
                grad_q = bottom_up_signal

            self.q += self.gamma * grad_q
            self.q = self.q / (torch.norm(self.q, dim=-1, keepdim=True) + 1e-8)

    def update_weights(self, lr: float = 0.01):
        """Learning phase: Quaternion Hebbian rotation update."""
        with torch.no_grad():
            B = self.q.size(0)
            q_conj = self.q.clone()
            q_conj[..., 1:] = -q_conj[..., 1:]

            q_conj_expand = q_conj.unsqueeze(2).expand(B, self.out_features, self.in_features, 4)
            e_expand = self.e.unsqueeze(1).expand(B, self.out_features, self.in_features, 4)

            grad_W = quaternion_mul(q_conj_expand, e_expand).mean(dim=0)
            self.W += lr * grad_W


class ComplexQuaternionPCNetwork(nn.Module):
    """
    Multi-layer Predictive Coding Network composed of Complex/Quaternion Phase Layers.
    Executes local relaxation and Hebbian weight updates without autograd loss backprop.
    """
    def __init__(self, layer_dims: list[int], gamma: float = 0.05, is_quaternion: bool = False):
        super().__init__()
        self.is_quaternion = is_quaternion
        self.layers = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            if is_quaternion:
                self.layers.append(QuaternionPhasePCLayer(layer_dims[i], layer_dims[i+1], gamma=gamma))
            else:
                self.layers.append(ComplexPhasePCLayer(layer_dims[i], layer_dims[i+1], gamma=gamma))

    def fit_step(self, input_wave: torch.Tensor, infer_steps: int = 20, lr: float = 0.01) -> float:
        """Run fast local inference relaxation and slow local weight updates."""
        batch_size = input_wave.size(0)
        device = input_wave.device

        for layer in self.layers:
            layer.reset_state(batch_size, device)

        for _ in range(infer_steps):
            for l in range(len(self.layers)):
                bottom_input = input_wave if l == 0 else self.layers[l - 1].z if not self.is_quaternion else self.layers[l - 1].q
                top_error = self.layers[l + 1].e if l < len(self.layers) - 1 else None
                self.layers[l].update_state(bottom_input, top_error)

        total_error = 0.0
        for layer in self.layers:
            layer.update_weights(lr=lr)
            total_error += torch.mean(torch.abs(layer.e)).item()

        return total_error / len(self.layers)
