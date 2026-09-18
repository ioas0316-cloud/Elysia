import heapq
import torch
import torch.nn as nn
from dataclasses import dataclass, field


@dataclass(order=True)
class EventPacket:
    priority: float
    target_node: int
    payload: torch.Tensor = field(compare=False)


class STEEventGate(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) for Event Spreading Gate.
    Forward: Hard binary step function (delta > threshold -> 1.0, else 0.0)
    Backward: Straight-through identity gradient pass-through.
    """
    @staticmethod
    def forward(ctx, delta: torch.Tensor, threshold: float) -> torch.Tensor:
        ctx.save_for_backward(delta)
        ctx.threshold = threshold
        return (delta > threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Straight-through estimator passes gradient as-is for delta, None for threshold scalar
        return grad_output, None


def ste_event_gate(delta: torch.Tensor, threshold: float) -> torch.Tensor:
    """Wrapper function for STEEventGate."""
    return STEEventGate.apply(delta, threshold)


class RelaxedEventGate(nn.Module):
    """
    Continuous Relaxation Gate for event firing threshold using temperature-controlled Sigmoid.
    soft_gate = sigmoid(beta * (delta - threshold))
    During training, beta can be annealed (increased) to sharpen into step function.
    """
    def __init__(self, threshold: float = 0.15, initial_beta: float = 5.0):
        super().__init__()
        self.threshold = threshold
        self.beta = initial_beta

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.beta * (delta - self.threshold))

    def step_temperature(self, factor: float = 1.05):
        """Anneal temperature factor beta to sharpen threshold."""
        self.beta *= factor


class EventDrivenAsyncGNN(nn.Module):
    """
    Asynchronous Event-Driven Graph Neural Network.

    Timeless, event-driven propagation using priority queue without global clock t.
    State updates occur locally only upon event packet reception.

    Key Attributes:
    - Volumetric Structural Tensor (volume_tensor): Node-level structural substrate attributes.
    - Node State Memory (node_states): Dynamic memory per node updated asynchronously.
    - Local Firing Threshold (threshold): Activation constraint for local event spreading.
    """
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        out_channels: int,
        threshold: float = 0.1,
        volumetric_dim: int = 4,
        use_ste: bool = True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.threshold = threshold
        self.use_ste = use_ste

        # 1. Volumetric Structural Substrate Tensor (nn.Parameter)
        self.volume_tensor = nn.Parameter(
            torch.randn(num_nodes, volumetric_dim) * 0.1
        )

        # 2. Dynamic Node States Memory
        self.register_buffer("node_states", torch.zeros(num_nodes, out_channels))

        # 3. Local Message & Update Nets
        self.message_net = nn.Linear(in_channels + volumetric_dim, out_channels)
        self.update_net = nn.GRUCell(out_channels, out_channels)
        self.relaxed_gate = RelaxedEventGate(threshold=threshold)

    def reset_states(self, device: torch.device = None):
        """Reset node states memory."""
        if device is None:
            device = self.volume_tensor.device
        self.node_states = torch.zeros(self.num_nodes, self.update_net.hidden_size, device=device)

    def forward_event_loop(
        self,
        adj_list: dict[int, list[int]],
        initial_events: list[tuple[float, int, torch.Tensor]],
        max_events: int = 1000,
    ) -> tuple[torch.Tensor, int]:
        """
        Asynchronous Event Processing Loop.

        adj_list: {node_id: [neighbor_ids...]}
        initial_events: [(priority, target_node, payload_tensor), ...]
        returns: (updated_node_states, total_processed_events_count)
        """
        device = self.volume_tensor.device
        event_queue = []
        for priority, node_id, payload in initial_events:
            event_pkt = EventPacket(priority=priority, target_node=node_id, payload=payload.to(device))
            heapq.heappush(event_queue, event_pkt)

        processed_events_count = 0

        while event_queue and processed_events_count < max_events:
            pkt = heapq.heappop(event_queue)
            priority = pkt.priority
            current_node = pkt.target_node
            payload = pkt.payload
            processed_events_count += 1

            # A. Couple incoming causal packet with local volumetric substrate
            vol = self.volume_tensor[current_node]
            msg_input = torch.cat([payload, vol], dim=-1)
            message = torch.relu(self.message_net(msg_input))

            # B. Local state transition
            old_state = self.node_states[current_node].clone()
            new_state = self.update_net(
                message.unsqueeze(0), old_state.unsqueeze(0)
            ).squeeze(0)

            # C. Delta computation and threshold gate evaluation
            delta = torch.norm(new_state - old_state)
            self.node_states[current_node] = new_state.detach()

            # D. Event Spreading check
            if self.use_ste:
                gate_fire = ste_event_gate(delta, self.threshold).item() > 0.5
            else:
                gate_fire = self.relaxed_gate(delta).item() > 0.5

            if gate_fire:
                for neighbor in adj_list.get(current_node, []):
                    new_priority = priority + 1.0  # Increment causal priority generation
                    new_pkt = EventPacket(priority=new_priority, target_node=neighbor, payload=new_state.detach())
                    heapq.heappush(event_queue, new_pkt)

        return self.node_states, processed_events_count
