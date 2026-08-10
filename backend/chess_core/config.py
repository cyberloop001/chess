from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # Shared AlphaZero-style heads
    board_planes: int = 19
    action_size: int = 4672
    value_size: int = 1

    # MLP
    mlp_hidden: tuple[int, ...] = (1024, 512, 256)

    # Transformer
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    ffn_dim: int = 512
    dropout: float = 0.1


@dataclass
class MCTSConfig:
    simulations: int = 96
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 0.0  # 0 = greedy argmax of visit counts


@dataclass
class GameConfig:
    white_model: str = "mlp"  # "mlp" | "transformer"
    black_model: str = "transformer"
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    move_delay_ms: int = 400
    max_moves: int = 300
