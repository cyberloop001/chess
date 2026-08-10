from .mlp import MLPNet
from .transformer import TransformerNet
from .base import AlphaZeroNet

__all__ = ["MLPNet", "TransformerNet", "AlphaZeroNet", "build_model"]


def build_model(name: str, **kwargs) -> AlphaZeroNet:
    key = name.lower().strip()
    if key in {"mlp", "mlp+mcts", "mlp_mcts"}:
        return MLPNet(**kwargs)
    if key in {"transformer", "transformer+mcts", "transformer_mcts", "vit"}:
        return TransformerNet(**kwargs)
    raise ValueError(f"Unknown model '{name}'. Use 'mlp' or 'transformer'.")
