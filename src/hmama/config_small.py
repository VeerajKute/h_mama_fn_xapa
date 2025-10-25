from pydantic import BaseModel


class ModelConfigSmall(BaseModel):
    text_model_name: str = "distilbert-base-uncased"
    clip_model_name: str = "openai/clip-vit-base-patch16"
    whiten_dim: int = 128
    device: str = "cpu"
    cache_dir: str = "./.cache"
    batch_size: int = 4


class TrainConfigSmall(BaseModel):
    batch_size: int = 2
    lr: float = 2e-4
    epochs: int = 1
    seed: int = 42


