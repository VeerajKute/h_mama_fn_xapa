from pydantic import BaseModel

class ModelConfig(BaseModel):
    text_model_name: str = "bert-base-uncased"
    clip_model_name: str = "openai/clip-vit-base-patch32"
    max_len: int = 160
    whiten_dim: int = 256
    device: str = "cpu"

class TrainConfig(BaseModel):
    batch_size: int = 2
    lr: float = 2e-4
    epochs: int = 1
    seed: int = 42
