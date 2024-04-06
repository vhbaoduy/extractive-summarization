from pydantic import BaseModel

# ModelCongif


class AppConfig(BaseModel):
    model: dict
    vocab_file: str
    glove_file: str
    embedding_size: int

    pretrained_model_path: str
    max_length: int = 10
