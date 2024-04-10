from modules.inference.engine import InferEngine
from utils.config import ModelConfig

from multiprocessing.queues import Queue
from multiprocessing import Process

from summary_app.schema.config import AppConfig


class SummaryExtractor:
    _instance = None

    def __init__(self, app_config: AppConfig) -> None:
        self.app_config = app_config
        self.engine = InferEngine(configs=ModelConfig(**app_config.model),
                                  vocab_file=app_config.vocab_file,
                                  glove_file=app_config.glove_file,
                                  embedding_size=app_config.embedding_size,
                                  pretrained_model=app_config.pretrained_model_path,
                                  max_sentences=app_config.max_length,
                                  )
        # self.extractor_queue = Queue()
        self.process = None

    @classmethod
    def init_instance(cls,  app_config: AppConfig):
        if cls._instance is None:
            cls._instance = SummaryExtractor(app_config)

    @classmethod
    def get_instance(cls, app_config: AppConfig = None):
        if cls._instance is None:
            cls._instance = SummaryExtractor(app_config)
        return cls._instance

    def set_config(self,
                   max_length: int = None):
        self.app_config.max_length = max_length

    def get_result(self, content: str):
        summary_content = self.engine.summarize(content=content,
                                                max_num_of_sentences=self.app_config.max_length)
        return summary_content
