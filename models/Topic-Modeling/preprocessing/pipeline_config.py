from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    min_token_len: int = 3
    max_token_len: int = 40
    min_freq: int = 5
    min_doc_freq: int = 2
    batch_size: int = 500
    n_process: int = 4
    oov_placeholder: str | None = None

    def validate(self) -> None:
        if self.min_token_len < 1:
            raise ValueError("min_token_len must be >= 1")
        if self.max_token_len < self.min_token_len:
            raise ValueError("max_token_len must be >= min_token_len")
        if self.min_freq < 1:
            raise ValueError("min_freq must be >= 1")
        if self.min_doc_freq < 1:
            raise ValueError("min_doc_freq must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.n_process < 1:
            raise ValueError("n_process must be >= 1")


DEFAULT_CONFIG = PipelineConfig()
