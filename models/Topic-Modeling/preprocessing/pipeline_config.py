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
    diagnostics_output: str | None = None
    diagnostics_top_n: int = 25
    enable_ngrams: bool = False
    ngram_min_count: int = 15
    ngram_threshold: float = 10.0
    max_doc_count: int = 1000

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
        if self.diagnostics_top_n < 1:
            raise ValueError("diagnostics_top_n must be >= 1")
        if self.ngram_min_count < 1:
            raise ValueError("ngram_min_count must be >= 1")
        if self.ngram_threshold <= 0:
            raise ValueError("ngram_threshold must be > 0")
        if self.max_doc_count <= 0:
            raise ValueError("max_doc_count must be > 0")


DEFAULT_CONFIG = PipelineConfig()
