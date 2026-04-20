"""Minimal backend integration layer for canonical document processing."""

from .document_builder import (
    build_processed_documents,
    build_stub_sentiment_output,
    build_stub_topic_output,
)
from .preprocessing import clean_documents, clean_text
from .schemas import (
    CLEAN_DOCUMENT_COLUMNS,
    PROCESSED_DOCUMENT_COLUMNS,
    RAW_DOCUMENT_COLUMNS,
    CleanDocument,
    ProcessedDocument,
    RawDocument,
)
from .storage import (
    build_sample_processed_documents,
    read_processed_documents,
    resolve_storage_backend,
    write_processed_documents,
)

__all__ = [
    "RAW_DOCUMENT_COLUMNS",
    "CLEAN_DOCUMENT_COLUMNS",
    "PROCESSED_DOCUMENT_COLUMNS",
    "RawDocument",
    "CleanDocument",
    "ProcessedDocument",
    "clean_text",
    "clean_documents",
    "build_processed_documents",
    "build_stub_sentiment_output",
    "build_stub_topic_output",
    "read_processed_documents",
    "write_processed_documents",
    "build_sample_processed_documents",
    "resolve_storage_backend",
]
