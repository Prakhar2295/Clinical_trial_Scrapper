from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    weaviate_url: str = "http://localhost:8080"
    weaviate_api_key: str = ""
    use_weaviate_embedded: bool = False
    embedding_model: str = "jinaai/jina-embeddings-v3"
    embedding_dimensions: int = 1024
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k_retrieval: int = 10
    confidence_threshold: float = 0.85
    max_pages_initial: int = 3
    max_pages_fallback: int = 8
    pdf_input_dir: str = "../clinical_trial_pdfs"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# Chunking is handled by MarkdownTextSplitter in chunker.py
# chunk_size=8000 chars (~2000 tokens), overlap=800 chars (~200 tokens)
settings = Settings()
