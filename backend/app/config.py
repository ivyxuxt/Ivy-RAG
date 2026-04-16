import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Mistral AI
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
    MISTRAL_EMBED_MODEL: str = os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")
    MISTRAL_CHAT_MODEL: str = os.getenv("MISTRAL_CHAT_MODEL", "mistral-small-latest")

    # Retrieval
    TOP_K: int = int(os.getenv("TOP_K", "5"))
    DENSE_TOPK: int = int(os.getenv("DENSE_TOPK", "50"))
    BM25_TOPK: int = int(os.getenv("BM25_TOPK", "50"))
    RRF_K: int = int(os.getenv("RRF_K", "60"))
    MMR_LAMBDA: float = float(os.getenv("MMR_LAMBDA", "0.7"))
    SIMILARITY_THRESHOLD_TOP1: float = float(os.getenv("SIMILARITY_THRESHOLD_TOP1", "0.35"))
    SIMILARITY_THRESHOLD_AVG3: float = float(os.getenv("SIMILARITY_THRESHOLD_AVG3", "0.28"))
    SENTENCE_EVIDENCE_THRESHOLD: float = float(os.getenv("SENTENCE_EVIDENCE_THRESHOLD", "0.50"))

    # Chunking
    CHUNK_TARGET_TOKENS: int = int(os.getenv("CHUNK_TARGET_TOKENS", "500"))
    CHUNK_OVERLAP_TOKENS: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "75"))
    MIN_CHUNK_CHARS: int = int(os.getenv("MIN_CHUNK_CHARS", "300"))

    # BM25
    BM25_K1: float = float(os.getenv("BM25_K1", "1.5"))
    BM25_B: float = float(os.getenv("BM25_B", "0.75"))

    # LLM
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    # Upload
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "25"))
    MAX_UPLOAD_FILES: int = int(os.getenv("MAX_UPLOAD_FILES", "20"))

    # CORS
    FRONTEND_ORIGIN: str = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")

    # Data paths
    DATA_DIR: str = os.path.join(os.path.dirname(__file__), "..", "data")

    @property
    def chunks_path(self) -> str:
        return os.path.join(self.DATA_DIR, "chunks.jsonl")

    @property
    def embeddings_path(self) -> str:
        return os.path.join(self.DATA_DIR, "embeddings.npy")

    @property
    def bm25_path(self) -> str:
        return os.path.join(self.DATA_DIR, "bm25.pkl")

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.DATA_DIR, "raw")


settings = Settings()
