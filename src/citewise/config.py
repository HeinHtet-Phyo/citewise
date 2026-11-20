"""Central configuration, read from environment variables."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime settings read from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM keys
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    nebius_api_key: str = Field(default="", alias="NEBIUS_API_KEY")
    nebius_base_url: str = Field(
        default="https://api.tokenfactory.nebius.com/v1",
        alias="NEBIUS_BASE_URL",
    )

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")
    qdrant_local_url: str = Field(default="http://localhost:6333", alias="QDRANT_LOCAL_URL")

    # Langfuse (optional tracing)
    langfuse_public_key: str = Field(default="", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        alias="LANGFUSE_HOST",
    )

    # Hugging Face
    hf_token: str = Field(default="", alias="HF_TOKEN")

    # Public demo
    default_config: str = Field(default="L2_dense", alias="CITEWISE_CONFIG")
    daily_budget_gbp: float = Field(default=0.50, alias="DAILY_BUDGET_GBP")
    rate_limit_per_minute: int = Field(default=10, alias="RATE_LIMIT_PER_MINUTE")

    # LLM cache
    llm_cache_enabled: bool = Field(default=True, alias="LLM_CACHE")
    llm_cache_dir: str = Field(default=".llm_cache", alias="LLM_CACHE_DIR")

    # GOV.UK ingestion
    govuk_user_agent: str = Field(
        default="citewise-portfolio-project (github.com/HeinHtet-Phyo/citewise)",
        alias="GOVUK_USER_AGENT",
    )
    govuk_rate_limit_rps: float = Field(default=5.0, alias="GOVUK_RATE_LIMIT_RPS")

    @property
    def tracing_enabled(self) -> bool:
        """True when all Langfuse keys are present."""
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


class RagConfig:
    """Configuration for one experiment / pipeline run."""

    def __init__(
        self,
        config_name: str = "default",
        chunk_size: int = 400,
        chunk_overlap: int = 60,
        collection_name: str = "citewise_400",
        search_mode: str = "hybrid",  # "dense" | "hybrid"
        top_k_retrieve: int = 30,
        use_reranker: bool = True,
        top_k_context: int = 6,
        refusal_threshold: float = 0.0,  # tuned on dev split
        llm_model: str = "openai/Qwen/Qwen3-235B-A22B-Instruct-2507",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        prompt_version: str = "v1",  # "v1" | "v2" | "v3"
        verify_claims: bool = False,
        expand_context: bool = False,
        context_token_budget: int = 900,
    ) -> None:
        self.config_name = config_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name
        self.search_mode = search_mode
        self.top_k_retrieve = top_k_retrieve
        self.use_reranker = use_reranker
        self.top_k_context = top_k_context
        self.refusal_threshold = refusal_threshold
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_version = prompt_version
        self.verify_claims = verify_claims
        self.expand_context = expand_context
        self.context_token_budget = context_token_budget


# Module-level singleton — import and use directly
settings = Settings()
