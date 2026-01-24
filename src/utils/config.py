import os
from pathlib import Path
from typing import List, Literal, Optional
from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).parent.parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"

class BaseConfigSettings(BaseSettings):
    """Base configuration for all settings"""
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        extra="ignore",
        frozen=True,
        env_nested_delimiter="__",
        case_sensitive=False,
    )

# Type definitions
FMOC_DocType = Literal["minutes", "statement", "presconf", "sep", "implementation"]

class FOMCSettings(BaseConfigSettings): 
    """FOMC document download settings"""
    base_url: str = "https://www.federalreserve.gov/monetarypolicy"
    start_year: int = 2020
    end_year: int = 2025
    output_dir: Path = Path("data/raw/")
    timeout: int = 10 
    target_docs: List[FMOC_DocType] = [
        "minutes", 
        "statement", 
        "presconf", 
        "sep", 
        "implementation"
    ]
    model_config = SettingsConfigDict(
        env_prefix="FOMC__", 
        env_file=[".env", str(ENV_FILE_PATH)], 
        extra="ignore",
        frozen=True
    )

class NoteSettings(BaseConfigSettings): 
    """Implementation notes download settings"""
    base_url: str = "https://www.federalreserve.gov/newsevents/pressreleases"
    output_dir: Path = Path("data/raw/")
    
    model_config = SettingsConfigDict(
        env_prefix="NOTE__", 
        env_file=[".env", str(ENV_FILE_PATH)], 
        extra="ignore",
        frozen=True
    )

class ProcessingSettings(BaseConfigSettings):
    """PDF processing and extraction settings"""
    # Chunking parameters
    chunk_size: int = Field(default=512, ge=100, le=2048)
    chunk_overlap: int = Field(default=50, ge=0, le=200)
    min_chunk_size: int = Field(default=100, ge=50, le=500)
    max_chunk_size: int = Field(default=1024, ge=512, le=4096)
    strategy: Literal["section_aware", "fixed_size"] = "section_aware"

    # Quality thresholds
    quality_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    # Table handling
    preserve_large_tables: bool = True
    small_table_threshold: int = Field(default=150, ge=50, le=500)
    multimodal_chunks: bool = True

    # Image processing
    pdf_dpi: int = Field(default=300, ge=150, le=600)
    image_max_size: int = Field(default=1600, ge=800, le=3200)
    image_quality: int = Field(default=95, ge=80, le=100)
    extract_figures: bool = True

    # VLM API settings (Qwen)
    vlm_api_url: str = Field(
        default="http://localhost:8001/v1",
        description="Qwen VLM API endpoint (VastAI via SSH tunnel or direct)"
    )
    vlm_model: str = "Qwen/Qwen2-VL-72B-Instruct-AWQ"
    vlm_api_key: str = Field(default="production-key")
    vlm_max_retries: int = Field(default=3, ge=1, le=10)
    vlm_temperature: float = Field(default=0.01, ge=0.0, le=1.0)
    vlm_max_tokens: int = Field(default=4096, ge=1024, le=8192)

    # Parallel processing
    max_workers: int = Field(default=4, ge=1, le=16)
    enable_parallel: bool = False

    # Surya remote processing (optional)
    surya_use_remote: bool = Field(
        default=False,
        description="Use remote Surya API instead of local models"
    )
    surya_api_url: str = Field(
        default="http://localhost:8002",
        description="Surya API endpoint (VastAI via SSH tunnel or direct)"
    )
    surya_api_key: str = Field(default="production-key")

    model_config = SettingsConfigDict(
        env_prefix="PROCESSING__",
        env_file=[".env", str(ENV_FILE_PATH)],
        extra="ignore",
        frozen=True
    )

class VectorDBSettings(BaseConfigSettings):
    """Vector database configuration"""
    persist_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("VECTOR_DB_DIR", "./data/vector_db"))
    )
    collection_name: str = "fomc_documents"
    
    # Embedding settings (model is defined separately)
    embedding_model: str = "BAAI/bge-small-en-v1.5"  # Reference for embedder
    embedding_dim: int = Field(default=384, ge=128, le=2048)
    
    # Distance metric
    distance_metric: Literal["cosine", "l2", "ip"] = "cosine"
    
    # Batch settings
    batch_size: int = Field(default=500, ge=100, le=2000)
    
    # Index settings
    hnsw_ef: int = Field(default=200, ge=50, le=500)
    hnsw_m: int = Field(default=16, ge=4, le=64)
    
    model_config = SettingsConfigDict(
        env_prefix="VECTORDB__",
        env_file=[".env", str(ENV_FILE_PATH)],
        extra="ignore",
        frozen=True
    )

class EmbeddingSettings(BaseConfigSettings):
    """Embedding model configuration"""
    model_name: str = "BAAI/bge-small-en-v1.5"
    device: Optional[str] = None  # None = auto-detect
    batch_size: int = Field(default=32, ge=8, le=128)
    normalize_embeddings: bool = True
    
    # Instruction prefixes for retrieval models
    query_instruction: str = "Represent this sentence for searching relevant passages: "
    doc_instruction: str = ""
    
    # Model-specific settings
    max_seq_length: int = Field(default=512, ge=128, le=2048)
    
    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING__",
        env_file=[".env", str(ENV_FILE_PATH)],
        extra="ignore",
        frozen=True
    )

class LoggingConfig(BaseConfigSettings):
    """Logging configuration"""
    level: str = "INFO" 
    log_file: Optional[Path] = Path("./logs/fomc_rag.log")
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    include_console: bool = True

    model_config = SettingsConfigDict(
        env_prefix="LOGGING__",
        env_file=[".env", str(ENV_FILE_PATH)],
        extra="ignore",
        frozen=True
    )

class DataSettings(BaseConfigSettings):
    """Data directories configuration"""
    data_dir: Path = Field(default_factory=lambda: Path(os.getenv("DATA_DIR", "./data")))
    raw_data_dir: Path = Field(default_factory=lambda: Path(os.getenv("RAW_DATA_DIR", "./data/raw")))
    processed_data_dir: Path = Field(default_factory=lambda: Path(os.getenv("PROCESSED_DATA_DIR", "./data/processed")))
    test_set_dir: Path = Field(default_factory=lambda: Path(os.getenv("TEST_SET_DIR", "./data/test_set")))

    model_config = SettingsConfigDict(
        env_prefix="DATA__",
        env_file=[".env", str(ENV_FILE_PATH)],
        extra="ignore",
        frozen=True
    )

class LLMSettings(BaseConfigSettings):
    """LLM inference settings for Llama-3 via vLLM"""
    # Remote vLLM API settings
    api_base_url: str = Field(
        default="http://localhost:8000/v1",
        description="Llama API endpoint (VastAI via SSH tunnel or direct)"
    )
    api_key: str = Field(
        default="EMPTY",
        description="API key for authentication (can be dummy for vLLM)"
    )
    model_name: str = Field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        description="Model identifier on vLLM server"
    )

    # Generation parameters
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=50, le=4096)

    # Entropy/Uncertainty settings
    enable_entropy: bool = Field(default=True, description="Enable entropy calculation")
    entropy_expansion_threshold: float = Field(
        default=1.5,
        ge=0.5,
        le=3.0,
        description="Entropy threshold for retrieval expansion"
    )
    entropy_abstention_threshold: float = Field(
        default=2.0,
        ge=1.0,
        le=4.0,
        description="Entropy threshold for abstention"
    )

    # Prompt settings
    system_prompt: str = Field(
        default="""You are a precise question-answering assistant for Federal Reserve FOMC documents.

Your core principles:
1. Answer ONLY based on provided context
2. Always cite your sources with [Document N] format
3. If uncertain or information is missing, say "INSUFFICIENT INFORMATION"
4. Never make up or infer information not explicitly stated
5. Be concise and direct in your responses

Remember: Accuracy and citation precision are paramount. It's better to say "I don't know" than to guess."""
    )

    model_config = SettingsConfigDict(
        env_prefix="LLM__",
        env_file=[".env", str(ENV_FILE_PATH)],
        extra="ignore",
        frozen=True
    )

class RetrievalSettings(BaseConfigSettings):
    """Retrieval and RAG settings"""
    top_k: int = Field(default=5, ge=1, le=50)
    expanded_top_k: int = Field(default=10, ge=5, le=50, description="Top-k for expanded retrieval")
    rerank: bool = False
    rerank_top_k: int = Field(default=10, ge=5, le=100)
    max_context_length: int = Field(default=4000, ge=1000, le=32000)

    model_config = SettingsConfigDict(
        env_prefix="RETRIEVAL__",
        env_file=[".env", str(ENV_FILE_PATH)],
        extra="ignore",
        frozen=True
    )

class AlignmentSettings(BaseConfigSettings):
    """RLAIF and DPO training settings"""
    # Teacher model (for grading)
    teacher_model: str = Field(
        default="gpt-4-turbo",
        description="Teacher model for RLAIF grading"
    )
    teacher_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for teacher model"
    )
    teacher_temperature: float = Field(default=0.1, ge=0.0, le=1.0)

    # Grading thresholds
    win_threshold: float = Field(
        default=7.0,
        ge=0.0,
        le=10.0,
        description="Score threshold for WIN label"
    )

    # Grading weights
    factual_grounding_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    citation_accuracy_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    abstention_quality_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    hallucination_penalty_weight: float = Field(default=0.1, ge=0.0, le=1.0)

    # Preference pair generation
    pairs_output_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("PAIRS_DIR", "./data/alignment/preference_pairs"))
    )
    min_score_gap: float = Field(
        default=0.5,
        description="Minimum score difference between chosen/rejected"
    )

    # DPO training hyperparameters
    dpo_beta: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="KL penalty coefficient for DPO"
    )
    dpo_learning_rate: float = Field(
        default=5e-7,
        ge=1e-8,
        le=1e-5,
        description="Learning rate for DPO training"
    )
    dpo_num_epochs: int = Field(default=3, ge=1, le=10)
    dpo_batch_size: int = Field(default=4, ge=1, le=16)
    dpo_gradient_accumulation_steps: int = Field(default=4, ge=1, le=16)

    # LoRA configuration
    use_lora: bool = Field(default=True, description="Use LoRA for efficient fine-tuning")
    lora_r: int = Field(default=16, ge=4, le=64, description="LoRA rank")
    lora_alpha: int = Field(default=32, ge=8, le=128, description="LoRA alpha")
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.2)
    lora_target_modules: List[str] = Field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # Quantization
    use_4bit: bool = Field(default=True, description="Use 4-bit quantization")
    bnb_4bit_compute_dtype: str = Field(default="bfloat16")

    # Model paths
    student_model_name: str = Field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        description="Student model for DPO training"
    )
    aligned_model_output_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("ALIGNED_MODEL_DIR", "./models/llama3_dpo_aligned"))
    )

    # Training control
    dpo_max_length: int = Field(default=512, ge=128, le=2048)
    dpo_max_prompt_length: int = Field(default=256, ge=64, le=1024)
    dpo_warmup_steps: int = Field(default=100, ge=0, le=500)
    dpo_save_steps: int = Field(default=100, ge=10, le=1000)
    dpo_logging_steps: int = Field(default=10, ge=1, le=100)
    dpo_fp16: bool = False
    dpo_bf16: bool = True

    model_config = SettingsConfigDict(
        env_prefix="ALIGNMENT__",
        env_file=[".env", str(ENV_FILE_PATH)],
        extra="ignore",
        frozen=True
    )

class EvaluationSettings(BaseConfigSettings):
    """Evaluation and metrics settings"""
    # Output directories
    eval_output_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("EVAL_DIR", "./data/evaluation"))
    )

    # Metric thresholds
    accuracy_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Threshold for considering answer correct"
    )

    # Test set configuration
    default_test_set: Path = Field(
        default_factory=lambda: Path("./data/test_set/sample_test_questions.json")
    )

    # Evaluation modes
    compute_retrieval_metrics: bool = True
    compute_answer_metrics: bool = True
    compute_uncertainty_metrics: bool = True

    # Uncertainty calibration
    ece_bins: int = Field(default=10, ge=5, le=20, description="Bins for ECE calculation")

    # Report settings
    save_detailed_results: bool = True
    save_per_question_metrics: bool = True

    model_config = SettingsConfigDict(
        env_prefix="EVALUATION__",
        env_file=[".env", str(ENV_FILE_PATH)],
        extra="ignore",
        frozen=True
    )

class VastAISettings(BaseConfigSettings):
    """VastAI SSH Connection Configuration

    This section documents SSH connection parameters for VastAI instances.
    For manual SSH tunneling, use these values to establish connections.

    Deployment Modes:
    1. Local SSH Tunnels: Use localhost URLs (PROCESSING__VLM_API_URL=http://localhost:8001/v1)
    2. Direct VastAI: Use instance IPs (PROCESSING__VLM_API_URL=http://<vastai-ip>:8001/v1)
    3. Hybrid: Mix of tunneled and direct connections
    """

    # Deployment mode documentation
    deployment_mode: Literal["local_tunnel", "direct_vastai", "hybrid"] = Field(
        default="local_tunnel",
        description="Deployment strategy: local_tunnel, direct_vastai, or hybrid"
    )

    # Qwen Model Instance
    qwen_host: str = Field(
        default="localhost",
        description="VastAI host for Qwen (e.g., 'ssh.vast.ai' or IP address)"
    )
    qwen_ssh_port: int = Field(
        default=22,
        description="SSH port for Qwen instance"
    )
    qwen_api_port: int = Field(
        default=8001,
        description="API port where Qwen vLLM server runs"
    )
    qwen_ssh_user: str = Field(default="root")

    # Llama Model Instance
    llama_host: str = Field(
        default="localhost",
        description="VastAI host for Llama (e.g., 'ssh.vast.ai' or IP address)"
    )
    llama_ssh_port: int = Field(
        default=22,
        description="SSH port for Llama instance"
    )
    llama_api_port: int = Field(
        default=8000,
        description="API port where Llama vLLM server runs"
    )
    llama_ssh_user: str = Field(default="root")

    # Surya Model Instance (Optional - for remote Surya processing)
    surya_host: str = Field(
        default="localhost",
        description="VastAI host for Surya (optional, for remote layout detection)"
    )
    surya_ssh_port: int = Field(
        default=22,
        description="SSH port for Surya instance"
    )
    surya_api_port: int = Field(
        default=8002,
        description="API port where Surya API server runs"
    )
    surya_ssh_user: str = Field(default="root")

    # SSH Authentication
    ssh_key_path: Optional[Path] = Field(
        default=None,
        description="Path to SSH private key (e.g., ~/.ssh/vastai_rsa)"
    )

    model_config = SettingsConfigDict(
        env_prefix="VASTAI__",
        env_file=[".env", str(ENV_FILE_PATH)],
        extra="ignore",
        frozen=True
    )

class Config(BaseModel):
    """Main configuration object"""
    fomc_downloader: FOMCSettings = Field(default_factory=FOMCSettings)
    note_downloader: NoteSettings = Field(default_factory=NoteSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    vector_db: VectorDBSettings = Field(default_factory=VectorDBSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    data: DataSettings = Field(default_factory=DataSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    alignment: AlignmentSettings = Field(default_factory=AlignmentSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    vastai: VastAISettings = Field(default_factory=VastAISettings)

    def validate_paths(self):
        """Create all necessary directories"""
        paths_to_create = [
            self.data.data_dir,
            self.data.raw_data_dir,
            self.data.processed_data_dir,
            self.data.test_set_dir,
            self.vector_db.persist_dir,
            self.alignment.pairs_output_dir,
            self.alignment.aligned_model_output_dir,
            self.evaluation.eval_output_dir
        ]

        if self.logging.log_file:
            paths_to_create.append(self.logging.log_file.parent)

        for path in paths_to_create:
            path.mkdir(parents=True, exist_ok=True)

# Global config instance
config = Config()

# Validate and create directories on import
config.validate_paths()

if __name__ == "__main__":
    print("=" * 60)
    print("FOMC RAG Configuration")
    print("=" * 60)
    print(f"\n📁 Data Directories:")
    print(f"  Raw: {config.data.raw_data_dir}")
    print(f"  Processed: {config.data.processed_data_dir}")
    print(f"  Vector DB: {config.vector_db.persist_dir}")

    print(f"\n⚙️  Processing Settings:")
    print(f"  Chunk Size: {config.processing.chunk_size}")
    print(f"  Strategy: {config.processing.strategy}")
    print(f"  VLM Model: {config.processing.vlm_model}")

    print(f"\n🔍 Embedding Settings:")
    print(f"  Model: {config.embedding.model_name}")
    print(f"  Device: {config.embedding.device or 'auto-detect'}")

    print(f"\n📊 Vector DB Settings:")
    print(f"  Collection: {config.vector_db.collection_name}")
    print(f"  Distance: {config.vector_db.distance_metric}")
    print(f"  Batch Size: {config.vector_db.batch_size}")

    print(f"\n🤖 LLM Settings:")
    print(f"  API Base: {config.llm.api_base_url}")
    print(f"  Model: {config.llm.model_name}")
    print(f"  Entropy Enabled: {config.llm.enable_entropy}")
    print(f"  Expansion Threshold: {config.llm.entropy_expansion_threshold}")
    print(f"  Abstention Threshold: {config.llm.entropy_abstention_threshold}")

    print(f"\n🔎 Retrieval Settings:")
    print(f"  Initial Top-K: {config.retrieval.top_k}")
    print(f"  Expanded Top-K: {config.retrieval.expanded_top_k}")