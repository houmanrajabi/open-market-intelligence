import os
from pathlib import Path
from typing import List, Literal, Optional
from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict



PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"

class BaseConfigSettings(BaseSettings):
    """
    Base configuration that defines WHERE to look for settings (.env location)
    """
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        # 2. Safety Valve
        # "ignore" means: If the .env file contains variables NOT defined in this class
        # (e.g., AWS_KEYS meant for another part of the app), do NOT raise an error.
        extra="ignore",
        # 3. Immutability
        # If True, you cannot change values in python code (e.g., config.timeout = 50).
        frozen=True,
        env_nested_delimiter="__",
        case_sensitive=False,
    )


# 1. Move the Type definition OUT HERE
FMOC_DocType = Literal["minutes", "statement", "presconf", "sep", "implementation"]

class FOMCSettings(BaseConfigSettings): 
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
    base_url: str = "https://www.federalreserve.gov/newsevents/pressreleases"
    output_dir: Path = Path("data/raw/")
    
    model_config = SettingsConfigDict(
        env_prefix="NOTE__", 
        env_file=[".env", str(ENV_FILE_PATH)], 
        extra="ignore",
        frozen=True
    )

class LoggingConfig(BaseConfigSettings):
    level: str = "INFO" 
    log_file: Optional[Path] = Path("./logs/fomc_rag.log")

    model_config = {
        **BaseConfigSettings.model_config,
        "env_prefix": "LOGGING__",  
    }

class DataSettings(BaseConfigSettings):
    """Data directories configuration"""
    data_dir: Path = Field(default_factory=lambda: Path(os.getenv("DATA_DIR", "./data")))
    raw_data_dir: Path = Field(default_factory=lambda: Path(os.getenv("RAW_DATA_DIR", "./data/raw")))
    processed_data_dir: Path = Field(default_factory=lambda: Path(os.getenv("PROCESSED_DATA_DIR", "./data/processed")))
    test_set_dir: Path = Field(default_factory=lambda: Path(os.getenv("TEST_SET_DIR", "./data/test_set")))

    model_config = {
        **BaseConfigSettings.model_config,
        "env_prefix": "DATA__",  # Allows env overrides like DATA__RAW_DATA_DIR
    }

class Config(BaseModel):
    fomc_downloader: FOMCSettings = Field(default_factory=FOMCSettings)
    note_downloader: NoteSettings = Field(default_factory=NoteSettings)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    data: DataSettings = Field(default_factory=DataSettings)




# Global config instance
config = Config()
if __name__ == "__main__":
    print(config.fomc_downloader.base_url)
    print(config.logging.level)