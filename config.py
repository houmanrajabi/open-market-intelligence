import os
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"

class BaseConfigSettings(BaseSettings):
    """
    Base configuration that defines WHERE to look for settings (.env location)
    and HOW to behave (frozen, ignore extras).
    """
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        extra="ignore",
        frozen=True,
        env_nested_delimiter="__",
        case_sensitive=False,
    )

FMOC_DocType = Literal["minutes", "statement", "presconf", "sep", "implementation"]
class FOMCSettings(BaseConfigSettings): 
    base_url: str = "https://www.federalreserve.gov/monetarypolicy"
    start_year: int = 2020
    end_year: int = 2025
    output_dir: Path = Path("fomc_docs")
    timeout: int = 10 
   
    target_docs: List[FMOC_DocType] = [
        "minutes", 
        "statement", 
        "presconf", 
        "sep", 
        "implementation"
    ]
    # Best Practice: Extend the existing config
    model_config = SettingsConfigDict(
        env_prefix="FOMC__", 
        env_file=[".env", str(ENV_FILE_PATH)], 
        extra="ignore",
        frozen=True
    )

settings = FOMCSettings()