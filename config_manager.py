# config_manager.py
import json
from pathlib import Path
from typing import TypedDict, Literal


class AppConfig(TypedDict):
    meta: dict
    common: dict
    training: dict
    validation: dict
    labelme_conversion: dict


class ConfigManager:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config: AppConfig = self._load_config()

    def _load_config(self) -> AppConfig:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        assert "training" in config, "Invalid config: missing 'training' section"
        return config

    def save_config(self, new_config: AppConfig):
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(new_config, f, indent=2, ensure_ascii=False)
        self.config = new_config