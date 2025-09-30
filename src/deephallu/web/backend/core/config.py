import os
import os.path as osp
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

HERE = osp.dirname(osp.abspath(__file__))

class DatasetConfig(BaseModel):
    name: str
    path: str
    type: str
    description: str

class ServerConfig(BaseModel):
    host: str = "localhost"
    port: int = 8000
    reload: bool = True

class CacheConfig(BaseModel):
    host: str = "localhost"
    port: int = 637

class ModelsConfig(BaseModel):
    name: str
    type: str
    description: str
    model_name: str

class HuggingFaceModelConfig(ModelsConfig):
    model_name: str
    hf_home: str

class Config:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(HERE).parent / "config.yaml"
        self.config_path = Path(config_path)
        self._config_data = self._load_config()

        self.datasets = {
            key: DatasetConfig(**value)
            for key, value in self._config_data.get("datasets", {}).items()
        }
        self.server = ServerConfig(**self._config_data.get("server", {}))
        self.cache = CacheConfig(**self._config_data.get("cache", {}))
        self.models = {
            key: HuggingFaceModelConfig(**value) if value["type"] == "huggingface" else ModelsConfig(**value)
            for key, value in self._config_data.get("models", {}).items()
        }

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")

    def get_dataset_config(self, dataset_name: str) -> Optional[DatasetConfig]:
        return self.datasets.get(dataset_name)

    def get_all_datasets(self) -> Dict[str, DatasetConfig]:
        return self.datasets

    def validate_dataset_paths(self) -> List[str]:
        valid_paths = []
        invalid_paths = []
        for name, dataset in self.datasets.items():
            if not Path(dataset.path).exists():
                invalid_paths.append(f"{name}: {dataset.path}")
            else:
                valid_paths.append(f"{name}: {dataset.path}")
        return valid_paths, invalid_paths

# Global config instance
config = Config()

if __name__ == "__main__":
    print(config.models)
    print(config.datasets)
    print(config.validate_dataset_paths()[0], config.validate_dataset_paths()[1])