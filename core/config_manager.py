import json
import os
import logging
from pathlib import Path
from typing import Any, Dict

import config as static_config

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Dynamic configuration manager.
    Loads settings from settings.json, falling back to static config.py variables.
    """
    def __init__(self, settings_file: str = "settings.json"):
        project_root = Path(__file__).resolve().parent.parent
        path = Path(settings_file)
        self.settings_file = path if path.is_absolute() else project_root / path
        self.settings: Dict[str, Any] = {}
        self.load()

    def load(self):
        """Load settings from JSON file if it exists."""
        if self.settings_file.exists():
            try:
                with self.settings_file.open("r", encoding="utf-8") as f:
                    self.settings = json.load(f)
                logger.debug(f"Loaded {len(self.settings)} dynamic settings from {self.settings_file}")
            except Exception as e:
                logger.error(f"Failed to load settings.json: {e}")
        else:
            self.settings = {}

    def save(self):
        """Save current dynamic settings to JSON file."""
        try:
            self.settings_file.parent.mkdir(parents=True, exist_ok=True)
            temp_file = self.settings_file.with_suffix(".tmp")
            with temp_file.open("w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=4)
            os.replace(temp_file, self.settings_file)
        except Exception as e:
            logger.error(f"Failed to save settings.json: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        Priority:
        1. settings.json (dynamic)
        2. config.py (static)
        3. default parameter
        """
        if key in self.settings:
            return self.settings[key]
        if hasattr(static_config, key):
            return getattr(static_config, key)
        return default

    def set(self, key: str, value: Any):
        """Set a dynamic configuration value."""
        self.settings[key] = value
        self.save()

    def update(self, values: Dict[str, Any]):
        """Apply and persist multiple values as one write."""
        self.settings.update(values)
        self.save()

    def reset(self):
        """Remove all overrides and return to config.py defaults."""
        self.settings = {}
        self.save()
        
    def get_all(self) -> Dict[str, Any]:
        """Return a dictionary of all relevant settings for UI."""
        # Start with a dictionary of all uppercase variables in config.py
        all_settings = {
            k: v for k, v in vars(static_config).items() 
            if k.isupper() and not k.startswith("ALPACA_") and not k.startswith("DATABASE_")
        }
        # Override with dynamic settings
        all_settings.update(self.settings)
        return all_settings

# Global singleton
config_mgr = ConfigManager()
