"""Named strategy profiles stored as local JSON."""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict

from core.config_manager import config_mgr


class ProfileManager:
    def __init__(self, directory: Path | None = None):
        self.directory = directory or Path(__file__).resolve().parent.parent / "profiles"
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _slug(name: str) -> str:
        slug = re.sub(r"[^a-z0-9_-]+", "-", name.strip().lower()).strip("-")
        if not slug:
            raise ValueError("Profile name must contain letters or numbers")
        return slug

    def list(self) -> list[Dict[str, Any]]:
        profiles = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                profiles.append({"id": path.stem, "name": data.get("name", path.stem)})
            except (OSError, json.JSONDecodeError):
                continue
        return profiles

    def save(self, name: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        profile_id = self._slug(name)
        payload = {"name": name.strip(), "settings": settings}
        path = self.directory / f"{profile_id}.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temp, path)
        return {"id": profile_id, "name": payload["name"]}

    def load(self, profile_id: str, apply: bool = False) -> Dict[str, Any]:
        path = self.directory / f"{self._slug(profile_id)}.json"
        if not path.exists():
            raise FileNotFoundError(profile_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if apply:
            config_mgr.update(payload["settings"])
        return payload

    def delete(self, profile_id: str):
        path = self.directory / f"{self._slug(profile_id)}.json"
        if not path.exists():
            raise FileNotFoundError(profile_id)
        path.unlink()


profile_mgr = ProfileManager()
