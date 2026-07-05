import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CATEGORIES_FILE = BASE_DIR / "categories.json"

DEFAULT_CATEGORIES = ["Protocol", "SAP", "ICF", "CSR", "IB", "Combined"]


class CategoryRegistry:
    """In-memory + file-persisted registry of valid document categories."""

    def __init__(self) -> None:
        self._categories: list[str] = list(DEFAULT_CATEGORIES)

        if CATEGORIES_FILE.exists():
            try:
                loaded = json.loads(CATEGORIES_FILE.read_text())
            except (json.JSONDecodeError, OSError):
                loaded = []

            existing_lower = {c.lower() for c in self._categories}
            for name in loaded:
                if isinstance(name, str) and name.lower() not in existing_lower:
                    self._categories.append(name)
                    existing_lower.add(name.lower())

        self._save()

    def add(self, name: str) -> str:
        name = name.strip()
        if not name:
            raise ValueError("Category name cannot be empty")
        if self.exists(name):
            raise ValueError(f"Category '{name}' already exists")

        formatted = name[0].upper() + name[1:]
        self._categories.append(formatted)
        self._save()
        return formatted

    def all(self) -> list[str]:
        return sorted(self._categories)

    def exists(self, name: str) -> bool:
        target = name.strip().lower()
        return any(c.lower() == target for c in self._categories)

    def _save(self) -> None:
        CATEGORIES_FILE.write_text(json.dumps(self._categories, indent=2))


category_registry = CategoryRegistry()
