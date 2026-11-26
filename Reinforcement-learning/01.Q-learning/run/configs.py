import json
from pathlib import Path


CONFIG = json.loads(Path(__file__).with_suffix(".json").read_text(encoding="utf-8"))
