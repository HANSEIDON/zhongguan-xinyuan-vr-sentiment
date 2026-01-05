from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_HOME = PROJECT_ROOT / "models"
BACKEND_HOME = PROJECT_ROOT / "backend"

USE_PRETRAINED = "hfl/chinese-roberta-wwm-ext"
USE_LOCAL_MODEL = "roberta"
USE_PT_NAME = "best.pt"
