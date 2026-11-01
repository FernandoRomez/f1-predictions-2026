"""Configuración congelada y versionada del modelo.

Todos los hiperparámetros viven en ``config.json`` y se cargan aquí como
constantes de módulo. El protocolo de validación ciego exige que:
  * se ajusten usando SOLO ``tuned_on`` (2023-2024),
  * ``holdout`` (2025-2026) no se use jamás para ajustar,
  * cualquier cambio quede registrado subiendo ``version``.

Los valores se exponen con el mismo nombre que antes (``PARAMS``, ``CHAOS``...)
para no romper la API pública, y además quedan disponibles como un único dict
``CONFIG`` para reproducibilidad (ver :func:`snapshot`).
"""
import json
from pathlib import Path

import numpy as np

_CONFIG_PATH = Path(__file__).with_name("config.json")


def load() -> dict:
    """Carga el JSON de configuración tal cual está en disco."""
    return json.loads(_CONFIG_PATH.read_text())


CONFIG = load()

# --- Modelo LightGBM ---------------------------------------------------------
PARAMS: dict = dict(CONFIG["params"])
SEASON_DECAY: float = float(CONFIG["season_decay"])
ERA_BOOST: float = float(CONFIG["era_boost"])
SEEDS: tuple[int, ...] = tuple(CONFIG["seeds"])

# --- Simulación Monte Carlo --------------------------------------------------
SIGMA_POST_QUALI: float = float(CONFIG["sigma_post_quali"])
SIGMA_PRE_QUALI: float = float(CONFIG["sigma_pre_quali"])
CHAOS: float = float(CONFIG["chaos"])
TEAM_SHARE: float = float(CONFIG["team_share"])
RAIN_NOISE: float = float(CONFIG["rain_noise"])
RAIN_DNF: float = float(CONFIG["rain_dnf"])

# --- Blend parrilla vs. modelo ---------------------------------------------
# blend_alpha=1 -> solo modelo; 0 -> solo parrilla (prior de victoria).
BLEND_ALPHA: float = float(CONFIG["blend_alpha"])

# --- Puntos -----------------------------------------------------------------
POINTS: np.ndarray = np.asarray(CONFIG["points"], dtype=float)
SPRINT_POINTS: np.ndarray = np.asarray(CONFIG["sprint_points"], dtype=float)

# --- Cola de riesgo del campeonato -------------------------------------------
_CHAMP = CONFIG["championship"]
DRIVER_DRIFT: float = float(_CHAMP["driver_drift"])
TEAM_DRIFT: float = float(_CHAMP["team_drift"])
INCIDENT_RATE: float = float(_CHAMP["incident_rate"])
INCIDENT_BOOST: float = float(_CHAMP["incident_boost"])
STRUCTURAL_RATE: float = float(_CHAMP["structural_rate"])
STRUCTURAL_BOOST: float = float(_CHAMP["structural_boost"])
STRUCTURAL_PERSIST: float = float(_CHAMP["structural_persist"])
STRUCTURAL_PACE: float = float(_CHAMP.get("structural_pace", 0.0))

# --- Metadatos de protocolo --------------------------------------------------
TUNED_ON: str = CONFIG["tuned_on"]
HOLDOUT: str = CONFIG["holdout"]
VERSION: str = CONFIG["version"]


def snapshot() -> dict:
    """Devuelve una copia del dict completo (para loguear/hashear una corrida)."""
    return json.loads(json.dumps(CONFIG))
