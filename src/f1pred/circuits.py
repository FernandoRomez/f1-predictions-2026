"""Propiedades de circuito que no son datos por vuelta sino un prior congelado.

Incluye la tasa histórica de safety car / virtual safety car por circuito, usada
para modular el caos de carrera de forma específica en lugar de un ``CHAOS``
global. Los valores son despliegues de SC/VSC por carrera (promedio aproximado
2018-2025); se pueden refinar con OpenF1 vía :func:`f1pred.data.refresh_sc_rates`.
"""
from __future__ import annotations

import numpy as np

# Despliegues de SC/VSC por carrera (prior histórico por circuito).
SC_RATE = {
    "monaco": 1.8, "marina_bay": 1.6, "baku": 1.5, "jeddah": 1.4,
    "miami": 1.3, "vegas": 1.3, "albert_park": 1.2, "spa": 1.2,
    "montreal": 1.2, "suzuka": 1.1, "silverstone": 1.1, "interlagos": 1.1,
    "imola": 1.0, "cota": 1.0, "rodriguez": 1.0, "zandvoort": 1.0,
    "losail": 1.0, "shanghai": 1.0, "madring": 1.1, "red_bull_ring": 0.9,
    "catalunya": 0.9, "hungaroring": 0.9, "yas_marina": 0.9, "bahrain": 0.8,
    "monza": 0.7,
}
_MEAN = float(np.mean(list(SC_RATE.values())))


def sc_chaos_factor(circuit: str) -> float:
    """Multiplicador de caos por circuito: >1 si suele haber más SC/VSC, <1 si menos."""
    rate = SC_RATE.get(circuit, _MEAN)
    return float(np.clip(rate / _MEAN, 0.6, 1.6))
