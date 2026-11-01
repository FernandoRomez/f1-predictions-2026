# Predicciones F1 2026

Pronósticos probabilísticos de Fórmula 1: probabilidad de victoria, podio y puntos de cada piloto, más una simulación del campeonato. Todas las métricas son walk-forward: cada carrera se predice solo con las carreras anteriores.

Inspirado en [mar-antaya/2025_f1_predictions](https://github.com/mar-antaya/2025_f1_predictions). Es una reescritura desde cero, sin código compartido.

![Pronóstico de México 2026](results/2026_19_rodriguez.png)

## Último pronóstico: GP de la Ciudad de México, después de la qualy

Parrilla real, con las sanciones de un puesto a Verstappen y Lawson.

| Piloto | Equipo | Victoria | Podio |
|---|---|---:|---:|
| HAM | ferrari | 44.6% | 81.9% |
| RUS | mercedes | 37.9% | 79.8% |
| LEC | ferrari | 8.0% | 55.8% |
| VER | red_bull | 6.0% | 44.8% |
| ANT | mercedes | 3.5% | 34.6% |

## Estado del modelo

- **Configuración:** los hiperparámetros están versionados en `src/f1pred/config.json`. Los de simulación y el blend se eligieron con `scripts/tune.py` usando solo 2023-2024.
- **Blend con la parrilla:** existe (`blend_alpha`), pero el ajuste ciego eligió `1.0`, así que está apagado. Con qualy, `f1pred predict` aplica el valor de `blend_alpha`; sin qualy nunca se aplica, porque la parrilla no se conoce.
- **Campeonato:** incluye incidentes por carrera y problemas estructurales por coche. Esos parámetros son priores a criterio, no ajustados a datos.

## Backtest

Tabla del periodo 2025-2026 (42 carreras, tras la qualy). Esas carreras no se usaron en el ajuste de la configuración 2.2.0, pero sí se vieron al diseñar el modelo y al evaluar versiones anteriores, así que no son un test totalmente ciego.

| Método | Log-loss | Brier | Ganador | Spearman |
|---|---:|---:|---:|---:|
| **Modelo con qualy** | **0.923** | **0.470** | 61.9% | **0.670** |
| Solo parrilla | 1.150 | 0.522 | **69.0%** | 0.654 |
| Modelo sin qualy | 1.586 | 0.746 | 38.1% | 0.587 |
| Solo forma del piloto | 1.951 | 0.841 | 23.8% | 0.566 |

En log-loss y Brier, menor es mejor. Con bootstrap pareado al 90%, el modelo con qualy supera a la parrilla en log-loss (diferencia -0.228, de -0.378 a -0.076). En ganadores y Brier la diferencia no es concluyente. El detalle está en `results/backtest*.csv`.

## Campeonato

`f1pred championship` escribe la probabilidad de título en `results/2026_championship.csv`. `title_lo` y `title_hi` miden el error de muestreo Monte Carlo, no la incertidumbre del modelo.

## Variables

- **Forma:** piloto y equipo a corto y largo plazo, comparación con el compañero y posiciones ganadas.
- **Qualy:** parrilla, diferencia con la pole, forma en qualy y penalizaciones.
- **Circuito:** historial de piloto y equipo, adelantamientos, abandonos, paradas, altitud, tipo callejero y tasa de safety car.
- **Clima:** lluvia, temperatura y viento con Open-Meteo (archivo, pronóstico o normales climáticas), más rendimiento histórico en lluvia.
- **Fiabilidad y pits:** tasas de abandono, tiempo de pits, dispersión y longitud de stint por equipo.

La simulación añade ruido compartido entre compañeros, caos de carrera modulado por la tasa de SC/VSC del circuito, y más incertidumbre con lluvia. El campeonato incluye deriva de ritmo, sprints, incidentes correlacionados, problemas estructurales poco frecuentes y desempate por victorias.

Sin fuga de datos: un test modifica el resultado de una carrera y comprueba que sus variables previas no cambian. El reglamento de 2022 y 2026 reinicia la forma de los equipos.

## Uso

```bash
uv venv --python 3.13 && uv pip install -e ".[dev]"
.venv/bin/f1pred predict
.venv/bin/f1pred predict --round 20
.venv/bin/f1pred predict --grid "VER=5,ANT=4,LAW=9,LIN=8"  # sanciones de parrilla
.venv/bin/f1pred championship
.venv/bin/f1pred backtest
.venv/bin/python -m pytest
```

No necesita API keys. Los datos vienen de [Jolpica](https://github.com/jolpica/jolpica-f1), [Open-Meteo](https://open-meteo.com) y, para SC y ritmo por vuelta, [OpenF1](https://openf1.org).

## Reproducibilidad

- `uv.lock` fija las dependencias.
- `scripts/snapshot.py` genera `data_cache_hashes.json` con el SHA-256 de cada archivo de `data/`. Registra el estado local; la CI no lo compara.
- `scripts/tune.py` rehace el ajuste ciego de 2023-2024 (opción `--write` para volcar a `config.json`).
- `.github/workflows/ci.yml` corre `pytest` y `backtest`.

## Limitaciones

- El ritmo por vuelta y la degradación de neumático de OpenF1 2026 **existen** (`data.fetch_openf1_laps`) pero aún no se cablean como features: falta el mapeo estable circuito/piloto con lagging sin fuga para toda la serie. Es el siguiente paso natural.
- La tasa de SC/VSC por circuito es un prior histórico; `data.refresh_sc_rates` la puede refinar con OpenF1 (mejor esfuerzo).
- El entrenamiento usa el clima real, mientras que una carrera futura usa el pronóstico o la normal climática.
- Los parámetros de la cola del campeonato (incidentes, problema estructural) son priores razonables, no ajustados a datos.

## Licencia

MIT
