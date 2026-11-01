# Predicciones F1 2026

Pronósticos probabilísticos de Fórmula 1: probabilidad de victoria, podio y puntos de cada piloto, más una simulación del campeonato. Todas las métricas son walk-forward: cada carrera se predice solo con las carreras anteriores.

Inspirado en [mar-antaya/2025_f1_predictions](https://github.com/mar-antaya/2025_f1_predictions). Es una reescritura desde cero, sin código compartido.

![Pronóstico de México 2026](results/2026_19_rodriguez.png)

## Último pronóstico: GP de la Ciudad de México, después de la qualy

Parrilla real, con las sanciones de un puesto a Verstappen y Lawson por su contacto en FP2.

| Piloto | Equipo | Salida | Victoria | Podio |
|---|---|---:|---:|---:|
| HAM | Ferrari | 1 | 35.1% | 76.6% |
| RUS | Mercedes | 2 | 30.8% | 73.1% |
| VER | Red Bull | 5 | 20.3% | 64.8% |
| LEC | Ferrari | 3 | 11.2% | 54.7% |
| ANT | Mercedes | 4 | 2.3% | 24.1% |

En el campeonato, Antonelli gana en todas las simulaciones, con 123 puntos de ventaja y 125 en juego.

## Variables

- **Forma:** piloto y equipo a corto y largo plazo, comparación con el compañero y posiciones ganadas.
- **Qualy:** parrilla, diferencia con la pole, forma en qualy y penalizaciones.
- **Circuito:** historial de piloto y equipo, adelantamientos, abandonos, paradas, altitud y tipo callejero.
- **Clima:** lluvia, temperatura y viento con Open-Meteo, más rendimiento histórico en lluvia.
- **Fiabilidad y pits:** tasas de abandono y tiempo de pits por equipo.

La simulación añade ruido compartido entre compañeros, caos de carrera como aproximación a safety cars y estrategia, y más incertidumbre con lluvia. El campeonato incluye deriva de ritmo, sprints y desempate por victorias.

Sin fuga de datos: un test modifica el resultado de una carrera y comprueba que sus variables previas no cambian. El reglamento de 2022 y 2026 reinicia la forma de los equipos.

## Backtest

El ruido se eligió solo con 2023-24. Las 42 carreras de 2025-26 no se usaron para ajustarlo, pero sí se vieron al diseñar el modelo, así que no son un test totalmente ciego.

| Método | Log-loss | Brier | Ganador | Top 3 | Spearman |
|---|---:|---:|---:|---:|---:|
| Modelo con qualy | **0.949** | **0.496** | 61.9% | **70.6%** | **0.671** |
| Solo parrilla | 1.150 | 0.522 | **69.0%** | 69.8% | 0.654 |
| Modelo sin qualy | 1.759 | 0.795 | 31.0% | 57.1% | 0.589 |
| Solo forma del piloto | 1.951 | 0.841 | 23.8% | 54.0% | 0.566 |

En log-loss y Brier, menor es mejor. Solo 2026 (18 carreras): el modelo con qualy gana en log-loss (1.110 frente a 1.198) y Spearman, pero la parrilla gana en Brier (0.489 frente a 0.511) y en ganadores (72.2% frente a 61.1%). La parrilla necesita la qualy, así que el modelo sin qualy se compara con la forma del piloto.

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

No necesita API keys. Los datos vienen de [Jolpica](https://github.com/jolpica/jolpica-f1) y [Open-Meteo](https://open-meteo.com).

## Limitaciones

- No hay telemetría ni neumáticos por vuelta: en las pruebas, FastF1 no devolvió vueltas de 2026 y OpenF1 pidió autenticación.
- Los safety cars no se modelan directamente; el caos de carrera es una aproximación.
- El entrenamiento usa el clima real, mientras que una carrera futura usa el pronóstico. Más allá de 14 días no hay dato.
- Los parámetros de caos, lluvia y deriva se fijaron a criterio y no se ajustaron.

## Licencia

MIT
