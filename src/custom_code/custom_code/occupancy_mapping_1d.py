"""One-dimensional occupancy grid mapping using a simple inverse sensor model."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


CELL_SIZE_CM = 10
MAP_MAX_DISTANCE_CM = 200
PRIOR_OCCUPANCY = 0.5
P_OCCUPIED_BEFORE = 0.3
P_OCCUPIED_AFTER = 0.6
NO_UPDATE_MARGIN_CM = 20


def logit(probability: float) -> float:
    """Convert a probability into log-odds."""
    return np.log(probability / (1.0 - probability))


def inverse_sensor_model(cell_distance: float, measurement: float) -> float | None:
    """
    Return the occupancy probability suggested by the measurement for a given cell.

    Cells further than `NO_UPDATE_MARGIN_CM` behind the measurement remain unchanged.
    Cells just before the measurement are likely free (low occupancy probability), while
    cells at or beyond the measured distance are considered more likely occupied.
    """
    if cell_distance < measurement:
        return logit(P_OCCUPIED_BEFORE)
    elif  cell_distance - measurement < NO_UPDATE_MARGIN_CM:
        return logit(P_OCCUPIED_AFTER)
    return 0


def compute_occupancy_grid(measurements_cm: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return cell coordinates and occupancy probabilities after processing measurements."""
    cells = np.arange(0, MAP_MAX_DISTANCE_CM + CELL_SIZE_CM, CELL_SIZE_CM, dtype=float)
    log_prior = logit(PRIOR_OCCUPANCY)
    log_belief = np.full_like(cells, log_prior, dtype=float)

    for measurement in measurements_cm:
        for idx, cell_distance in enumerate(cells):
            probability = inverse_sensor_model(cell_distance, measurement)
            log_belief[idx] += probability - log_prior

    occupancy = 1.0 / (1.0 + np.exp(-log_belief))
    return cells, occupancy


def main() -> None:
    measurements = [101, 82, 91, 112, 99, 151, 96, 85, 99, 105]
    cells, occupancy = compute_occupancy_grid(measurements)

    plt.figure(figsize=(10, 4))
    plt.plot(cells, occupancy, marker="o")
    plt.xlabel("Distancia desde el robot [cm]")
    plt.ylabel("Probabilidad de ocupación")
    plt.title("Mapa de grilla 1D a partir de mediciones de distancia")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

