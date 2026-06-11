import numpy as np
import pandas as pd
from pathlib import Path


def generate_pricing_history(n_steps=10000, seed=42):
    """
    Simulate historical demand data for a dynamic pricing environment.
    Price tiers: 0=low, 1=medium, 2=high, 3=premium
    """
    rng = np.random.default_rng(seed)

    price_levels = [80, 100, 120, 150]  # €
    base_demands = [120, 90, 60, 30]    # units per period

    records = []
    for t in range(n_steps):
        hour = t % 24
        day  = (t // 24) % 7
        # Peak hours: 8-10, 12-14, 18-21 — weekdays higher
        peak = (8 <= hour <= 10) or (12 <= hour <= 14) or (18 <= hour <= 21)
        weekday = day < 5
        season_factor = 1 + 0.2 * np.sin(2 * np.pi * t / (24 * 365))

        for a, (price, base_d) in enumerate(zip(price_levels, base_demands)):
            demand_mean = base_d * season_factor * (1.3 if peak and weekday else 0.8 if not weekday else 1.0)
            demand = max(0, int(rng.poisson(demand_mean)))
            revenue = price * demand
            records.append({
                'step': t, 'action': a, 'price': price,
                'hour': hour, 'day_of_week': day,
                'is_peak': int(peak), 'is_weekday': int(weekday),
                'demand': demand, 'revenue': revenue,
            })

    return pd.DataFrame(records)


def load_or_generate(csv_path, **kwargs):
    path = Path(csv_path)
    if path.exists():
        return pd.read_csv(path)
    df = generate_pricing_history(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
