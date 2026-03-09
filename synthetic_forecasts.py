"""Generates realistic synthetic predictions for trading signals.

Used during development/testing before ML models are fully trained.
Predictions are:
  • Scaled and capped to [-20, +20]
  • Distributed with ~50% in [-10, +10] (average absolute ~10)
  • Realistic variation to test portfolio logic
"""

from datetime import datetime

import numpy as np


def generate_synthetic_forecasts(
    instruments: list[str],
    seed: int = None,
    scale: float = 10.0,
    cap: float = 20.0,
) -> dict[str, float]:
    """Generate synthetic predictions for a list of instruments.

    Args:
        instruments: List of instrument IDs (e.g., ["BAS.XETRA", "SAP.XETRA"])
        seed: Optional random seed for reproducibility
        scale: Standard deviation of the underlying normal distribution (default 10)
        cap: Maximum absolute value for predictions (default 20)

    Returns:
        Dictionary mapping instrument ID to prediction value in [-cap, +cap]

    Notes:
        • Predictions follow N(0, scale) clipped to [-cap, +cap]
        • With scale=10 and cap=20, ~68% fall in [-10, +10]
        • Average absolute value ≈ 10 as required
        • Use different seeds for different time periods to get variety
    """
    if seed is not None:
        np.random.seed(seed)

    predictions = {}
    for instrument in instruments:
        # Generate from normal distribution
        pred = np.random.normal(loc=0, scale=scale)
        # Clip to [-cap, +cap]
        pred = np.clip(pred, -cap, cap)
        predictions[instrument] = float(pred)

    return predictions


def get_predictions_for_bar(
    bar_timestamp: datetime,
    instruments: list[str],
) -> dict[str, float]:
    """Get synthetic predictions for a given bar/timestamp.

    This function uses the timestamp to seed the random generator,
    ensuring consistency: the same timestamp always produces the same
    predictions (within a session).

    Args:
        bar_timestamp: The datetime of the bar
        instruments: List of instrument IDs

    Returns:
        Dictionary mapping instrument ID to prediction
    """
    # Use timestamp components as seed for reproducibility
    seed = int(bar_timestamp.timestamp()) % (2**31 - 1)
    return generate_synthetic_forecasts(instruments, seed=seed)


def validate_predictions(predictions: dict[str, float]) -> bool:
    """Validate that predictions are properly scaled and capped.

    Args:
        predictions: Dictionary of instrument_id -> prediction_value

    Returns:
        True if all predictions are in [-20, +20]
    """
    for instrument, pred in predictions.items():
        if not (-20 <= pred <= 20):
            print(f"Warning: {instrument} prediction {pred} outside [-20, +20]")
            return False
    return True


if __name__ == "__main__":
    # Test the synthetic prediction generator
    instruments = ["BAS.XETRA", "SAP.XETRA", "VOW3.XETRA", "SIE.XETRA"]

    print("Sample synthetic predictions:")
    print("=" * 50)

    # Generate multiple batches to show variety
    for i in range(3):
        preds = generate_synthetic_forecasts(instruments, seed=1000 + i)
        abs_values = [abs(v) for v in preds.values()]
        print(f"\nBatch {i+1}:")
        for inst, pred in preds.items():
            signal = "BUY" if pred > 10 else "SELL" if pred < -10 else "HOLD"
            print(f"  {inst:15s} {pred:7.2f}  [{signal}]")
        print(f"  Avg absolute: {np.mean(abs_values):.2f}")

        # Verify they're capped
        assert validate_predictions(preds)

    print("\n" + "=" * 50)
    print("✓ All predictions properly scaled to [-20, +20]")
    print("✓ Ready to use in the trading system")
