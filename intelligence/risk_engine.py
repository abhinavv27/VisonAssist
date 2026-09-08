"""
Risk Engine
===========
Calculates transparent, rule-based risk score for each detected object.
Formula from Master Project Report Section 09:
    RISK SCORE = proximity_score + centrality_score + object_danger_score + uncertainty_factor
"""

from typing import Dict, Any, Optional
from config import (
    OBJECT_DANGER_WEIGHTS,
    DEFAULT_OBJECT_DANGER,
    RISK_THRESHOLD_HIGH,
    RISK_THRESHOLD_MEDIUM,
)


class RiskEngine:
    """
    Transparent, explainable risk scoring system.
    """

    def __init__(self, danger_weights: Dict[str, int] = OBJECT_DANGER_WEIGHTS):
        self.danger_weights = danger_weights

    def calculate_risk(
        self,
        object_name: str,
        distance_meters: float,
        position: str,
        is_moving: bool = False,
        confidence: float = 1.0,
        frame_brightness: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculates composite risk score and priority tier.
        
        Returns:
            {
                "risk_score": float,       # 0.0 - 100.0
                "priority": str,          # 'HIGH' | 'MEDIUM' | 'LOW'
                "components": {
                    "base_danger": float,
                    "proximity_score": float,
                    "centrality_score": float,
                    "uncertainty_factor": float
                }
            }
        """
        # 1. Base Object Danger Score (0 - 50 points)
        raw_danger = float(self.danger_weights.get(object_name.lower(), DEFAULT_OBJECT_DANGER))
        object_danger_score = (raw_danger / 100.0) * 50.0

        # 2. Proximity Score (0 - 30 points) - Closer objects score higher
        if distance_meters <= 1.0:
            proximity_score = 30.0
        elif distance_meters <= 2.5:
            proximity_score = 20.0 + (2.5 - distance_meters) * 6.6
        elif distance_meters <= 4.5:
            proximity_score = 10.0 + (4.5 - distance_meters) * 5.0
        else:
            proximity_score = 2.0

        # Small non-hazard items (bottle, cup, phone) do not pose collision risk
        if raw_danger <= 20:
            proximity_score *= (raw_danger / 50.0)

        # 3. Centrality Score (0 - 20 points) - Center path is most dangerous
        pos_str = position.lower()
        if pos_str == "centre":
            centrality_score = 20.0
        elif "slightly" in pos_str:
            centrality_score = 12.0
        elif pos_str in ("left", "right"):
            centrality_score = 8.0
        else:
            centrality_score = 2.0

        # 4. Movement / Uncertainty Factor (0 - 10 points)
        movement_factor = 10.0 if is_moving else 0.0
        uncertainty_adjustment = 0.0
        if confidence < 0.5:
            # Slightly downweight very uncertain detections
            uncertainty_adjustment -= 5.0

        # Lighting penalty for extreme glare or severe underexposure
        if frame_brightness is not None:
            if frame_brightness < 45.0:
                uncertainty_adjustment -= 6.0
            elif frame_brightness > 215.0:
                uncertainty_adjustment -= 5.0

        composite_score = object_danger_score + proximity_score + centrality_score + movement_factor + uncertainty_adjustment
        composite_score = max(0.0, min(100.0, round(composite_score, 1)))

        # Priority Classification
        if composite_score >= RISK_THRESHOLD_HIGH:
            priority = "HIGH"
        elif composite_score >= RISK_THRESHOLD_MEDIUM:
            priority = "MEDIUM"
        else:
            priority = "LOW"

        return {
            "risk_score": composite_score,
            "priority": priority,
            "components": {
                "base_danger": round(object_danger_score, 1),
                "proximity_score": round(proximity_score, 1),
                "centrality_score": round(centrality_score, 1),
                "movement_factor": round(movement_factor, 1),
            }
        }
