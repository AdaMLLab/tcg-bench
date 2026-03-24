"""
Information set abstraction for CFR in Sacra Battle.
Based on theoretical analysis to reduce state space complexity.

This module maintains backward compatibility while delegating to
the new abstraction_levels module.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import sys
import os

# Add parent directory to import game components
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import GameState, Player, Card
from abstraction_levels import (
    AbstractionLevel,
    HighInfoSet as InfoSet,  # Maintain backward compatibility
    HighAbstractor as StateAbstractor,  # Maintain backward compatibility
    create_abstractor
)


# The original InfoSet and StateAbstractor classes have been moved to
# abstraction_levels.py as HighInfoSet and HighAbstractor.
# They are imported above with aliases for backward compatibility.


def abstract_action(card: Card) -> str:
    """
    Create abstract representation of an action (card play).

    Actions are abstracted to:
    - Champions: aggressive/defensive/balanced based on stats
    - Spells/Tricks: by card ID (limited set)

    This function is used across all abstraction levels.
    """
    if card.card_type in ["Champion", "بطل"]:
        if not card.power or not card.guard:
            return f"champ_special_{card.id}"
        elif card.power > card.guard:
            return f"champ_aggressive_{card.id}"
        elif card.guard > card.power:
            return f"champ_defensive_{card.id}"
        else:
            return f"champ_balanced_{card.id}"
    else:
        # Use card ID directly for spells and tricks
        return card.id