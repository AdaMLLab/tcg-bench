"""
Abstraction level definitions for CFR in Sacra Battle.

Provides different levels of state abstraction to trade off between
convergence speed and strategic accuracy.
"""

from enum import Enum
from typing import List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import sys
import os

# Add parent directory to import game components
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import GameState, Player, Card


class AbstractionLevel(Enum):
    """Abstraction level options for CFR."""
    HIGH = "high"      # Current implementation - aggressive abstraction
    MEDIUM = "medium"  # More granular for better accuracy
    LOW = "low"        # Minimal abstraction for maximum fidelity


@dataclass(frozen=True)
class BaseInfoSet(ABC):
    """Base class for information sets."""

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass


@dataclass(frozen=True)
class HighInfoSet(BaseInfoSet):
    """
    High abstraction information set (current implementation).
    Aggressively buckets states for fast convergence.
    ~2,000-10,000 unique states in practice.
    """
    phase: str                              # early/mid/late
    life_diff: int                         # [-10, +10] bucketed
    hand_strength: str                     # weak/medium/strong
    board_control: str                     # losing/neutral/winning
    tricks_active: int                     # [0, 3] capped
    hand_composition: Tuple[int, int, int] # (champions, spells, tricks) bucketed

    def __hash__(self):
        return hash((
            self.phase,
            self.life_diff,
            self.hand_strength,
            self.board_control,
            self.tricks_active,
            self.hand_composition
        ))

    def __str__(self):
        return (
            f"HighInfoSet(phase={self.phase}, life_diff={self.life_diff}, "
            f"hand={self.hand_strength}, board={self.board_control}, "
            f"tricks={self.tricks_active}, composition={self.hand_composition})"
        )


@dataclass(frozen=True)
class MediumInfoSet(BaseInfoSet):
    """
    Medium abstraction information set.
    Less aggressive bucketing for better strategic accuracy.
    ~50,000-200,000 unique states expected.
    """
    phase: str                              # 5 buckets: opening/early/mid/late/endgame
    life_diff: int                         # Exact values [-20, +20]
    hand_strength: float                   # Continuous value, 5 buckets
    board_control: float                   # Continuous score, 5 buckets
    tricks_active: int                     # Exact count
    hand_composition: Tuple[int, int, int] # Exact counts up to 5
    hand_size: int                         # Bucketed: 0, 1-2, 3-4, 5-6, 7+
    opponent_board_size: int               # Bucketed: 0, 1, 2, 3+

    def __hash__(self):
        return hash((
            self.phase,
            self.life_diff,
            int(self.hand_strength * 100),  # Convert float to int for hashing
            int(self.board_control * 100),
            self.tricks_active,
            self.hand_composition,
            self.hand_size,
            self.opponent_board_size
        ))

    def __str__(self):
        return (
            f"MediumInfoSet(phase={self.phase}, life_diff={self.life_diff}, "
            f"hand_str={self.hand_strength:.1f}, board_ctrl={self.board_control:.1f}, "
            f"tricks={self.tricks_active}, comp={self.hand_composition}, "
            f"hand_size={self.hand_size}, opp_board={self.opponent_board_size})"
        )


@dataclass(frozen=True)
class LowInfoSet(BaseInfoSet):
    """
    Low abstraction information set.
    Minimal bucketing for maximum strategic fidelity.
    ~1,000,000+ unique states possible.
    """
    turn_number: int                        # Exact turn
    life_diff: int                         # Exact difference
    my_life: int                           # Exact life points
    hand_cards: Tuple[str, ...]           # Card IDs in hand (sorted)
    board_champions: Tuple[Tuple[str, int, int], ...]  # (id, power, guard) tuples
    opponent_board_size: int               # Exact count
    opponent_tricks: int                   # Exact count
    deck_remaining: int                    # Cards left in deck
    last_played_type: Optional[str]        # Type of last card played

    def __hash__(self):
        return hash((
            self.turn_number,
            self.life_diff,
            self.my_life,
            self.hand_cards,
            self.board_champions,
            self.opponent_board_size,
            self.opponent_tricks,
            self.deck_remaining,
            self.last_played_type
        ))

    def __str__(self):
        return (
            f"LowInfoSet(turn={self.turn_number}, life_diff={self.life_diff}, "
            f"life={self.my_life}, hand={len(self.hand_cards)} cards, "
            f"board={len(self.board_champions)} champs, "
            f"opp_board={self.opponent_board_size}, deck={self.deck_remaining})"
        )


class AbstractorBase(ABC):
    """Base class for state abstractors."""

    @abstractmethod
    def abstract_state(self, game_state: GameState, player: Player) -> BaseInfoSet:
        """Convert game state to information set."""
        pass


class HighAbstractor(AbstractorBase):
    """
    High-level abstractor (current implementation).
    Aggressive bucketing for fast convergence.
    """

    # Thresholds from current implementation
    EARLY_GAME_TURNS = 5
    MID_GAME_TURNS = 15

    LIFE_DIFF_MIN = -10
    LIFE_DIFF_MAX = 10

    HAND_STRENGTH_WEAK_THRESHOLD = 5.0
    HAND_STRENGTH_STRONG_THRESHOLD = 10.0

    BOARD_CONTROL_LOSING_THRESHOLD = -3.0
    BOARD_CONTROL_WINNING_THRESHOLD = 3.0

    MAX_TRICKS_TRACKED = 3

    def abstract_state(self, game_state: GameState, player: Player) -> HighInfoSet:
        """Convert to high-abstraction information set."""
        phase = self._get_game_phase(game_state)
        life_diff = self._get_life_differential(game_state, player)
        hand_strength = self._evaluate_hand_strength(player.hand)
        board_control = self._evaluate_board_control(game_state, player)
        tricks_active = self._count_active_tricks(player)
        hand_composition = self._abstract_hand_composition(player.hand)

        return HighInfoSet(
            phase=phase,
            life_diff=life_diff,
            hand_strength=hand_strength,
            board_control=board_control,
            tricks_active=tricks_active,
            hand_composition=hand_composition
        )

    def _get_game_phase(self, game_state: GameState) -> str:
        if game_state.turn_number <= self.EARLY_GAME_TURNS:
            return "early"
        elif game_state.turn_number <= self.MID_GAME_TURNS:
            return "mid"
        else:
            return "late"

    def _get_life_differential(self, game_state: GameState, player: Player) -> int:
        opponent = game_state.players[1] if game_state.players[0] == player else game_state.players[0]
        diff = player.lp - opponent.lp
        return max(self.LIFE_DIFF_MIN, min(self.LIFE_DIFF_MAX, diff))

    def _evaluate_hand_strength(self, hand: List[Card]) -> str:
        if not hand:
            return "weak"

        total_value = 0.0
        for card in hand:
            if card.card_type in ["Champion", "بطل"]:
                power_val = card.power if card.power else 0
                guard_val = card.guard if card.guard else 0
                total_value += power_val * 1.5 + guard_val
            elif card.card_type in ["Spell", "سحر"]:
                effect_lower = card.effect_text.lower()
                if "damage" in effect_lower or "3" in effect_lower:
                    total_value += 4.5
                elif "draw" in effect_lower:
                    total_value += 3.0
                elif "shield" in effect_lower or "guard" in effect_lower:
                    total_value += 2.5
                elif "life" in effect_lower:
                    total_value += 2.0
                else:
                    total_value += 1.5
            elif card.card_type in ["Trick", "خدعة"]:
                total_value += 1.0

        if total_value < self.HAND_STRENGTH_WEAK_THRESHOLD:
            return "weak"
        elif total_value < self.HAND_STRENGTH_STRONG_THRESHOLD:
            return "medium"
        else:
            return "strong"

    def _evaluate_board_control(self, game_state: GameState, player: Player) -> str:
        opponent = game_state.players[1] if game_state.players[0] == player else game_state.players[0]

        my_power = sum(c.power for c in player.board if c.power)
        my_guard = sum(c.guard for c in player.board if c.guard)
        my_count = len(player.board)

        opp_power = sum(c.power for c in opponent.board if c.power)
        opp_guard = sum(c.guard for c in opponent.board if c.guard)
        opp_count = len(opponent.board)

        my_score = my_power * 1.2 + my_guard * 0.8 + my_count * 0.5
        opp_score = opp_power * 1.2 + opp_guard * 0.8 + opp_count * 0.5

        diff = my_score - opp_score

        if diff < self.BOARD_CONTROL_LOSING_THRESHOLD:
            return "losing"
        elif diff > self.BOARD_CONTROL_WINNING_THRESHOLD:
            return "winning"
        else:
            return "neutral"

    def _count_active_tricks(self, player: Player) -> int:
        return min(self.MAX_TRICKS_TRACKED, len(player.tricks_in_play))

    def _abstract_hand_composition(self, hand: List[Card]) -> Tuple[int, int, int]:
        champions = sum(1 for c in hand if c.card_type in ["Champion", "بطل"])
        spells = sum(1 for c in hand if c.card_type in ["Spell", "سحر"])
        tricks = sum(1 for c in hand if c.card_type in ["Trick", "خدعة"])

        # Bucket counts
        champions_bucket = min(3, champions)  # 0, 1, 2, 3+
        spells_bucket = min(3, spells)        # 0, 1, 2, 3+
        tricks_bucket = min(2, tricks)        # 0, 1, 2+

        return (champions_bucket, spells_bucket, tricks_bucket)


class MediumAbstractor(AbstractorBase):
    """
    Medium-level abstractor.
    More granular bucketing for better accuracy.
    """

    def abstract_state(self, game_state: GameState, player: Player) -> MediumInfoSet:
        """Convert to medium-abstraction information set."""
        phase = self._get_game_phase(game_state)
        life_diff = self._get_life_differential(game_state, player)
        hand_strength = self._evaluate_hand_strength(player.hand)
        board_control = self._evaluate_board_control(game_state, player)
        tricks_active = len(player.tricks_in_play)  # Exact count
        hand_composition = self._get_hand_composition(player.hand)
        hand_size = self._bucket_hand_size(len(player.hand))
        opp_board_size = self._bucket_opponent_board(game_state, player)

        return MediumInfoSet(
            phase=phase,
            life_diff=life_diff,
            hand_strength=hand_strength,
            board_control=board_control,
            tricks_active=tricks_active,
            hand_composition=hand_composition,
            hand_size=hand_size,
            opponent_board_size=opp_board_size
        )

    def _get_game_phase(self, game_state: GameState) -> str:
        turn = game_state.turn_number
        if turn <= 3:
            return "opening"
        elif turn <= 7:
            return "early"
        elif turn <= 12:
            return "mid"
        elif turn <= 20:
            return "late"
        else:
            return "endgame"

    def _get_life_differential(self, game_state: GameState, player: Player) -> int:
        opponent = game_state.players[1] if game_state.players[0] == player else game_state.players[0]
        diff = player.lp - opponent.lp
        # Cap at ±20 but keep exact values within range
        return max(-20, min(20, diff))

    def _evaluate_hand_strength(self, hand: List[Card]) -> float:
        """Return continuous hand strength value."""
        if not hand:
            return 0.0

        total_value = 0.0
        for card in hand:
            if card.card_type in ["Champion", "بطل"]:
                power_val = card.power if card.power else 0
                guard_val = card.guard if card.guard else 0
                total_value += power_val * 1.8 + guard_val * 1.2
            elif card.card_type in ["Spell", "سحر"]:
                effect_lower = card.effect_text.lower()
                if "damage" in effect_lower or "3" in effect_lower:
                    total_value += 5.0
                elif "draw" in effect_lower:
                    total_value += 4.0
                elif "shield" in effect_lower or "guard" in effect_lower:
                    total_value += 3.0
                elif "life" in effect_lower:
                    total_value += 2.5
                else:
                    total_value += 2.0
            elif card.card_type in ["Trick", "خدعة"]:
                total_value += 1.5

        # Bucket into 5 levels
        if total_value < 3:
            return 1.0
        elif total_value < 6:
            return 2.0
        elif total_value < 10:
            return 3.0
        elif total_value < 15:
            return 4.0
        else:
            return 5.0

    def _evaluate_board_control(self, game_state: GameState, player: Player) -> float:
        """Return continuous board control score."""
        opponent = game_state.players[1] if game_state.players[0] == player else game_state.players[0]

        my_power = sum(c.power for c in player.board if c.power)
        my_guard = sum(c.guard for c in player.board if c.guard)
        my_count = len(player.board)

        opp_power = sum(c.power for c in opponent.board if c.power)
        opp_guard = sum(c.guard for c in opponent.board if c.guard)
        opp_count = len(opponent.board)

        my_score = my_power * 1.5 + my_guard + my_count * 0.7
        opp_score = opp_power * 1.5 + opp_guard + opp_count * 0.7

        diff = my_score - opp_score

        # Bucket into 5 levels
        if diff < -10:
            return -2.0
        elif diff < -3:
            return -1.0
        elif diff < 3:
            return 0.0
        elif diff < 10:
            return 1.0
        else:
            return 2.0

    def _get_hand_composition(self, hand: List[Card]) -> Tuple[int, int, int]:
        """Exact counts up to 5 per type."""
        champions = sum(1 for c in hand if c.card_type in ["Champion", "بطل"])
        spells = sum(1 for c in hand if c.card_type in ["Spell", "سحر"])
        tricks = sum(1 for c in hand if c.card_type in ["Trick", "خدعة"])

        # Cap at 5 but keep exact counts
        return (min(5, champions), min(5, spells), min(5, tricks))

    def _bucket_hand_size(self, size: int) -> int:
        """Bucket hand size."""
        if size == 0:
            return 0
        elif size <= 2:
            return 1
        elif size <= 4:
            return 2
        elif size <= 6:
            return 3
        else:
            return 4

    def _bucket_opponent_board(self, game_state: GameState, player: Player) -> int:
        """Bucket opponent board size."""
        opponent = game_state.players[1] if game_state.players[0] == player else game_state.players[0]
        opp_board_size = len(opponent.board)

        if opp_board_size == 0:
            return 0
        elif opp_board_size == 1:
            return 1
        elif opp_board_size == 2:
            return 2
        else:
            return 3


class LowAbstractor(AbstractorBase):
    """
    Low-level abstractor.
    Minimal abstraction for maximum fidelity.
    """

    def abstract_state(self, game_state: GameState, player: Player) -> LowInfoSet:
        """Convert to low-abstraction information set."""
        opponent = game_state.players[1] if game_state.players[0] == player else game_state.players[0]

        # Get exact hand cards (sorted for consistency)
        hand_cards = tuple(sorted(c.id for c in player.hand if c.id))

        # Get board champions with exact stats
        board_champions = tuple(sorted(
            (c.id, c.power if c.power else 0, c.guard if c.guard else 0)
            for c in player.board
            if c.card_type in ["Champion", "بطل"]
        ))

        # Get last played card type if available
        last_played_type = None
        if hasattr(game_state, 'last_played_card') and game_state.last_played_card:
            last_played_type = game_state.last_played_card.card_type

        return LowInfoSet(
            turn_number=game_state.turn_number,
            life_diff=player.lp - opponent.lp,
            my_life=player.lp,
            hand_cards=hand_cards,
            board_champions=board_champions,
            opponent_board_size=len(opponent.board),
            opponent_tricks=len(opponent.tricks_in_play),
            deck_remaining=len(player.deck),
            last_played_type=last_played_type
        )


def create_abstractor(level: AbstractionLevel) -> AbstractorBase:
    """Factory function to create appropriate abstractor."""
    # Handle both enum and string values
    if isinstance(level, str):
        level = AbstractionLevel(level)

    if level == AbstractionLevel.HIGH:
        return HighAbstractor()
    elif level == AbstractionLevel.MEDIUM:
        return MediumAbstractor()
    elif level == AbstractionLevel.LOW:
        return LowAbstractor()
    else:
        raise ValueError(f"Unknown abstraction level: {level}")