import copy
import random
import logging
import sys
import os

# Add parent directory to path to import from data folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("SacraBattle")

# Define supported languages
SUPPORTED_LANGUAGES = ["en", "ar"]



# Dynamic engine import
engine = None

def get_card_pool(language: str = "en"):
    """Get the card pool for a specific language.

    Args:
        language: Language for the cards (en/ar)
    """
    global engine
    import community_engine
    engine = community_engine
    return engine.get_card_pool(language)


class Card:
    """Represents a card in the game."""

    def __init__(self, card_dict: dict) -> None:
        self.id = card_dict.get("id")
        self.name = card_dict["name"]
        self.card_type = card_dict["type"]
        self.effect_text = card_dict.get("effect", "")
        self.trigger = card_dict.get("trigger", None)
        self.power = card_dict.get("power", None)
        self.guard = card_dict.get("guard", None)
        self.summon_turn = None
        self.can_attack_immediately = False
        
        # For champions, track whether they have attacked or blocked this turn
        if self.card_type in ["Champion", "بطل"]:
            self.has_attacked = False
            self.has_blocked = False

    def clone(self) -> "Card":
        return copy.deepcopy(self)

    def __str__(self) -> str:
        if self.card_type in ["Champion", "بطل"]:
            return f"{self.name} ({self.card_type}) [ID: {self.id}, Power: {self.power}, Guard: {self.guard}]: {self.effect_text}"
        elif self.card_type in ["Trick", "خدعة"]:
            return f"{self.name} ({self.card_type}) [ID: {self.id}, Trigger: {self.trigger}]: {self.effect_text}"
        else:
            return f"{self.name} ({self.card_type}) [ID: {self.id}]: {self.effect_text}"


class Player:
    """Represents a player in the game."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.lp = 10
        self.deck = []
        self.hand = []
        self.board = []
        self.discard = []
        self.tricks_in_play = []
        self.spell_played = False
        self.cannot_play_more = False
        self.move_log = []
        self.decision_times = []
        self.token_usage = []

    def draw(self):
        if not self.deck:
            return None
        card = self.deck.pop(0)
        self.hand.append(card)
        logger.debug(f"{self.name} draws {card.name}.")
        return card

    def play_card(self, card: "Card"):
        if self.cannot_play_more:
            logger.debug(f"{self.name} cannot play more cards this turn.")
            return None
        if card in self.hand:
            self.hand.remove(card)
            # For Spell cards, move to discard; Champions and Tricks remain in play
            if card.card_type in ["Spell", "سحر"]:
                self.discard.append(card)
            self.move_log.append({"type": card.card_type, "name": card.name})
            return card
        return None

    def clone(self) -> "Player":
        new = Player(self.name)
        new.lp = self.lp
        new.deck = [c.clone() for c in self.deck]
        new.hand = [c.clone() for c in self.hand]
        new.board = [c.clone() for c in self.board]
        new.discard = [c.clone() for c in self.discard]
        new.tricks_in_play = [c.clone() for c in self.tricks_in_play]
        new.spell_played = self.spell_played
        new.cannot_play_more = self.cannot_play_more
        new.move_log = list(self.move_log)
        new.decision_times = list(self.decision_times)
        new.token_usage = list(self.token_usage)
        return new

    def __str__(self) -> str:
        return f"{self.name}: LP={self.lp}, Hand={[c.name for c in self.hand]}, Board={[c.name for c in self.board]}"


class GameState:
    """Represents the overall state of the game."""

    def __init__(self, player1: Player, player2: Player) -> None:
        self.players = [player1, player2]
        self.current_player_idx = 0
        self.turn_number = 1
        self.last_played_card = None
        self.game_log = []
        self.skip_combat = False

    def current_player(self) -> Player:
        return self.players[self.current_player_idx]

    def opponent(self) -> Player:
        return self.players[1 - self.current_player_idx]

    def next_turn(self) -> None:
        # Handle end-of-turn effects before switching
        global engine
        if engine is not None:
            engine.handle_end_of_turn(self)

        self.skip_combat = False
        self.current_player_idx = 1 - self.current_player_idx
        self.turn_number += 1
        current = self.current_player()
        current.spell_played = False
        current.cannot_play_more = False

        # Reset each champion's attack and block flags on the new active board
        for champ in current.board:
            if hasattr(champ, "has_attacked"):
                champ.has_attacked = False
            if hasattr(champ, "has_blocked"):
                champ.has_blocked = False

    def clone(self) -> "GameState":
        p1_clone = self.players[0].clone()
        p2_clone = self.players[1].clone()
        new = GameState(p1_clone, p2_clone)
        new.current_player_idx = self.current_player_idx
        new.turn_number = self.turn_number
        new.last_played_card = copy.deepcopy(self.last_played_card) if self.last_played_card else None
        new.game_log = list(self.game_log)
        new.skip_combat = self.skip_combat
        return new


def resolve_card_effect(game_state: GameState, player: Player, card: Card) -> None:
    """Resolve the effect of a played card."""
    global engine
    if engine is None:
        import community_engine
        engine = community_engine
    return engine.resolve_card_effect(game_state, player, card)


def combat_phase(game_state: GameState) -> None:
    """Handle the combat phase."""
    if game_state.skip_combat:
        logger.debug("Combat phase skipped.")
        return

    global engine
    if engine is None:
        import community_engine
        engine = community_engine

    # Use the engine's combat handler
    return engine.handle_combat(game_state)


def simulate_random_game(game_state: GameState) -> tuple:
    """Simulate a random game for MCTS."""
    sim = game_state.clone()
    max_turns = 50
    while (
        sim.players[0].lp > 0 and 
        sim.players[1].lp > 0 and 
        sim.turn_number < max_turns
    ):
        current = sim.current_player()
        current.draw()

        if current.hand:
            card = random.choice(current.hand)
            current.play_card(card)
            resolve_card_effect(sim, current, card)

        combat_phase(sim)

        if sim.players[0].lp <= 0 or sim.players[1].lp <= 0:
            break

        sim.next_turn()

    winner = 0 if sim.players[0].lp > sim.players[1].lp else 1
    return winner, sim.turn_number