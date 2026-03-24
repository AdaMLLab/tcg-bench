"""
Community card effect engine for TCG-Bench.
Handles resolution of all community card effects.
"""

import random
import logging

logger = logging.getLogger("SacraBattle")

# Community card pools for each language
CARD_POOLS = {
    "en": [
        # Champion Cards (5) - Simple, unique effects
        {
            "id": "stone_golem",
            "name": "Stone Golem",
            "type": "Champion",
            "power": 1,
            "guard": 4,
            "effect": "Stone Golem cannot attack. (It can only defend.)",
        },
        {
            "id": "swift_scout",
            "name": "Swift Scout",
            "type": "Champion",
            "power": 2,
            "guard": 1,
            "effect": "Swift Scout can attack immediately when played (no summoning sickness).",
        },
        {
            "id": "healbot",
            "name": "Healbot Assistant",
            "type": "Champion",
            "power": 0,
            "guard": 3,
            "effect": "At the end of each of your turns, gain 1 Life Point.",
        },
        {
            "id": "twin_blade",
            "name": "Twin Blade Fighter",
            "type": "Champion",
            "power": 2,
            "guard": 2,
            "effect": "Twin Blade Fighter can attack twice per turn.",
        },
        {
            "id": "frost_mage",
            "name": "Frost Mage",
            "type": "Champion",
            "power": 2,
            "guard": 2,
            "effect": "When Frost Mage attacks, the defending champion cannot attack next turn.",
        },
        # Spell Cards (5) - Simple, immediate effects
        {
            "id": "draw_power",
            "name": "Draw Power",
            "type": "Spell",
            "effect": "Draw 2 cards from your deck.",
        },
        {
            "id": "direct_strike",
            "name": "Direct Strike",
            "type": "Spell",
            "effect": "Deal 3 damage directly to your opponent's Life Points.",
        },
        {
            "id": "shield_boost",
            "name": "Shield Boost",
            "type": "Spell",
            "effect": "Give one of your Champions +3 Guard permanently.",
        },
        {
            "id": "reset_hand",
            "name": "Reset Hand",
            "type": "Spell",
            "effect": "Put all cards from your hand back into your deck, shuffle, then draw 3 cards.",
        },
        {
            "id": "sacrifice_ritual",
            "name": "Sacrifice Ritual",
            "type": "Spell",
            "effect": "Destroy one of your Champions to gain 4 Life Points.",
        },
        # Trick Cards (5) - Clear trigger conditions
        {
            "id": "revenge_strike",
            "name": "Revenge Strike",
            "type": "Trick",
            "trigger": "When you take damage from any source.",
            "effect": "Deal 2 damage back to your opponent's Life Points.",
        },
        {
            "id": "emergency_summon",
            "name": "Emergency Summon",
            "type": "Trick",
            "trigger": "When your Life Points drop below 3.",
            "effect": "Draw 3 cards immediately.",
        },
        {
            "id": "double_block",
            "name": "Double Block",
            "type": "Trick",
            "trigger": "When an enemy Champion attacks.",
            "effect": "You may block with up to 2 Champions instead of 1.",
        },
        {
            "id": "spell_echo",
            "name": "Spell Echo",
            "type": "Trick",
            "trigger": "When you play a Spell card.",
            "effect": "Copy that Spell's effect and apply it again.",
        },
        {
            "id": "final_stand",
            "name": "Final Stand",
            "type": "Trick",
            "trigger": "When you have exactly 1 Life Point remaining.",
            "effect": "All your Champions gain +2 Power until end of turn.",
        },
    ],
    "ar": [
        # Arabic translations of community cards
        # Champion Cards (5)
        {
            "id": "stone_golem",
            "name": "جولم الحجر",
            "type": "بطل",
            "power": 1,
            "guard": 4,
            "effect": "جولم الحجر لا يستطيع الهجوم. (يمكنه الدفاع فقط.)",
        },
        {
            "id": "swift_scout",
            "name": "الكشاف السريع",
            "type": "بطل",
            "power": 2,
            "guard": 1,
            "effect": "الكشاف السريع يمكنه الهجوم فوراً عند اللعب (بدون تعب الاستدعاء).",
        },
        {
            "id": "healbot",
            "name": "مساعد الشفاء",
            "type": "بطل",
            "power": 0,
            "guard": 3,
            "effect": "في نهاية كل دور من أدوارك، اكسب نقطة حياة واحدة.",
        },
        {
            "id": "twin_blade",
            "name": "مقاتل النصل المزدوج",
            "type": "بطل",
            "power": 2,
            "guard": 2,
            "effect": "مقاتل النصل المزدوج يمكنه الهجوم مرتين في الدور.",
        },
        {
            "id": "frost_mage",
            "name": "ساحر الصقيع",
            "type": "بطل",
            "power": 2,
            "guard": 2,
            "effect": "عندما يهاجم ساحر الصقيع، البطل المدافع لا يمكنه الهجوم في الدور التالي.",
        },
        # Spell Cards (5)
        {
            "id": "draw_power",
            "name": "قوة السحب",
            "type": "سحر",
            "effect": "اسحب بطاقتين من مجموعتك.",
        },
        {
            "id": "direct_strike",
            "name": "الضربة المباشرة",
            "type": "سحر",
            "effect": "ألحق 3 ضرر مباشر بنقاط حياة خصمك.",
        },
        {
            "id": "shield_boost",
            "name": "تعزيز الدرع",
            "type": "سحر",
            "effect": "امنح أحد أبطالك +3 درع بشكل دائم.",
        },
        {
            "id": "reset_hand",
            "name": "إعادة تعيين اليد",
            "type": "سحر",
            "effect": "ضع جميع البطاقات من يدك مرة أخرى في مجموعتك، اخلط، ثم اسحب 3 بطاقات.",
        },
        {
            "id": "sacrifice_ritual",
            "name": "طقوس التضحية",
            "type": "سحر",
            "effect": "دمر أحد أبطالك لتكسب 4 نقاط حياة.",
        },
        # Trick Cards (5)
        {
            "id": "revenge_strike",
            "name": "ضربة الانتقام",
            "type": "خدعة",
            "trigger": "عندما تتلقى ضرراً من أي مصدر.",
            "effect": "ألحق 2 ضرر بنقاط حياة خصمك.",
        },
        {
            "id": "emergency_summon",
            "name": "استدعاء طارئ",
            "type": "خدعة",
            "trigger": "عندما تنخفض نقاط حياتك عن 3.",
            "effect": "اسحب 3 بطاقات فوراً.",
        },
        {
            "id": "double_block",
            "name": "الحجب المزدوج",
            "type": "خدعة",
            "trigger": "عندما يهاجم بطل عدو.",
            "effect": "يمكنك الحجب بما يصل إلى بطلين بدلاً من واحد.",
        },
        {
            "id": "spell_echo",
            "name": "صدى السحر",
            "type": "خدعة",
            "trigger": "عندما تلعب بطاقة سحر.",
            "effect": "انسخ تأثير ذلك السحر وطبقه مرة أخرى.",
        },
        {
            "id": "final_stand",
            "name": "الموقف الأخير",
            "type": "خدعة",
            "trigger": "عندما يكون لديك نقطة حياة واحدة بالضبط.",
            "effect": "جميع أبطالك يكسبون +2 قوة حتى نهاية الدور.",
        },
    ],
}


def get_card_pool(language: str = "en"):
    """Get the community card pool for a specific language."""
    if language not in CARD_POOLS:
        logger.warning(f"Language {language} not supported. Defaulting to English.")
        language = "en"
    return CARD_POOLS[language]


def deal_damage(game_state, target_player, amount, source="unknown"):
    """Central damage function that handles LP changes and triggers."""
    if amount <= 0:
        return

    old_lp = target_player.lp
    target_player.lp = max(0, target_player.lp - amount)
    logger.debug(f"{target_player.name} takes {amount} damage from {source}. LP: {old_lp} -> {target_player.lp}")

    # Trigger damage-related tricks for the damaged player
    trigger_tricks(game_state, "damage_taken", player=target_player)

    # Check LP thresholds
    if target_player.lp < 3 and old_lp >= 3:
        trigger_tricks(game_state, "low_health", player=target_player)
    if target_player.lp == 1 and old_lp != 1:
        trigger_tricks(game_state, "critical_health", player=target_player)


def resolve_card_effect(game_state, player, card):
    """Resolve the effect of a card."""
    logger.debug(f"{player.name} plays {card.name} ({card.card_type}).")
    opponent = game_state.opponent()

    if card.card_type in ["Champion", "بطل"]:
        # Summon champion to board
        card.summon_turn = game_state.turn_number
        player.board.append(card)

        # Stone Golem: Cannot attack (set flag)
        if card.id == "stone_golem":
            card.cannot_attack = True
            logger.debug(f"Effect: {card.name} can only defend, not attack.")

        # Swift Scout: Can attack immediately
        elif card.id == "swift_scout":
            card.can_attack_immediately = True
            logger.debug(f"Effect: {card.name} can attack immediately.")

        # Healbot: Will heal at end of turn (set flag)
        elif card.id == "healbot":
            card.heals_owner = True
            logger.debug(f"Effect: {card.name} will heal 1 LP at end of turn.")

        # Twin Blade Fighter: Can attack twice
        elif card.id == "twin_blade":
            card.attacks_per_turn = 2
            card.attacks_remaining = 2
            logger.debug(f"Effect: {card.name} can attack twice per turn.")

        # Frost Mage: Freezes defenders (handled in combat)
        elif card.id == "frost_mage":
            card.freezes_on_attack = True
            logger.debug(f"Effect: {card.name} will freeze defenders when attacking.")

    elif card.card_type in ["Spell", "سحر"]:
        player.spell_played = True

        # Draw Power: Draw 2 cards
        if card.id == "draw_power":
            cards_drawn = []
            for _ in range(2):
                drawn = player.draw()
                if drawn:
                    cards_drawn.append(drawn.name)
            logger.debug(f"Effect: {card.name} draws {cards_drawn}.")

        # Direct Strike: 3 damage to opponent LP
        elif card.id == "direct_strike":
            deal_damage(game_state, opponent, 3, source=card.name)
            logger.debug(f"Effect: {card.name} deals 3 damage to {opponent.name}.")

        # Shield Boost: +3 Guard to a champion
        elif card.id == "shield_boost":
            if player.board:
                target = random.choice(player.board)  # Could be made strategic
                target.guard += 3
                logger.debug(f"Effect: {card.name} gives +3 Guard to {target.name}.")
            else:
                logger.debug(f"Effect: {card.name} has no target (no champions in play).")

        # Reset Hand: Shuffle hand back, draw 3
        elif card.id == "reset_hand":
            # Put hand cards back in deck
            cards_returned = len(player.hand)
            player.deck.extend(player.hand)
            player.hand.clear()

            # Shuffle deck
            random.shuffle(player.deck)

            # Draw 3 cards
            cards_drawn = []
            for _ in range(3):
                drawn = player.draw()
                if drawn:
                    cards_drawn.append(drawn.name)
            logger.debug(f"Effect: {card.name} returns {cards_returned} cards and draws {cards_drawn}.")

        # Sacrifice Ritual: Destroy own champion for 4 LP
        elif card.id == "sacrifice_ritual":
            if player.board:
                sacrificed = random.choice(player.board)  # Could be made strategic
                player.board.remove(sacrificed)
                player.lp += 4
                logger.debug(f"Effect: {card.name} sacrifices {sacrificed.name} for 4 LP.")
            else:
                logger.debug(f"Effect: {card.name} has no target (no champions to sacrifice).")

        # Check for Spell Echo trick
        spell_echo = next((t for t in player.tricks_in_play if t.id == "spell_echo"), None)
        if spell_echo:
            player.tricks_in_play.remove(spell_echo)
            logger.debug(f"{player.name} triggers Spell Echo, repeating the spell effect.")

            # Repeat the spell effect
            if card.id == "draw_power":
                for _ in range(2):
                    player.draw()
            elif card.id == "direct_strike":
                deal_damage(game_state, opponent, 3, source="Spell Echo")
            elif card.id == "shield_boost" and player.board:
                target = random.choice(player.board)
                target.guard += 3
            elif card.id == "reset_hand":
                # Don't repeat this one as it would be confusing
                logger.debug("Reset Hand cannot be echoed.")
            elif card.id == "sacrifice_ritual" and player.board:
                sacrificed = random.choice(player.board)
                player.board.remove(sacrificed)
                player.lp += 4

    elif card.card_type in ["Trick", "خدعة"]:
        player.tricks_in_play.append(card)
        logger.debug(f"{player.name} sets Trick card: {card.name}.")

    game_state.last_played_card = card


def handle_combat(game_state):
    """Handle combat phase."""
    attacker = game_state.current_player()
    defender = game_state.opponent()

    # Trigger combat-related tricks
    trigger_tricks(game_state, "combat_start")

    # Find eligible attackers
    eligible = [
        c for c in attacker.board
        if not c.has_attacked and not getattr(c, 'cannot_attack', False) and (
            c.can_attack_immediately or
            (c.summon_turn is not None and game_state.turn_number > c.summon_turn)
        )
    ]

    for atk in eligible:
        # Check if frozen (can't attack this turn)
        if hasattr(atk, 'frozen_until_turn') and game_state.turn_number < atk.frozen_until_turn:
            logger.debug(f"{atk.name} is frozen and cannot attack.")
            continue

        # Handle Twin Blade multiple attacks
        if hasattr(atk, 'attacks_remaining'):
            if atk.attacks_remaining <= 0:
                continue
            atk.attacks_remaining -= 1

        atk.has_attacked = True

        # Check for Double Block trick (check DEFENDER's tricks, not attacker's)
        can_double_block = trigger_tricks(game_state, "enemy_attack", player=defender)

        # Find available blockers
        available_blockers = [c for c in defender.board if not c.has_blocked]

        if available_blockers:
            # Pick blockers (up to 2 if double block is active)
            blockers = []
            if can_double_block and len(available_blockers) >= 2:
                # Choose two blockers with highest guard
                blockers = sorted(available_blockers, key=lambda c: c.guard or 0, reverse=True)[:2]
                for b in blockers:
                    b.has_blocked = True
                logger.debug(f"{defender.name} double blocks with {blockers[0].name} and {blockers[1].name}.")
            else:
                # Single blocker
                blocker = max(available_blockers, key=lambda c: c.guard or 0)
                blocker.has_blocked = True
                blockers = [blocker]

            # Calculate combat
            total_guard = sum(b.guard for b in blockers)
            damage = atk.power - total_guard

            # Handle Frost Mage freeze effect
            if hasattr(atk, 'freezes_on_attack'):
                for blocker in blockers:
                    blocker.frozen_until_turn = game_state.turn_number + 2
                    logger.debug(f"{atk.name} freezes {blocker.name}.")

            if damage > 0:
                # Attacker wins
                for blocker in blockers:
                    defender.board.remove(blocker)
                deal_damage(game_state, defender, damage, source=atk.name)
                logger.debug(f"{atk.name} defeats blockers and deals {damage} damage.")
            else:
                # Blockers win
                attacker.board.remove(atk)
                logger.debug(f"Blockers defeat {atk.name}.")
        else:
            # Direct damage
            deal_damage(game_state, defender, atk.power, source=atk.name)
            logger.debug(f"{atk.name} deals {atk.power} direct damage to {defender.name}.")

        # Reset attack count for Twin Blade if they have attacks left
        if hasattr(atk, 'attacks_remaining') and atk.attacks_remaining > 0:
            atk.has_attacked = False  # Allow another attack this turn


def trigger_tricks(game_state, trigger_type, player=None, **kwargs):
    """Check and trigger community trick cards."""
    # If player is specified, use that player, otherwise use current player
    if player is None:
        player = game_state.current_player()

    # Revenge Strike: When you take damage
    if trigger_type == "damage_taken":
        revenge = next((t for t in player.tricks_in_play if t.id == "revenge_strike"), None)
        if revenge:
            player.tricks_in_play.remove(revenge)
            # Find the opponent of the damaged player
            other_player = game_state.players[1] if game_state.players[0] == player else game_state.players[0]
            other_player.lp -= 2
            logger.debug(f"{player.name} triggers Revenge Strike, dealing 2 damage back.")

    # Emergency Summon: When LP drops below 3
    if trigger_type == "low_health":
        if player.lp < 3:
            emergency = next((t for t in player.tricks_in_play if t.id == "emergency_summon"), None)
            if emergency:
                player.tricks_in_play.remove(emergency)
                logger.debug(f"{player.name} triggers Emergency Summon.")
                for _ in range(3):
                    player.draw()

    # Double Block: When enemy attacks (handled in combat)
    if trigger_type == "enemy_attack":
        double_block = next((t for t in player.tricks_in_play if t.id == "double_block"), None)
        if double_block:
            player.tricks_in_play.remove(double_block)  # Remove after use
            logger.debug(f"{player.name} triggers Double Block.")
            return True  # Signal that double blocking is allowed

    # Final Stand: When at exactly 1 LP
    if trigger_type == "critical_health":
        if player.lp == 1:
            final_stand = next((t for t in player.tricks_in_play if t.id == "final_stand"), None)
            if final_stand:
                player.tricks_in_play.remove(final_stand)
                for champ in player.board:
                    champ.power += 2
                    champ.temp_final_stand_bonus = 2
                    champ.bonus_expires_turn = game_state.turn_number + 1
                logger.debug(f"{player.name} triggers Final Stand, all champions gain +2 Power.")

    return False


def handle_end_of_turn(game_state):
    """Handle end-of-turn effects."""
    current_player = game_state.current_player()

    # Trigger end-of-turn tricks
    trigger_tricks(game_state, "end_of_turn")

    # Healbot effect
    for champ in current_player.board:
        if hasattr(champ, 'heals_owner') and champ.heals_owner:
            current_player.lp += 1
            logger.debug(f"{champ.name} heals {current_player.name} for 1 LP.")

    # Reset Twin Blade attacks
    for champ in current_player.board:
        if hasattr(champ, 'attacks_per_turn'):
            champ.attacks_remaining = champ.attacks_per_turn

    # Remove temporary bonuses from BOTH players' boards (Final Stand can affect either player)
    for player in game_state.players:
        for champ in player.board:
            # Remove Final Stand bonus if expired
            if hasattr(champ, 'temp_final_stand_bonus') and hasattr(champ, 'bonus_expires_turn'):
                if game_state.turn_number >= champ.bonus_expires_turn:
                    champ.power -= champ.temp_final_stand_bonus
                    del champ.temp_final_stand_bonus
                    del champ.bonus_expires_turn
                    logger.debug(f"{champ.name}'s Final Stand bonus expires.")

            # Clear frozen status if expired
            if hasattr(champ, 'frozen_until_turn'):
                if game_state.turn_number >= champ.frozen_until_turn:
                    del champ.frozen_until_turn
                    logger.debug(f"{champ.name} is no longer frozen.")