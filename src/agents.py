import time
import random
import re
import logging
import os
import openai
import tiktoken
import asyncio
from game import simulate_random_game, resolve_card_effect, get_card_pool, SUPPORTED_LANGUAGES

logger = logging.getLogger("SacraBattle")

# Import CFR agent if available
try:
    from cfr.cfr_agent import CFRAgent
except ImportError:
    CFRAgent = None
    logger.debug("CFR agent not available")


class Agent:
    def __init__(self, name: str) -> None:
        self.name = name

    def choose_move(self, game_state, player):
        raise NotImplementedError


class RolloutAgent(Agent):
    def __init__(
        self, name: str, rollout_count: int = 10, exploration: float = 1.41
    ) -> None:
        super().__init__(name)
        self.rollout_count = rollout_count
        self.exploration = exploration

    def choose_move(self, game_state, player):
        start_time = time.time()
        best_card = None
        best_score = -float("inf")
        for card in player.hand:
            score = self.evaluate_move(game_state, player, card)
            logger.debug(
                f"{player.name} (Rollout) evaluated {card.name} with score {score:.2f}."
            )
            if score > best_score:
                best_score = score
                best_card = card
        decision_time = time.time() - start_time
        player.decision_times.append(decision_time)
        logger.debug(
            f"{player.name} (RolloutAgent) chooses {best_card.name if best_card else 'None'} with score {best_score:.2f} in {decision_time:.4f} sec."
        )
        return best_card

    def evaluate_move(self, game_state, player, card):
        wins = 0
        total_turns = 0
        for _ in range(self.rollout_count):
            sim_state = game_state.clone()
            sim_player = sim_state.current_player()
            matching = [c for c in sim_player.hand if c.name == card.name]
            if matching:
                resolve_card_effect(sim_state, sim_player, matching[0])
            winner, turns = simulate_random_game(sim_state)
            total_turns += turns
            if winner == game_state.current_player_idx:
                wins += 1
        win_rate = wins / self.rollout_count
        avg_turns = total_turns / self.rollout_count
        score = win_rate * 100 - avg_turns
        return score


class LLMAgent(Agent):
    def __init__(
        self,
        name: str,
        model_name: str,
        append_cards: bool,
        card_data: list,
        rules_text: str,
        add_rules: bool = False,
        include_cards: bool = True,
        language: str = "en",
        llm_type: str = "openrouter",
        vllm_host: str = "localhost",
        vllm_port: int = 8000,
        qwen_think: bool = False,
        prompt_strategy: str = "baseline",  # New parameter for prompt ablation
        parse_mode: str = "soft",  # "strict" or "soft" parsing
    ) -> None:
        super().__init__(name)
        self.llm_type = llm_type
        self.vllm_host = vllm_host
        self.vllm_port = vllm_port
        self.model_name = model_name
        self.append_cards = append_cards
        self.card_data = card_data
        self.rules_text = rules_text
        self.add_rules = add_rules
        self.include_cards = include_cards  # include card details only in the first prompt as needed
        self.language = language
        self.qwen_think = qwen_think
        self.prompt_strategy = prompt_strategy  # Store prompt strategy
        self.parse_mode = parse_mode  # Store parsing mode
        
        if self.llm_type == "openrouter":
            base = "https://openrouter.ai/api/v1"
            key = os.environ.get("OPENROUTER_API_KEY")
        else:  # vLLM
            base = f"http://{self.vllm_host}:{self.vllm_port}/v1"
            key = "EMPTY"
        self.client = openai.AsyncOpenAI(api_key=key, base_url=base)

        # Initialize conversation history as a list of messages (chat style)
        self.conversation_history = []

    async def choose_move_async(self, game_state, player):
        start_time = time.time()
        prompt = self.build_prompt(game_state, player)
        logger.debug(f"{player.name} (LLMAgent) async prompt:\n{prompt}")

        self.conversation_history.append({"role": "user", "content": prompt})

        # Initialize parsing result tracking
        parsing_result = {
            "mode": self.parse_mode,
            "method": None,
            "success": False,
            "move": None
        }

        if self.language == "ar":
            system_message = (
                "أنت وكيل استراتيجي في لعبة ورق تبادلية تُدعى 'Sacra Battle'. "
                "قبل اتخاذ خطوتك، فكر خطوة بخطوة في حالة اللعبة الحالية واعتبر الخيارات المتاحة لديك. "
                "بعد التفكير، حدد أفضل حركة من البطاقات المتوفرة في يدك. "
                "أخيرًا، اخرج اسم البطاقة النهائية بين العلامتين <BEGIN_MOVE> و <END_MOVE>. "
                "يمكن أن تأتي أسبابك قبل العلامتين، ولكن لا يظهر شيء آخر بينهما. "
            )
        else:
            system_message = (
                "You are a strategic TCG agent playing Sacra Battle. "
                "Before making your move, think step-by-step about the current game state and reason through your options. "
                "After your reasoning, determine the best move from the available cards in your hand. "
                "Finally, output your final move by placing only the card name between <BEGIN_MOVE> and <END_MOVE> markers. "
                "Your reasoning can come before the markers, but nothing else should appear within them. "
            )

        if "qwen3" in self.model_name.lower() and self.qwen_think:
            system_message += "/think"
        elif "qwen3" in self.model_name.lower() and not self.qwen_think:
            system_message += "/no_think"

        messages = [
            {"role": "system", "content": system_message}
        ] + self.conversation_history
        messages = self.prune_conversation_history(messages)

        encoding = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = sum(len(encoding.encode(m["content"])) for m in messages)

        # async API call
        content = await self.call_openai_async(messages)

        cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        # Parse move based on mode
        if self.parse_mode == "strict":
            card, parsing_result = self._parse_strict_with_tracking(cleaned, player)
        else:
            card, parsing_result = self._parse_soft_with_tracking(cleaned, player)

        self.conversation_history.append({"role": "assistant", "content": cleaned})
        decision_time = time.time() - start_time
        player.decision_times.append(decision_time)
        output_tokens = len(encoding.encode(content))
        player.token_usage.append(prompt_tokens + output_tokens)

        # Store parsing result for metrics
        if hasattr(player, 'parsing_results'):
            player.parsing_results.append(parsing_result)
        else:
            player.parsing_results = [parsing_result]

        return card

    def _parse_strict_with_tracking(self, content: str, player):
        """Strict parsing with tracking: only accept moves within tags."""
        parsing_result = {
            "mode": "strict",
            "method": None,
            "success": False,
            "move": None
        }

        m = re.search(r"<BEGIN_MOVE>(.*?)<END_MOVE>", content, re.DOTALL)
        if m:
            move = m.group(1).strip()
            parsing_result["method"] = "tags_found"
            for card in player.hand:
                if card.name.lower() == move.lower():
                    logger.debug(f"Strict parse found valid move: {card.name}")
                    parsing_result["success"] = True
                    parsing_result["move"] = card.name
                    return card, parsing_result
            logger.debug(f"Strict parse found move '{move}' but not in hand")
            parsing_result["method"] = "tags_invalid_card"
        else:
            logger.debug("Strict parse failed: no tags found")
            parsing_result["method"] = "no_tags"

        # Fallback to first card
        parsing_result["method"] = "fallback_first_card"
        card = player.hand[0] if player.hand else None
        if card:
            parsing_result["success"] = True
            parsing_result["move"] = card.name
        return card, parsing_result

    def _parse_strict(self, content: str, player):
        """Legacy strict parsing for backwards compatibility."""
        card, _ = self._parse_strict_with_tracking(content, player)
        return card

    def _parse_soft_with_tracking(self, content: str, player):
        """Soft parsing with tracking: try tags first, then search for card names."""
        parsing_result = {
            "mode": "soft",
            "method": None,
            "success": False,
            "move": None
        }

        # First try tags
        m = re.search(r"<BEGIN_MOVE>(.*?)<END_MOVE>", content, re.DOTALL)
        if m:
            move = m.group(1).strip()
            for card in player.hand:
                if card.name.lower() == move.lower():
                    logger.debug(f"Soft parse found tagged move: {card.name}")
                    parsing_result["method"] = "tags_found"
                    parsing_result["success"] = True
                    parsing_result["move"] = card.name
                    return card, parsing_result

        # If no valid tags, search for card names in text
        logger.debug("Soft parse: no valid tags found, searching for card names in text")
        content_lower = content.lower()

        # Track the last found card and its position
        last_found_card = None
        last_found_position = -1

        # Search for exact card names
        for card in player.hand:
            card_name_lower = card.name.lower()
            if card_name_lower in content_lower:
                pattern = r'\b' + re.escape(card_name_lower) + r'\b'
                match = re.search(pattern, content_lower)
                if match:
                    position = match.start()
                    if position > last_found_position:
                        last_found_card = card
                        last_found_position = position
                        logger.debug(f"Soft parse found card name at position {position}: {card.name}")

        if last_found_card:
            logger.debug(f"Soft parse returning last found card: {last_found_card.name}")
            parsing_result["method"] = "exact_text_match"
            parsing_result["success"] = True
            parsing_result["move"] = last_found_card.name
            return last_found_card, parsing_result

        # Try partial matching for multi-word card names
        last_partial_card = None
        last_partial_position = -1

        for card in player.hand:
            words = card.name.split()
            for word in words:
                if len(word) > 3:
                    word_lower = word.lower()
                    if word_lower in content_lower:
                        position = content_lower.rfind(word_lower)
                        if position > last_partial_position:
                            last_partial_card = card
                            last_partial_position = position
                            logger.debug(f"Soft parse found partial match at position {position}: {card.name}")

        if last_partial_card:
            logger.debug(f"Soft parse returning last partial match: {last_partial_card.name}")
            parsing_result["method"] = "partial_text_match"
            parsing_result["success"] = True
            parsing_result["move"] = last_partial_card.name
            return last_partial_card, parsing_result

        logger.debug("Soft parse failed: no card names found in text")
        # Final fallback to first card
        parsing_result["method"] = "fallback_first_card"
        card = player.hand[0] if player.hand else None
        if card:
            parsing_result["success"] = True
            parsing_result["move"] = card.name
        return card, parsing_result

    def _parse_soft(self, content: str, player):
        """Legacy soft parsing for backwards compatibility."""
        card, _ = self._parse_soft_with_tracking(content, player)
        return card

    def build_prompt(self, game_state, player):
        opponent = game_state.opponent()
        
        # Apply prompt strategy variants
        if self.prompt_strategy == "minimal":
            # Minimal prompt - just state and cards
            if self.language == "ar":
                prompt = f"حياتك: {player.lp} | حياة الخصم: {opponent.lp}\niدك: {[c.name for c in player.hand]}\nاختر بطاقة:"
            else:
                prompt = f"You: {player.lp}HP | Opponent: {opponent.lp}HP\nHand: {[c.name for c in player.hand]}\nChoose card:"
            return prompt
            
        elif self.prompt_strategy == "chain_of_thought":
            # Chain-of-thought prompting
            if self.language == "ar":
                prompt = (
                    f"فكر خطوة بخطوة:\n"
                    f"1. حالة اللعبة: حياتك {player.lp}, حياة الخصم {opponent.lp}\n"
                    f"2. لوحة لعبك: {[c.name for c in player.board]}\n"
                    f"3. لوحة الخصم: {[c.name for c in opponent.board]}\n"
                    f"4. يدك: {[c.name for c in player.hand]}\n"
                    f"حلل الوضع واختر أفضل حركة بالتفصيل."
                )
            else:
                prompt = (
                    f"Think step-by-step:\n"
                    f"1. Game state: You {player.lp}HP, Opponent {opponent.lp}HP\n"
                    f"2. Your board: {[c.name for c in player.board]}\n"
                    f"3. Opponent board: {[c.name for c in opponent.board]}\n"
                    f"4. Your hand: {[c.name for c in player.hand]}\n"
                    f"Analyze the situation and choose the best move with reasoning."
                )
                
        else:  # baseline or default
            # Original baseline prompt
            if self.language == "ar":
                prompt = (
                    f"حالة اللعبة:\n"
                    f"نقاط حياتك: {player.lp}\n"
                    f"نقاط حياة الخصم: {opponent.lp}\n"
                    f"لوحة لعبك: {[c.name for c in player.board]}\n"
                    f"لوحة الخصم: {[c.name for c in opponent.board]}\n"
                    f"يدك: {[c.name for c in player.hand]}\n"
                )
            else:
                prompt = (
                    f"Game State:\nYour Life Points: {player.lp}\nOpponent Life Points: {opponent.lp}\n"
                    f"Your Board: {[c.name for c in player.board]}\nOpponent Board: {[c.name for c in opponent.board]}\n"
                    f"Your Hand: {[c.name for c in player.hand]}\n"
                )
        
        # Add card details and rules for non-minimal strategies
        if self.prompt_strategy != "minimal":
            if self.include_cards:
                if self.language == "ar":
                    prompt += "\nتفاصيل مجموعة البطاقات:\n"
                else:
                    prompt += "\nCard Pool Details:\n"
                for card in self.card_data:
                    prompt += str(card) + "\n"
                self.include_cards = False
                
            if self.add_rules:
                if self.language == "ar":
                    prompt += "\nيرجى ملاحظة قواعد اللعبة التفصيلية التالية:\n" + self.rules_text + "\n"
                else:
                    prompt += "\nPlease note the detailed game rules below:\n" + self.rules_text + "\n"
                    
        return prompt

    def prune_conversation_history(self, messages):
        """
        Prune messages so that the total token count is below a threshold.
        Uses tiktoken for an accurate token count.
        When pruning, user-assistant pairs (after the system message) are removed together.
        """
        threshold = 8192
        encoding = tiktoken.encoding_for_model("gpt-4o")

        def count_tokens(msgs):
            return sum(len(encoding.encode(m["content"])) for m in msgs)

        total_tokens = count_tokens(messages)
        while total_tokens > threshold and len(messages) > 1:
            if len(messages) >= 3:
                del messages[1:3]
            else:
                messages.pop(1)
            total_tokens = count_tokens(messages)
        return messages

    async def call_openai_async(self, messages, max_retries: int = 5) -> str:
        """Async retry loop for LLM calls, with explicit None/empty check."""
        for attempt in range(max_retries):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model_name, messages=messages
                )
                if not resp or not getattr(resp, "choices", None):
                    raise RuntimeError(f"Empty response object: {resp!r}")

                return resp.choices[0].message.content

            except Exception as e:
                logger.error(f"Async OpenAI API error on attempt {attempt+1}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(min(2 ** attempt, 60))

        logger.error("Giving up on async API; using fallback move.")
        return "<BEGIN_MOVE>No Move<END_MOVE>"


class MCTSAgent(Agent):
    """
    Proper Monte Carlo Tree Search implementation with UCB1 exploration.
    Maintains a tree structure across moves for better performance.
    """

    def __init__(self, name: str, rollout_count: int = 100, c_puct: float = 1.41) -> None:
        super().__init__(name)
        self.rollout_count = rollout_count
        self.c_puct = c_puct  # UCB exploration constant
        self.tree = {}  # Persistent tree structure
        
    class MCTSNode:
        """Node in the MCTS tree."""
        def __init__(self, state_hash: str, parent=None, move=None):
            self.state_hash = state_hash
            self.parent = parent
            self.move = move
            self.visits = 0
            self.wins = 0
            self.children = []
            self.untried_moves = None
            
        def ucb1(self, c_param: float = 1.41) -> float:
            """Calculate UCB1 value for node selection."""
            from math import log
            if self.visits == 0:
                return float('inf')
            exploitation = self.wins / self.visits
            exploration = c_param * (2 * log(self.parent.visits) / self.visits) ** 0.5
            return exploitation + exploration
            
        def best_child(self, c_param: float = 1.41):
            """Select best child using UCB1."""
            return max(self.children, key=lambda c: c.ucb1(c_param))
            
        def update(self, result: float):
            """Update node statistics."""
            self.visits += 1
            self.wins += result
            
    def get_state_hash(self, game_state) -> str:
        """Create a hash of the game state for tree indexing."""
        # Simple hash based on key game state features
        p1 = game_state.current_player()
        p2 = game_state.opponent()
        hash_str = f"{p1.lp}_{p2.lp}_{len(p1.hand)}_{len(p2.hand)}_{len(p1.board)}_{len(p2.board)}"
        return hash_str
    
    def choose_move(self, game_state, player):
        """Choose a move using MCTS with tree search."""
        from math import log
        
        start_time = time.time()
        
        # Get legal moves
        legal_moves = player.hand if player.hand else []
        if not legal_moves:
            decision_time = time.time() - start_time
            player.decision_times.append(decision_time)
            return None
            
        # Initialize root node
        state_hash = self.get_state_hash(game_state)
        if state_hash not in self.tree:
            root = self.MCTSNode(state_hash)
            root.untried_moves = legal_moves.copy()
            self.tree[state_hash] = root
        else:
            root = self.tree[state_hash]
            root.untried_moves = legal_moves.copy()
        
        # Run MCTS iterations
        for _ in range(self.rollout_count):
            node = root
            sim_state = game_state.clone()
            sim_player = sim_state.current_player()
            
            # Selection: traverse tree using UCB1
            path = [node]
            while node.untried_moves is not None and len(node.untried_moves) == 0 and len(node.children) > 0:
                node = node.best_child(self.c_puct)
                path.append(node)
                # Apply move in simulation
                if node.move:
                    matching = [c for c in sim_player.hand if c.name == node.move.name]
                    if matching:
                        resolve_card_effect(sim_state, sim_player, matching[0])
            
            # Expansion: add new node if we have untried moves
            if node.untried_moves and len(node.untried_moves) > 0:
                move = random.choice(node.untried_moves)
                node.untried_moves.remove(move)
                
                # Apply move and create child node
                matching = [c for c in sim_player.hand if c.name == move.name]
                if matching:
                    resolve_card_effect(sim_state, sim_player, matching[0])
                    
                child_hash = self.get_state_hash(sim_state)
                child = self.MCTSNode(child_hash, parent=node, move=move)
                node.children.append(child)
                self.tree[child_hash] = child
                path.append(child)
            
            # Simulation: play out random game
            winner, _ = simulate_random_game(sim_state)
            
            # Backpropagation: update statistics along path
            result = 1.0 if winner == game_state.current_player_idx else 0.0
            for node in reversed(path):
                node.update(result)
        
        # Choose best move based on visit count
        if root.children:
            best_child = max(root.children, key=lambda c: c.visits)
            best_move = best_child.move
        else:
            # Fallback to random if no children
            best_move = random.choice(legal_moves) if legal_moves else None
            
        decision_time = time.time() - start_time
        player.decision_times.append(decision_time)
        
        logger.debug(
            f"{player.name} (MCTSAgent) chooses {best_move.name if best_move else 'None'} "
            f"after {self.rollout_count} iterations in {decision_time:.4f} sec."
        )
        
        return best_move


class RandomAgent(Agent):
    def __init__(self, name: str = "RandomAgent") -> None:
        super().__init__(name)

    def choose_move(self, game_state, player):
        start_time = time.time()
        chosen = random.choice(player.hand) if player.hand else None
        decision_time = time.time() - start_time
        player.decision_times.append(decision_time)
        logger.debug(
            f"{player.name} (RandomAgent) chooses {chosen.name if chosen else 'None'} in {decision_time:.4f} sec."
        )
        return chosen