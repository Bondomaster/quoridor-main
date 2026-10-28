"""Monte Carlo Tree Search (MCTS) Agent for Wrongway (Quoridor 9x9).
Implemented using Python standard library (no numpy required).
"""

from collections import defaultdict
import math
import random
from game_state import GameState


class SearchNode:
    def __init__(self, state: GameState, parent=None, parent_action=None, player_one_maximizer=False):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.number_of_visits = 0
        self.results = defaultdict(int)
        self.results[1] = 0
        self.results[-1] = 0
        self.player_one_maximizer = player_one_maximizer
        self._untried_actions = self.untried_actions()

    def untried_actions(self):
        # Current turn in tree node depends on who is moving
        cur_player_is_one = self.state.player_one
        self._untried_actions = self.state.get_all_child_states(player_one_maximizer=cur_player_is_one,
                                                                include_state=True)
        random.shuffle(self._untried_actions)
        return self._untried_actions

    def q(self):
        wins = self.results[1]
        losses = self.results[-1]
        return wins - losses

    def n(self):
        return self.number_of_visits

    def expand(self):
        next_state, action = self._untried_actions.pop()
        child_node = SearchNode(next_state, parent=self, parent_action=action,
                                player_one_maximizer=self.player_one_maximizer)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.is_goal_state()

    def rollout(self):
        current_rollout_state = self.state
        steps = 0
        max_steps = 20
        while not current_rollout_state.is_goal_state() and steps < max_steps:
            child_states = current_rollout_state.get_all_child_states(
                player_one_maximizer=current_rollout_state.player_one,
                include_state=True
            )
            if not child_states:
                break
            current_rollout_state = self.rollout_policy(child_states)[0]
            steps += 1
        return current_rollout_state.game_result(self.player_one_maximizer)

    def back_propagate(self, result):
        self.number_of_visits += 1
        self.results[result] += 1
        if self.parent:
            self.parent.back_propagate(result)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c_param=0.1):
        choices_weights = [
            (c.q() / c.n()) + c_param * math.sqrt(2 * math.log(max(1, self.n())) / c.n())
            if c.n() > 0 else float("inf")
            for c in self.children
        ]
        best_idx = max(range(len(choices_weights)), key=lambda i: choices_weights[i])
        return self.children[best_idx]

    def rollout_policy(self, possible_moves):
        return random.choice(possible_moves)

    def tree_policy(self):
        current_node = self
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                if not current_node.children:
                    break
                current_node = current_node.best_child()
        return current_node

    def best_action(self):
        if not self.player_one_maximizer:
            simulation_no = 20 if self.state.player_two_wall_num != 0 else 30
        else:
            simulation_no = 20 if self.state.player_one_walls_num != 0 else 30

        for _ in range(simulation_no):
            v = self.tree_policy()
            reward = v.rollout()
            v.back_propagate(reward)

        if not self.children:
            return None
        return self.best_child(c_param=0.0)


def choose_action(observation):
    """Entry point for Wrongway arena."""
    moves = observation.get("legal_moves", [])
    goal_row = observation["you"]["goal_row"]
    for m in moves:
        if m[0] == goal_row:
            return {"type": "move", "to": list(m)}

    game_state = GameState.from_observation(observation)
    player_one_maximizer = (observation["player_id"] == 0)

    root = SearchNode(game_state, player_one_maximizer=player_one_maximizer)
    best_child_node = root.best_action()

    if best_child_node and best_child_node.parent_action:
        return best_child_node.parent_action

    if moves:
        distances = observation.get("distances")
        if distances:
            reachable = [m for m in moves if distances[m[0]][m[1]] >= 0]
            candidates = reachable or moves
            best_m = min(candidates, key=lambda m: distances[m[0]][m[1]] if distances[m[0]][m[1]] >= 0 else float("inf"))
            return {"type": "move", "to": list(best_m)}
        return {"type": "move", "to": list(moves[0])}

    return {"type": "pass"}
