"""Expectimax Agent for Wrongway (Quoridor 9x9).
"""

from state_evaluation_heuristic import state_evaluation_heuristic
from game_state import GameState


def expectimax(game_state: GameState, depth, maximizing_player, player_one_maximizer):
    if depth == 0 or game_state.is_goal_state():
        return state_evaluation_heuristic(game_state, player_one_maximizer, True)

    if maximizing_player:
        children = game_state.get_all_child_states(player_one_maximizer=player_one_maximizer)
        if not children:
            return state_evaluation_heuristic(game_state, player_one_maximizer, True)
        return max([expectimax(child[0], depth - 1, False, player_one_maximizer) for child in children])
    else:
        children = game_state.get_all_child_states(player_one_maximizer=not player_one_maximizer)
        if not children:
            return state_evaluation_heuristic(game_state, player_one_maximizer, True)
        values = [expectimax(child[0], depth - 1, True, player_one_maximizer) for child in children]
        return sum(values) / len(values) if values else 0.0


def choose_action(observation):
    """Entry point for Wrongway arena."""
    moves = observation.get("legal_moves", [])
    goal_row = observation["you"]["goal_row"]
    for m in moves:
        if m[0] == goal_row:
            return {"type": "move", "to": list(m)}

    game_state = GameState.from_observation(observation)
    player_one_maximizer = (observation["player_id"] == 0)

    children = game_state.get_all_child_states(player_one_maximizer=player_one_maximizer)
    if not children:
        return {"type": "pass"}

    best_val = -float("inf")
    best_action = children[0][1]

    depth = 2
    for child_state, action in children:
        val = expectimax(child_state, depth - 1, False, player_one_maximizer)
        if val > best_val:
            best_val = val
            best_action = action

    return best_action
