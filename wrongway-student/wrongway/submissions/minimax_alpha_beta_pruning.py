"""Minimax with Alpha-Beta Pruning Agent for Wrongway (Quoridor 9x9).
"""

import math
from game_state import GameState
from state_evaluation_heuristic import state_evaluation_heuristic


def minimax_alpha_beta_pruning(game_state: GameState, depth, alpha, beta, maximizing_player, player_one_minimax):
    if depth == 0 or game_state.is_goal_state():
        return state_evaluation_heuristic(game_state, player_one_minimax, False)

    if maximizing_player:
        max_eval = -math.inf
        children = game_state.get_all_child_states(player_one_maximizer=player_one_minimax)
        if not children:
            return state_evaluation_heuristic(game_state, player_one_minimax, False)
        for child in children:
            ev = minimax_alpha_beta_pruning(child[0], depth - 1, alpha, beta, False, player_one_minimax)
            max_eval = max(max_eval, ev)
            alpha = max(alpha, ev)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = math.inf
        children = game_state.get_all_child_states(player_one_maximizer=not player_one_minimax)
        if not children:
            return state_evaluation_heuristic(game_state, player_one_minimax, False)
        for child in children:
            ev = minimax_alpha_beta_pruning(child[0], depth - 1, alpha, beta, True, player_one_minimax)
            min_eval = min(min_eval, ev)
            beta = min(beta, ev)
            if beta <= alpha:
                break
        return min_eval


def choose_action(observation):
    """Entry point for Wrongway arena."""
    moves = observation.get("legal_moves", [])
    goal_row = observation["you"]["goal_row"]
    for m in moves:
        if m[0] == goal_row:
            return {"type": "move", "to": list(m)}

    game_state = GameState.from_observation(observation)
    player_one_minimax = (observation["player_id"] == 0)

    children = game_state.get_all_child_states(player_one_maximizer=player_one_minimax)
    if not children:
        return {"type": "pass"}

    best_val = -math.inf
    best_action = children[0][1]
    alpha = -math.inf
    beta = math.inf

    depth = 2
    for child_state, action in children:
        val = minimax_alpha_beta_pruning(child_state, depth - 1, alpha, beta, False, player_one_minimax)
        if val > best_val:
            best_val = val
            best_action = action
        alpha = max(alpha, val)
        if beta <= alpha:
            break

    return best_action
