"""Simple path finding heuristic agent for Wrongway (Quoridor 9x9).
Chooses moves that minimize the distance to the goal row.
"""

from game_state import GameState


def simple_path_finding_heuristic(game_state: GameState):
    """Heuristic function based on distance to goal row."""
    if game_state.player_one:
        return 100 * abs(game_state.player_one_pos[0])
    else:
        return 100 * (abs(game_state.player_two_pos[0] - 8))


def choose_action(observation):
    """Greedy agent: chooses the move closest to the goal row."""
    moves = observation.get("legal_moves", [])
    if not moves:
        return {"type": "pass"}

    goal_row = observation["you"]["goal_row"]
    # If distances matrix is present in observation, use it; otherwise use Manhattan/row distance
    distances = observation.get("distances")
    if distances:
        reachable = [m for m in moves if distances[m[0]][m[1]] >= 0]
        candidates = reachable or moves
        best_move = min(candidates, key=lambda m: distances[m[0]][m[1]] if distances[m[0]][m[1]] >= 0 else float("inf"))
    else:
        best_move = min(moves, key=lambda m: abs(m[0] - goal_row))

    return {"type": "move", "to": list(best_move)}
