"""State evaluation heuristic for Wrongway (Quoridor 9x9).
Can be used as a standalone 1-ply greedy heuristic agent or imported by Minimax / MCTS.
"""

from game_state import GameState, bfs_shortest_path


def state_evaluation_heuristic(game_state: GameState, player_one_maximizer: bool, is_expectimax: bool = False) -> float:
    d0, _ = bfs_shortest_path(game_state.player_one_pos, 0,
                              game_state.horizontal_walls, game_state.vertical_walls)
    d1, _ = bfs_shortest_path(game_state.player_two_pos, 8,
                              game_state.horizontal_walls, game_state.vertical_walls)

    # Goal reaching bonuses
    if game_state.player_one_pos[0] == 0:
        return 10000.0 if player_one_maximizer else -10000.0
    if game_state.player_two_pos[0] == 8:
        return -10000.0 if player_one_maximizer else 10000.0

    if player_one_maximizer:
        my_dist = d0
        opp_dist = d1
        my_walls = game_state.player_one_walls_num
        opp_walls = game_state.player_two_wall_num
    else:
        my_dist = d1
        opp_dist = d0
        my_walls = game_state.player_two_wall_num
        opp_walls = game_state.player_one_walls_num

    # Main heuristic formula: opponent distance advantage + wall difference
    result = (opp_dist - my_dist) * 10.0
    result += (my_walls - opp_walls) * 2.0

    # Path inverse bonus (closer to goal gets extra weight)
    if my_dist > 0:
        result += 100.0 / my_dist
    if opp_dist > 0:
        result -= 50.0 / opp_dist

    if is_expectimax:
        result += 1.5 * opp_dist

    return float(result)


def choose_action(observation):
    """Standalone 1-ply action selector using state_evaluation_heuristic."""
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

    best_score = -float("inf")
    best_action = children[0][1]

    for child_state, action in children:
        score = state_evaluation_heuristic(child_state, player_one_maximizer, False)
        if score > best_score:
            best_score = score
            best_action = action

    return best_action
