from game_state import bfs_shortest_path

def astar(game_state, is_visualize=False):
    pid = 0 if game_state.player_one else 1
    pos = game_state.player_one_pos if pid == 0 else game_state.player_two_pos
    goal_row = 0 if pid == 0 else 8
    dist, _ = bfs_shortest_path(pos, goal_row, game_state.horizontal_walls, game_state.vertical_walls)
    return dist if dist < 999 else 0

