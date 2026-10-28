"""Shared GameState implementation for Wrongway (Quoridor 9x9).
Used by Minimax, Alpha-Beta, MCTS, Expectimax, and Heuristics.
"""

from collections import deque
import math
from typing import List, Tuple, Set, Optional, Dict, Any

BOARD_SIZE = 9
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def edge_open(c1: Tuple[int, int], c2: Tuple[int, int],
              h_walls: Set[Tuple[int, int]], v_walls: Set[Tuple[int, int]]) -> bool:
    r1, col1 = c1
    r2, col2 = c2
    if not (0 <= r1 < BOARD_SIZE and 0 <= col1 < BOARD_SIZE and
            0 <= r2 < BOARD_SIZE and 0 <= col2 < BOARD_SIZE):
        return False
    if r1 != r2:
        r = min(r1, r2)
        return (r, col1) not in h_walls and (r, col1 - 1) not in h_walls
    col = min(col1, col2)
    return (r1, col) not in v_walls and (r1 - 1, col) not in v_walls


def bfs_shortest_path(start: Tuple[int, int], goal_row: int,
                      h_walls: Set[Tuple[int, int]], v_walls: Set[Tuple[int, int]]) -> Tuple[int, List[Tuple[int, int]]]:
    """Returns (distance, path_as_list_of_cells). If no path, returns (inf, [])."""
    if start[0] == goal_row:
        return 0, [start]
    visited = {start: None}
    queue = deque([start])
    while queue:
        curr = queue.popleft()
        if curr[0] == goal_row:
            # Reconstruct path
            path = []
            node = curr
            while node is not None:
                path.append(node)
                node = visited[node]
            path.reverse()
            return len(path) - 1, path
        r, c = curr
        for dr, dc in DIRECTIONS:
            nxt = (r + dr, c + dc)
            if nxt not in visited and edge_open(curr, nxt, h_walls, v_walls):
                visited[nxt] = curr
                queue.append(nxt)
    return 999, []


class GameState:
    def __init__(self,
                 player_one_pos: Tuple[int, int] = (8, 4),
                 player_two_pos: Tuple[int, int] = (0, 4),
                 player_one_walls_num: int = 10,
                 player_two_wall_num: int = 10,
                 horizontal_walls: Optional[Set[Tuple[int, int]]] = None,
                 vertical_walls: Optional[Set[Tuple[int, int]]] = None,
                 current_player: int = 0):
        self.player_one_pos = player_one_pos
        self.player_two_pos = player_two_pos
        self.player_one_walls_num = player_one_walls_num
        self.player_two_wall_num = player_two_wall_num
        self.horizontal_walls = frozenset(horizontal_walls) if horizontal_walls else frozenset()
        self.vertical_walls = frozenset(vertical_walls) if vertical_walls else frozenset()
        self.current_player = current_player
        self.player_one = (current_player == 0)
        self.value = 0

    @classmethod
    def from_observation(cls, obs: Dict[str, Any]) -> "GameState":
        player_id = obs["player_id"]
        you = obs["you"]
        opp = obs["opponent"]
        walls = obs["walls"]
        h_set = {tuple(w) for w in walls.get("horizontal", [])}
        v_set = {tuple(w) for w in walls.get("vertical", [])}

        if player_id == 0:
            p1_pos = tuple(you["position"])
            p1_walls = you["walls_remaining"]
            p2_pos = tuple(opp["position"])
            p2_walls = opp["walls_remaining"]
        else:
            p2_pos = tuple(you["position"])
            p2_walls = you["walls_remaining"]
            p1_pos = tuple(opp["position"])
            p1_walls = opp["walls_remaining"]

        return cls(
            player_one_pos=p1_pos,
            player_two_pos=p2_pos,
            player_one_walls_num=p1_walls,
            player_two_wall_num=p2_walls,
            horizontal_walls=h_set,
            vertical_walls=v_set,
            current_player=player_id
        )

    def is_goal_state(self) -> bool:
        return self.player_one_pos[0] == 0 or self.player_two_pos[0] == 8

    def game_result(self, player_one_maximizer: bool) -> int:
        if player_one_maximizer:
            if self.player_one_pos[0] == 0:
                return 1
            if self.player_two_pos[0] == 8:
                return -1
        else:
            if self.player_two_pos[0] == 8:
                return 1
            if self.player_one_pos[0] == 0:
                return -1
        return 0

    def get_legal_moves(self, player_id: int) -> List[Tuple[int, int]]:
        pos = self.player_one_pos if player_id == 0 else self.player_two_pos
        opp = self.player_two_pos if player_id == 0 else self.player_one_pos
        moves = []
        r, c = pos
        for dr, dc in DIRECTIONS:
            adj = (r + dr, c + dc)
            if not edge_open(pos, adj, self.horizontal_walls, self.vertical_walls):
                continue
            if adj != opp:
                moves.append(adj)
                continue
            # Jump over opponent
            straight = (adj[0] + dr, adj[1] + dc)
            if edge_open(adj, straight, self.horizontal_walls, self.vertical_walls):
                moves.append(straight)
            else:
                for sdr, sdc in ((dc, dr), (-dc, -dr)):
                    diag = (adj[0] + sdr, adj[1] + sdc)
                    if edge_open(adj, diag, self.horizontal_walls, self.vertical_walls):
                        moves.append(diag)
        return moves

    def is_wall_legal(self, orientation: str, r: int, c: int) -> bool:
        if not (0 <= r < BOARD_SIZE - 1 and 0 <= c < BOARD_SIZE - 1):
            return False
        if orientation == "H":
            if (r, c) in self.horizontal_walls or (r, c - 1) in self.horizontal_walls or (r, c + 1) in self.horizontal_walls:
                return False
            if (r, c) in self.vertical_walls:
                return False
            new_h = self.horizontal_walls | {(r, c)}
            new_v = self.vertical_walls
        else:
            if (r, c) in self.vertical_walls or (r - 1, c) in self.vertical_walls or (r + 1, c) in self.vertical_walls:
                return False
            if (r, c) in self.horizontal_walls:
                return False
            new_h = self.horizontal_walls
            new_v = self.vertical_walls | {(r, c)}

        # Check path exists for both players
        d0, _ = bfs_shortest_path(self.player_one_pos, 0, new_h, new_v)
        if d0 >= 999:
            return False
        d1, _ = bfs_shortest_path(self.player_two_pos, 8, new_h, new_v)
        if d1 >= 999:
            return False
        return True

    def get_candidate_walls(self, player_id: int) -> List[Dict[str, Any]]:
        walls_rem = self.player_one_walls_num if player_id == 0 else self.player_two_wall_num
        if walls_rem <= 0:
            return []

        opp_pos = self.player_two_pos if player_id == 0 else self.player_one_pos
        opp_goal = 8 if player_id == 0 else 0
        _, path = bfs_shortest_path(opp_pos, opp_goal, self.horizontal_walls, self.vertical_walls)
        if len(path) < 2:
            return []

        candidates = set()
        # Look at the first few steps of the opponent's path
        for i in range(min(4, len(path) - 1)):
            c1, c2 = path[i], path[i + 1]
            if c1[0] != c2[0]:
                # Vertical step between rows -> blocked by H wall
                r = min(c1[0], c2[0])
                c = c1[1]
                for col_anchor in (c - 1, c):
                    if 0 <= r < BOARD_SIZE - 1 and 0 <= col_anchor < BOARD_SIZE - 1:
                        candidates.add(("H", r, col_anchor))
            else:
                # Horizontal step between cols -> blocked by V wall
                r = c1[0]
                c = min(c1[1], c2[1])
                for row_anchor in (r - 1, r):
                    if 0 <= row_anchor < BOARD_SIZE - 1 and 0 <= c < BOARD_SIZE - 1:
                        candidates.add(("V", row_anchor, c))

        valid_walls = []
        for orient, r, c in candidates:
            if self.is_wall_legal(orient, r, c):
                valid_walls.append({"type": "wall", "orientation": orient, "row": r, "col": c})
        return valid_walls

    def get_all_child_states(self, player_one_maximizer: bool = True, include_state: bool = True):
        """Returns list of (child_state, action) pairs."""
        pid = 0 if player_one_maximizer else 1
        children = []

        # 1. Pawn Moves
        legal_moves = self.get_legal_moves(pid)
        for move in legal_moves:
            if pid == 0:
                child = GameState(
                    player_one_pos=move,
                    player_two_pos=self.player_two_pos,
                    player_one_walls_num=self.player_one_walls_num,
                    player_two_wall_num=self.player_two_wall_num,
                    horizontal_walls=set(self.horizontal_walls),
                    vertical_walls=set(self.vertical_walls),
                    current_player=1
                )
            else:
                child = GameState(
                    player_one_pos=self.player_one_pos,
                    player_two_pos=move,
                    player_one_walls_num=self.player_one_walls_num,
                    player_two_wall_num=self.player_two_wall_num,
                    horizontal_walls=set(self.horizontal_walls),
                    vertical_walls=set(self.vertical_walls),
                    current_player=0
                )
            action = {"type": "move", "to": list(move)}
            children.append((child, action))

        # 2. Targeted Wall Placements
        walls = self.get_candidate_walls(pid)
        for wall in walls:
            o, r, c = wall["orientation"], wall["row"], wall["col"]
            new_h = set(self.horizontal_walls)
            new_v = set(self.vertical_walls)
            if o == "H":
                new_h.add((r, c))
            else:
                new_v.add((r, c))

            child = GameState(
                player_one_pos=self.player_one_pos,
                player_two_pos=self.player_two_pos,
                player_one_walls_num=self.player_one_walls_num - (1 if pid == 0 else 0),
                player_two_wall_num=self.player_two_wall_num - (1 if pid == 1 else 0),
                horizontal_walls=new_h,
                vertical_walls=new_v,
                current_player=1 - pid
            )
            children.append((child, wall))

        return children

