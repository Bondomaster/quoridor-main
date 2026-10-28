"""Wrongway: exact rules, selective alpha-beta, and a solved pawn-only endgame.

Submission for student 25127400. Python 3.9+, standard library only.
Cells use row * 9 + column; a wall removes two undirected graph edges.
Search is deliberately selective: it does not claim to solve the full game.
"""

from functools import lru_cache
from heapq import heappop, heappush
from time import perf_counter

SIZE = 9
CELLS = 81
UNREACHABLE = 81
FULL = (1 << CELLS) - 1
TOP = (1 << SIZE) - 1
LEFT = sum(1 << (SIZE * row) for row in range(SIZE))
BASE_MASKS = (FULL ^ TOP, FULL ^ (TOP << 72), FULL ^ LEFT, FULL ^ (LEFT << 8))
DELTAS = (-9, 9, -1, 1)
BITS = tuple(1 << cell for cell in range(CELLS))
ALL_WALLS = (1 << 128) - 1
WALL_OFFSET = 81
PASS = 209
MATE = 100_000
INFINITY = 1_000_000
TIME_LIMIT = 0.65  # Leave time for JSON/IPC and operating-system scheduling.
RESERVE = tuple(90 * n for n in range(11))
PROGRESS_BIAS = 140


def _geometry():
    """Precompute wall deletions, conflicts, and the walls cutting each edge."""
    removals, conflicts, edges = [], [], {}
    for wall in range(128):
        vertical, anchor = divmod(wall, 64)
        row, col = divmod(anchor, 8)
        cell = row * 9 + col
        mask = BITS[cell] | BITS[cell + (9 if vertical else 1)]
        removals.append((0, 0, mask << 1, mask) if vertical else (mask << 9, mask, 0, 0))
        conflict = (1 << wall) | (1 << (wall ^ 64))
        for delta in (-1, 1):
            r, c = (row + delta, col) if vertical else (row, col + delta)
            if 0 <= r < 8 and 0 <= c < 8:
                conflict |= 1 << (vertical * 64 + r * 8 + c)
        conflicts.append(conflict)
        pairs = (
            ((cell, cell + 1), (cell + 9, cell + 10))
            if vertical
            else ((cell, cell + 9), (cell + 1, cell + 10))
        )
        for a, b in pairs:
            edges[a, b] = edges[b, a] = edges.get((a, b), 0) | (1 << wall)
    return tuple(removals), tuple(conflicts), edges


REMOVALS, CONFLICTS, EDGE_WALLS = _geometry()


def add_wall(masks: tuple, wall: int) -> tuple:
    return tuple(mask & ~removed for mask, removed in zip(masks, REMOVALS[wall]))


def bit_distance(masks: tuple, start: int, goal: int) -> int:
    """Exact BFS on the wall graph; ignores pawns, returning 81 if disconnected."""
    up, down, left, right = masks
    target = TOP << (goal * 9)
    frontier = visited = BITS[start]
    distance = 0
    while frontier:
        if frontier & target:
            return distance
        frontier = (
            ((frontier & up) >> 9)
            | ((frontier & down) << 9)
            | ((frontier & left) >> 1)
            | ((frontier & right) << 1)
        ) & ~visited
        visited |= frontier
        distance += 1
    return UNREACHABLE


def pawn_moves(masks: tuple, pawn: int, opponent: int) -> list:
    """All legal moves, including straight jumps and *forced* diagonals."""
    result = []
    for direction, delta in enumerate(DELTAS):
        if not masks[direction] & BITS[pawn]:
            continue
        adjacent = pawn + delta
        if adjacent != opponent:
            result.append(adjacent)
        elif masks[direction] & BITS[opponent]:
            result.append(opponent + delta)
        else:
            for side in (2, 3) if direction < 2 else (0, 1):
                if masks[side] & BITS[opponent]:
                    destination = opponent + DELTAS[side]
                    if destination not in result:
                        result.append(destination)
    return result


class Layout:
    """Immutable wall geometry with goal maps and shortest-path certificates."""

    __slots__ = ("horizontal", "vertical", "masks", "available", "graph", "maps", "cuts")

    def __init__(self, horizontal, vertical):
        self.horizontal, self.vertical = horizontal, vertical
        masks, available = BASE_MASKS, ALL_WALLS
        occupied = horizontal | (vertical << 64)
        while occupied:
            bit = occupied & -occupied
            wall = bit.bit_length() - 1
            occupied ^= bit
            masks = add_wall(masks, wall)
            available &= ~CONFLICTS[wall]
        self.masks, self.available = masks, available
        self.graph = tuple(
            tuple(cell + delta for d, delta in enumerate(DELTAS) if masks[d] & BITS[cell])
            for cell in range(CELLS)
        )
        maps, cuts = [], []
        for goal in (0, 8):
            distances, cutters = [UNREACHABLE] * CELLS, [0] * CELLS
            queue = list(range(goal * 9, goal * 9 + 9))
            for cell in queue:
                distances[cell] = 0
            for cell in queue:
                following = distances[cell] + 1
                for neighbor in self.graph[cell]:
                    if distances[neighbor] == UNREACHABLE:
                        distances[neighbor] = following
                        cutters[neighbor] = cutters[cell] | EDGE_WALLS[cell, neighbor]
                        queue.append(neighbor)
            maps.append(tuple(distances))
            cuts.append(tuple(cutters))
        self.maps, self.cuts = tuple(maps), tuple(cuts)

    def place(self, wall):
        if wall < 64:
            return get_layout(self.horizontal | (1 << wall), self.vertical)
        return get_layout(self.horizontal, self.vertical | (1 << (wall - 64)))

    def wall_distances(self, wall, pawn, opponent, goal):
        """Validate both routes, cheaply reusing an untouched shortest path.

        This certificate only saves BFS calls; it never prunes a wall from search.
        Goal index 0 means row 0; goal index 1 means row 8.
        """
        bit = 1 << wall
        masks = None
        distances = []
        for cell, index in ((pawn, goal), (opponent, 1 - goal)):
            if self.cuts[index][cell] & bit:
                if masks is None:
                    masks = add_wall(self.masks, wall)
                distance = bit_distance(masks, cell, index * 8)
            else:
                distance = self.maps[index][cell]
            if distance == UNREACHABLE:
                return None
            distances.append(distance)
        return tuple(distances)


@lru_cache(maxsize=1024)
def get_layout(horizontal, vertical):
    return Layout(horizontal, vertical)


def read_position(observation: dict) -> tuple:
    horizontal = sum(1 << (r * 8 + c) for r, c in observation["walls"]["horizontal"])
    vertical = sum(1 << (r * 8 + c) for r, c in observation["walls"]["vertical"])
    you, other = observation["you"], observation["opponent"]
    pawn = you["position"][0] * 9 + you["position"][1]
    opponent = other["position"][0] * 9 + other["position"][1]
    return (
        get_layout(horizontal, vertical),
        pawn,
        opponent,
        you["walls_remaining"],
        other["walls_remaining"],
        you["goal_row"] // 8,
        observation["max_turns"] - observation["turn"],
    )


def action_dict(action: int) -> dict:
    if action < WALL_OFFSET:
        return {"type": "move", "to": list(divmod(action, 9))}
    if action == PASS:
        return {"type": "pass"}
    vertical, anchor = divmod(action - WALL_OFFSET, 64)
    row, col = divmod(anchor, 8)
    return {"type": "wall", "orientation": "V" if vertical else "H", "row": row, "col": col}


class Deadline(Exception):
    """Abort an incomplete iteration without publishing its values or TT entry."""


class Endgame:
    """Retrograde WDL/DTM for one fixed layout, including jumps and pass.

    Distances count *plies*: a winner minimizes time, a loser maximizes delay.
    Reverse edges come from real forward moves, not an assumed symmetric graph.
    """

    def __init__(self, layout, goal, check_time):
        count = CELLS * CELLS * 2
        self.outcome = [0] * count
        self.distance = [0] * count
        predecessors = [[] for _ in range(count)]
        pending, degrees, longest = [], [0] * count, [0] * count
        goals = (goal * 8, (1 - goal) * 8)
        for pawn in range(CELLS):
            check_time()
            for opponent in range(CELLS):
                if pawn == opponent:
                    continue
                positions = (pawn, opponent)
                winners = [i for i in (0, 1) if positions[i] // 9 == goals[i]]
                if len(winners) > 1:
                    continue  # Cannot arise in a real game, which stops at the first goal.
                base = (pawn * CELLS + opponent) * 2
                for side in (0, 1):
                    state = base + side
                    if winners:
                        self.outcome[state] = 1 if winners[0] == side else -1
                        heappush(pending, (0, state))
                        continue
                    moves = pawn_moves(layout.masks, positions[side], positions[1 - side])
                    moves.append(positions[side])  # Pass is a real action.
                    degrees[state] = len(moves)
                    for move in moves:
                        p, q = (move, opponent) if side == 0 else (pawn, move)
                        child = (p * CELLS + q) * 2 + 1 - side
                        predecessors[child].append(state)
        processed = 0
        while pending:
            distance, child = heappop(pending)
            processed += 1
            if processed % 128 == 0:
                check_time()
            for parent in predecessors[child]:
                if self.outcome[parent]:
                    continue
                if self.outcome[child] == -1:
                    self.outcome[parent] = 1
                    self.distance[parent] = distance + 1
                    heappush(pending, (distance + 1, parent))
                else:
                    degrees[parent] -= 1
                    longest[parent] = max(longest[parent], distance + 1)
                    if degrees[parent] == 0:
                        self.outcome[parent] = -1
                        self.distance[parent] = longest[parent]
                        heappush(pending, (longest[parent], parent))

    def value(self, pawn, opponent, remaining):
        state = (pawn * CELLS + opponent) * 2
        outcome, distance = self.outcome[state], self.distance[state]
        if not outcome or distance > remaining:
            return 0
        return outcome * (MATE + remaining - distance)


# Only a few layouts need a complete endgame table; discard earlier games safely.
_ENDGAMES = {}


class Search:
    """Iterative alpha-beta over all pawn moves and a diverse wall shortlist."""

    def __init__(self, time_limit=TIME_LIMIT, max_depth=12, node_limit=None):
        self.deadline = perf_counter() + time_limit
        self.max_depth, self.node_limit = max_depth, node_limit
        self.nodes = self.completed_depth = self.cutoffs = 0
        self.table, self.orders = {}, {}
        self.endgame = None
        self.root_goal = 0
        self.value = 0

    def check_time(self):
        if perf_counter() >= self.deadline or (
            self.node_limit is not None and self.nodes >= self.node_limit
        ):
            raise Deadline

    @staticmethod
    def evaluate(layout, pawn, opponent, walls, enemy_walls, goal):
        own_map, other_map = layout.maps[goal], layout.maps[1 - goal]
        own_moves = pawn_moves(layout.masks, pawn, opponent)
        other_moves = pawn_moves(layout.masks, opponent, pawn)
        own_distance = 1 + min((own_map[x] for x in own_moves), default=own_map[pawn])
        other_distance = 1 + min((other_map[x] for x in other_moves), default=other_map[opponent])
        own_options = sum(own_map[x] < own_map[pawn] for x in own_moves)
        other_options = sum(other_map[x] < other_map[opponent] for x in other_moves)
        return (
            100 * (other_distance - own_distance)
            + 45
            + RESERVE[walls]
            - RESERVE[enemy_walls]
            + 8 * (own_options - other_options)
        )

    def actions(self, layout, pawn, opponent, walls, enemy_walls, goal, root=False):
        key = (layout.horizontal, layout.vertical, pawn, opponent, walls, enemy_walls, goal, root)
        if key in self.orders:
            return self.orders[key]
        own_map, other_map = layout.maps[goal], layout.maps[1 - goal]
        own_distance, other_distance = own_map[pawn], other_map[opponent]
        candidates = []
        for move in pawn_moves(layout.masks, pawn, opponent):
            value = (
                MATE
                if move // 9 == goal * 8
                else self.evaluate(layout, move, opponent, walls, enemy_walls, goal) - 90
            )
            candidates.append((value, move))
        candidates.append((100 * (other_distance - own_distance) - 110, PASS))
        wall_candidates = []
        threat, threat_gain = None, 1
        available = layout.available if walls else 0
        while available:
            self.check_time()
            bit = available & -available
            wall = bit.bit_length() - 1
            available ^= bit
            distances = layout.wall_distances(wall, pawn, opponent, goal)
            if distances is None:
                continue
            own_after, other_after = distances
            harm, gain = own_after - own_distance, other_after - other_distance
            score = 100 * (other_after - own_after) + RESERVE[walls - 1] - RESERVE[enemy_walls]
            row, col = divmod(wall % 64, 8)
            proximity = abs(opponent // 9 - row) + abs(opponent % 9 - col)
            touches_path = bool(layout.cuts[1 - goal][opponent] & bit)
            setup = 8 * touches_path - proximity
            wall_candidates.append((score, WALL_OFFSET + wall, gain, harm, setup))
            if harm - gain > threat_gain:
                threat, threat_gain = wall, harm - gain
        wall_candidates.sort(reverse=True)
        count = 18 if root else 6
        selected = {entry[1] for entry in wall_candidates[:count]}
        # Keep some quiet preparation and preemptive defense, even at zero BFS gain.
        quiet = [entry for entry in wall_candidates if entry[2] == entry[3] == 0]
        quiet.sort(key=lambda entry: (entry[4], entry[0]), reverse=True)
        selected.update(entry[1] for entry in quiet[: (3 if root else 1)])
        if threat is not None and enemy_walls:
            defensive = [
                entry
                for entry in wall_candidates
                if CONFLICTS[entry[1] - WALL_OFFSET] & (1 << threat) and entry[3] == 0
            ]
            selected.update(entry[1] for entry in defensive[:2])
        all_actions = candidates + [(entry[0], entry[1]) for entry in wall_candidates]
        all_actions.sort(reverse=True)
        shortlist = [
            action
            for _, action in all_actions
            if action < WALL_OFFSET or action == PASS or action in selected
        ]
        result = shortlist, [action for _, action in all_actions]
        self.orders[key] = result
        return result

    def child(self, layout, pawn, opponent, walls, enemy_walls, goal, remaining, action):
        if action < WALL_OFFSET:
            pawn = action
        elif action != PASS:
            layout = layout.place(action - WALL_OFFSET)
            walls -= 1
        return layout, opponent, pawn, enemy_walls, walls, 1 - goal, remaining - 1

    @staticmethod
    def root_cost(position, action):
        """A heuristic cost for hiding future wall threats by waiting or retreating.

        Only the root decision uses this preference; interior TT values do not.
        Contact tactics, short horizons, and solved pawn-only play remain unbiased.
        """
        layout, pawn, opponent, walls, enemy_walls, goal, remaining = position
        separation = abs(pawn // 9 - opponent // 9) + abs(pawn % 9 - opponent % 9)
        if not (walls or enemy_walls) or remaining <= 12 or separation <= 1:
            return 0
        if action >= WALL_OFFSET and action != PASS:
            return 0
        destination = pawn if action == PASS else action
        change = layout.maps[goal][destination] - layout.maps[goal][pawn]
        return PROGRESS_BIAS * max(0, 1 + change)

    def negamax(
        self, layout, pawn, opponent, walls, enemy_walls, goal, remaining, depth, alpha, beta
    ):
        self.nodes += 1
        self.check_time()
        if opponent // 9 == (1 - goal) * 8:
            return -MATE - remaining
        if pawn // 9 == goal * 8:
            return MATE + remaining
        if remaining <= 0:
            return 0
        moves = pawn_moves(layout.masks, pawn, opponent)
        if any(move // 9 == goal * 8 for move in moves):
            return MATE + remaining - 1
        if not walls and not enemy_walls and self.endgame is not None:
            if goal == self.root_goal:
                return self.endgame.value(pawn, opponent, remaining)
            # The table is labeled by root player, but now the other side moves.
            state = (opponent * CELLS + pawn) * 2 + 1
            result, distance = self.endgame.outcome[state], self.endgame.distance[state]
            return result * (MATE + remaining - distance) if distance <= remaining else 0
        if depth <= 0:
            return self.evaluate(layout, pawn, opponent, walls, enemy_walls, goal)
        key = (
            layout.horizontal,
            layout.vertical,
            pawn,
            opponent,
            walls,
            enemy_walls,
            goal,
            remaining,
        )
        original_alpha, original_beta = alpha, beta
        cached = self.table.get(key)
        preferred = None
        if cached:
            cached_depth, score, bound, preferred = cached
            if cached_depth >= depth:
                if bound == 0:
                    return score
                if bound == 1:
                    alpha = max(alpha, score)
                else:
                    beta = min(beta, score)
                if alpha >= beta:
                    return score
        actions, _ = self.actions(layout, pawn, opponent, walls, enemy_walls, goal)
        if preferred in actions:
            actions = [preferred] + [action for action in actions if action != preferred]
        best, best_action = -INFINITY, actions[0]
        for action in actions:
            child = self.child(layout, pawn, opponent, walls, enemy_walls, goal, remaining, action)
            score = -self.negamax(*child, depth - 1, -beta, -alpha)
            if score > best:
                best, best_action = score, action
            alpha = max(alpha, score)
            if alpha >= beta:
                self.cutoffs += 1
                break
        bound = -1 if best <= original_alpha else (1 if best >= original_beta else 0)
        self.table[key] = (depth, best, bound, best_action)
        return best

    def run(self, position, fallback):
        layout, pawn, opponent, walls, enemy_walls, goal, remaining = position
        self.root_goal = goal
        best_action = fallback
        try:
            if not walls and not enemy_walls:
                key = (layout.horizontal, layout.vertical, goal)
                self.endgame = _ENDGAMES.get(key)
                if self.endgame is None:
                    table = Endgame(layout, goal, self.check_time)
                    if len(_ENDGAMES) >= 4:
                        _ENDGAMES.clear()
                    _ENDGAMES[key] = self.endgame = table
            shortlist, all_actions = self.actions(*position[:6], root=True)
            previous = {}
            for depth in range(1, self.max_depth + 1):
                actions = all_actions if depth == 1 else shortlist
                if not walls and enemy_walls:
                    # With no barricades left, repeatedly waiting for an imagined
                    # future wall can consume the whole lead. Away from pawn
                    # contact, commit to the best forward route. The complete
                    # pawn-only table still includes every move and pass.
                    separation = abs(pawn // 9 - opponent // 9) + abs(pawn % 9 - opponent % 9)
                    forward = [
                        action
                        for action in actions
                        if action < WALL_OFFSET
                        and layout.maps[goal][action] < layout.maps[goal][pawn]
                    ]
                    if separation > 2 and forward:
                        actions = forward
                actions = sorted(
                    actions, key=lambda action: previous.get(action, -INFINITY), reverse=True
                )
                alpha, iteration_best, iteration_action = -INFINITY, -INFINITY, best_action
                scores = {}
                for action in actions:
                    self.check_time()
                    child = self.child(*position, action)
                    cost = self.root_cost(position, action)
                    score = -self.negamax(*child, depth - 1, -INFINITY, -alpha - cost) - cost
                    scores[action] = score
                    if score > iteration_best:
                        iteration_best, iteration_action = score, action
                    alpha = max(alpha, score)
                best_action, self.value = iteration_action, iteration_best
                self.completed_depth, previous = depth, scores
                if abs(iteration_best) >= MATE - 1000 or self.endgame is not None:
                    break
        except Deadline:
            pass  # Only completed iterations replace the previously safe decision.
        return best_action


def choose_action(observation: dict) -> dict:
    """Return a legal action under the one-second Arena protocol."""
    if observation.get("done"):
        return {"type": "pass"}
    moves = observation["legal_moves"]
    goal = observation["you"]["goal_row"]
    for move in moves:
        if move[0] == goal:
            return {"type": "move", "to": list(move)}
    distances = observation["distances"]
    fallback = min(moves, key=lambda move: distances[move[0]][move[1]], default=None)
    fallback_id = fallback[0] * 9 + fallback[1] if fallback is not None else PASS
    search = Search()
    action = action_dict(search.run(read_position(observation), fallback_id))
    # A final boundary check protects the live game from an internal-model mistake.
    if action["type"] == "move" and action["to"] in moves:
        return action
    if action["type"] == "wall":
        compact = {key: value for key, value in action.items() if key != "type"}
        if compact in observation["legal_walls"]:
            return action
    if action["type"] == "pass":
        return action
    return action_dict(fallback_id)
