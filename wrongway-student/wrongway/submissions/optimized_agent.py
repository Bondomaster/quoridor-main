"""Independent strategy for the Wrongway Arena submission protocol."""
from collections import deque
from time import perf_counter

N = 9
CELLS = 81
GOALS = (0, 8)

# Each possible wall blocks two undirected board edges. Wall ids 0..63 are H,
# 64..127 are V, matching the arena's anchor coordinates.
BASE = tuple(tuple((r + dr) * 9 + c + dc
                   for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0))
                   if 0 <= r + dr < 9 and 0 <= c + dc < 9)
             for r in range(9) for c in range(9))
WALL_EDGES = []
CONFLICTS = []
EDGE_WALLS = [dict() for _ in range(CELLS)]
for vertical in range(2):
    for row in range(8):
        for col in range(8):
            wid = vertical * 64 + row * 8 + col
            a = row * 9 + col
            pairs = (((a, a + 1), (a + 9, a + 10)) if vertical
                     else ((a, a + 9), (a + 1, a + 10)))
            WALL_EDGES.append(pairs)
            conflict = (1 << wid)
            for dr, dc in (((-1, 0), (1, 0)) if vertical
                           else ((0, -1), (0, 1))):
                rr, cc = row + dr, col + dc
                if 0 <= rr < 8 and 0 <= cc < 8:
                    conflict |= 1 << (vertical * 64 + rr * 8 + cc)
            CONFLICTS.append(conflict)
            for x, y in pairs:
                EDGE_WALLS[x][y] = EDGE_WALLS[x].get(y, 0) | (1 << wid)
                EDGE_WALLS[y][x] = EDGE_WALLS[y].get(x, 0) | (1 << wid)
WALL_EDGES = tuple(WALL_EDGES)
CONFLICTS = tuple(CONFLICTS)
ALL_WALLS = (1 << 128) - 1


def _distances(walls, goal):
    """Distance-to-goal map using reverse BFS over currently open edges."""
    d = [-1] * CELLS
    q = deque()
    for c in range(9):
        p = goal * 9 + c
        d[p] = 0
        q.append(p)
    while q:
        p = q.popleft()
        nd = d[p] + 1
        for nxt in BASE[p]:
            if d[nxt] < 0 and not (walls & EDGE_WALLS[p][nxt]):
                d[nxt] = nd
                q.append(nxt)
    return d


def _moves(p, other, walls):
    out = set()
    for q in BASE[p]:
        if walls & EDGE_WALLS[p][q]:
            continue
        if q != other:
            out.add(q)
            continue
        # Straight jump when the square behind the adjacent pawn is reachable.
        behind = 2 * q - p
        if behind in BASE[q] and not (walls & EDGE_WALLS[q][behind]):
            out.add(behind)
        else:
            # At a boundary or blocked jump, sidestep around the opponent.
            for side in BASE[q]:
                if side != p and not (walls & EDGE_WALLS[q][side]):
                    out.add(side)
    return out


def _wall_id(w):
    return (64 if w['orientation'] == 'V' else 0) + w['row'] * 8 + w['col']


def _wall_dict(wid):
    return {'type': 'wall', 'orientation': 'V' if wid >= 64 else 'H',
            'row': (wid % 64) // 8, 'col': wid % 8}


def _legal_wall_ids(walls, d0, d1, p0, p1, deadline):
    """Generate valid hypothetical walls; reject overlaps and path closures."""
    used = walls
    out = []
    for wid in range(128):
        if perf_counter() > deadline:
            break
        bit = 1 << wid
        if used & CONFLICTS[wid]:
            continue
        candidate = used | bit
        a = _distances(candidate, 0)
        if a[p0] < 0:
            continue
        b = _distances(candidate, 8)
        if b[p1] < 0:
            continue
        out.append((wid, a[p0] - d0[p0], b[p1] - d1[p1]))
    return out


def _score(pos0, pos1, walls, turn, stocks, remaining):
    d0 = _distances(walls, 0)
    d1 = _distances(walls, 8)
    a, b = d0[pos0], d1[pos1]
    if a < 0:
        return -100000.0
    if b < 0:
        return 100000.0
    if pos0 // 9 == 0:
        return 100000.0
    if pos1 // 9 == 8:
        return -100000.0
    # A wall placement costs a tempo. Distances capture route quality; stock
    # value decays as either pawn approaches the finish.
    race = 100.0 * (b - a)
    reserve = min(8, a + b) * 1.8 * (stocks[0] - stocks[1])
    initiative = 7.0 if turn == 0 else -7.0
    if remaining <= 2:
        reserve *= 0.25
    return race + reserve + initiative


def _wall_candidates(walls, pos0, pos1, stocks, turn, limit, deadline):
    if stocks[turn] <= 0:
        return []
    d0 = _distances(walls, 0)
    d1 = _distances(walls, 8)
    candidates = _legal_wall_ids(walls, d0, d1, pos0, pos1, deadline)
    # Walls that lengthen the opponent route while preserving ours rank first.
    # Keep a few defensive/neutral choices to avoid a brittle greedy cutoff.
    ownp, oppp = (pos0, pos1) if turn == 0 else (pos1, pos0)
    ownm, oppm = (d0, d1) if turn == 0 else (d1, d0)
    ranked = []
    for wid, gain0, gain1 in candidates:
        if perf_counter() > deadline:
            break
        gain_own = gain0 if turn == 0 else gain1
        gain_opp = gain1 if turn == 0 else gain0
        if gain_own > 0:
            continue
        value = 100 * (gain_opp - gain_own)
        # Slightly reward walls on the opponent's current shortest corridor.
        if gain_opp > 0 and oppm[oppp] <= ownm[ownp] + 3:
            value += 5
        ranked.append((value, wid))
    ranked.sort(reverse=True)
    return [wid for _, wid in ranked[:limit]]


def _actions(pos0, pos1, walls, turn, stocks, root_moves, root_walls, deadline, wall_limit=5):
    p, other = (pos0, pos1) if turn == 0 else (pos1, pos0)
    moves = set(root_moves) if root_moves is not None else _moves(p, other, walls)
    actions = [('m', x) for x in moves]
    if stocks[turn] > 0:
        if root_walls is not None:
            walls_to_try = [_wall_id(w) for w in root_walls]
        else:
            d0, d1 = _distances(walls, 0), _distances(walls, 8)
            walls_to_try = _wall_candidates(walls, pos0, pos1, stocks, turn,
                                            wall_limit, deadline)
        actions.extend(('w', w) for w in walls_to_try)
    return actions


def _apply(pos0, pos1, walls, stocks, turn, action):
    kind, value = action
    ns = list(stocks)
    if kind == 'm':
        if turn == 0:
            pos0 = value
        else:
            pos1 = value
    else:
        walls |= 1 << value
        ns[turn] -= 1
    return pos0, pos1, walls, tuple(ns), 1 - turn


def choose_action(observation):
    moves = observation.get('legal_moves', [])
    if not moves:
        return {'type': 'pass'}
    me = observation['player_id']
    you = observation['you']
    opp = observation['opponent']
    own = you['position'][0] * 9 + you['position'][1]
    enemy = opp['position'][0] * 9 + opp['position'][1]
    goal = you['goal_row']
    for move in moves:
        if move[0] == goal:
            return {'type': 'move', 'to': list(move)}

    started = perf_counter()
    deadline = started + 0.68
    walls = 0
    for orientation, key in (('H', 'horizontal'), ('V', 'vertical')):
        for row, col in observation['walls'][key]:
            walls |= 1 << ((64 if orientation == 'V' else 0) + row * 8 + col)
    pos = [0, 0]
    pos[me] = own
    pos[1 - me] = enemy
    stocks = [0, 0]
    stocks[me] = you['walls_remaining']
    stocks[1 - me] = opp['walls_remaining']
    pos0, pos1 = pos
    turn = me

    # Score every arena-validated root choice by its immediate race effect, then
    # use a one-reply adversarial check on the strongest distinct candidates.
    root_actions = [('m', m[0] * 9 + m[1]) for m in moves]
    if stocks[me] > 0:
        root_actions.extend(('w', _wall_id(w)) for w in observation.get('legal_walls', []))
    immediate = []
    for action in root_actions:
        if perf_counter() > deadline:
            break
        state = _apply(pos0, pos1, walls, tuple(stocks), turn, action)
        val = _score(state[0], state[1], state[2], state[4], state[3],
                     observation.get('max_turns', 500) - observation.get('turn', 0) - 1)
        if action[0] == 'w':
            val += -100.0 if me == 0 else 100.0
        immediate.append((val, action, state))
    if not immediate:
        return {'type': 'move', 'to': list(moves[0])}
    immediate.sort(key=lambda x: x[0], reverse=(me == 0))

    best_value, best_action = immediate[0][0], immediate[0][1]
    # Reply search: moves are exhaustive; only evaluate a small set of walls
    # selected by their route impact, preserving the 1-second arena budget.
    shortlist = immediate[:min(12, len(immediate))]
    for root_value, action, state in shortlist:
        if perf_counter() > deadline:
            break
        p0, p1, wb, sb, next_turn = state
        if (p0 // 9 == 0) or (p1 // 9 == 8):
            value = _score(p0, p1, wb, next_turn, sb, 0)
        else:
            rep_moves = _moves(p1, p0, wb) if next_turn == 1 else _moves(p0, p1, wb)
            rep_walls = _wall_candidates(wb, p0, p1, sb, next_turn, 3, deadline)
            replies = [('m', x) for x in rep_moves] + [('w', x) for x in rep_walls]
            value = float('inf') if me == 0 else -float('inf')
            for reply in replies:
                if perf_counter() > deadline:
                    break
                leaf = _apply(p0, p1, wb, sb, next_turn, reply)
                score = _score(leaf[0], leaf[1], leaf[2], leaf[4], leaf[3],
                               observation.get('max_turns', 500) - observation.get('turn', 0) - 2)
                if reply[0] == 'w':
                    score += -100.0 if next_turn == 0 else 100.0
                value = min(value, score) if me == 0 else max(value, score)
            if value == float('inf') or value == -float('inf'):
                value = root_value
        if (value > best_value if me == 0 else value < best_value):
            best_value, best_action = value, action

    kind, value = best_action
    if kind == 'm':
        return {'type': 'move', 'to': [value // 9, value % 9]}
    return _wall_dict(value)
