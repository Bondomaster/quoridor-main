"""Wrongway: time-bounded adversarial search, BFS paths, and wall reserves.

Standalone submission. No Arena imports or third-party packages are needed.
"""
from collections import deque
from time import perf_counter


_BASE = tuple(tuple(rr * 9 + cc for rr, cc in
                   ((r - 1, c), (r, c - 1), (r, c + 1), (r + 1, c))
                   if 0 <= rr < 9 and 0 <= cc < 9)
              for r in range(9) for c in range(9))
_EDGES = []
_CONFLICTS = []
for _orientation in range(2):
    for _r in range(8):
        for _c in range(8):
            _a = _r * 9 + _c
            _EDGES.append(((_a, _a + 9), (_a + 1, _a + 10)) if _orientation == 0
                          else ((_a, _a + 1), (_a + 9, _a + 10)))
            _conflicts = (1 << (_r * 8 + _c)) | (1 << (64 + _r * 8 + _c))
            for _delta in (-1, 1):
                _rr = _r + (_delta if _orientation else 0)
                _cc = _c + (_delta if not _orientation else 0)
                if 0 <= _rr < 8 and 0 <= _cc < 8:
                    _conflicts |= 1 << (_orientation * 64 + _rr * 8 + _cc)
            _CONFLICTS.append(_conflicts)
_EDGES = tuple(_EDGES)
_CONFLICTS = tuple(_CONFLICTS)
_ALL_WALLS = (1 << 128) - 1
_POPCOUNT = getattr(int, 'bit_count', lambda value: bin(value).count('1'))
_BLOCKERS = [dict.fromkeys(neighbors, 0) for neighbors in _BASE]
for _wall, _pairs in enumerate(_EDGES):
    for _a, _b in _pairs:
        _BLOCKERS[_a][_b] |= 1 << _wall
        _BLOCKERS[_b][_a] |= 1 << _wall

# Losing the last defensive walls is more costly than spending one of ten.
# Values are in the same units as evaluate(): four points per path step.
_RESERVE_VALUE = (0.0, 5.6, 10.0, 13.6, 16.8, 19.6, 22.2, 24.7, 27.1, 29.4, 31.6)


def _distance_map(graph, goal):
    distance = [-1] * 81
    pending = list(range(goal * 9, goal * 9 + 9))
    for cell in pending:
        distance[cell] = 0
    for cell in pending:
        following = distance[cell] + 1
        for neighbor in graph[cell]:
            if distance[neighbor] < 0:
                distance[neighbor] = following
                pending.append(neighbor)
    return distance


def _moves(graph, pawn, opponent):
    destinations = set()
    for adjacent in graph[pawn]:
        if adjacent != opponent:
            destinations.add(adjacent)
        else:
            beyond = opponent + opponent - pawn
            if beyond in graph[opponent]:
                destinations.add(beyond)
            else:
                destinations.update(cell for cell in graph[opponent] if cell != pawn)
    return sorted(destinations)


class _Layout:
    __slots__ = ('graph', 'bits', 'forbidden', 'distances', 'profiles')

    def __init__(self, graph, bits, forbidden, goals):
        self.graph = graph
        self.bits = bits
        self.forbidden = forbidden
        self.distances = (_distance_map(graph, goals[0]), _distance_map(graph, goals[1]))
        self.profiles = None

    def wall_profiles(self):
        """Single-wall reachability and short-route bottlenecks, in parallel.

        Each bit represents one of the 128 possible walls. Intersecting masks
        along all short routes finds walls that cut EVERY such route, rather
        than merely counting neighbors near the pawn.
        """
        if self.profiles is not None:
            return self.profiles
        available = _ALL_WALLS & ~self.forbidden
        reachable = []
        critical = []
        for distance in self.distances:
            # Propagate all wall scenarios together. A scenario may use an
            # edge unless that wall blocks it. This also removes illegal
            # complete enclosures from the subsequent threat calculation.
            reach = [0] * 81
            pending = deque(cell for cell in range(81) if distance[cell] == 0)
            queued = set(pending)
            for cell in pending:
                reach[cell] = available
            while pending:
                cell = pending.popleft()
                queued.discard(cell)
                for neighbor in self.graph[cell]:
                    extra = reach[cell] & ~_BLOCKERS[cell][neighbor] & ~reach[neighbor]
                    if extra:
                        reach[neighbor] |= extra
                        if neighbor not in queued:
                            pending.append(neighbor)
                            queued.add(neighbor)
            reachable.append(reach)

            order = sorted((c for c in range(81) if distance[c] >= 0), key=distance.__getitem__)
            layers = []
            for slack in range(4):
                layer = [0] * 81
                for cell in order:
                    if distance[cell] == 0:
                        continue
                    common = available
                    for neighbor in self.graph[cell]:
                        if distance[neighbor] < 0:
                            continue
                        remaining = slack + distance[cell] - distance[neighbor] - 1
                        if remaining < 0:
                            continue
                        onward = layer[neighbor] if remaining == slack else layers[remaining][neighbor]
                        common &= _BLOCKERS[cell][neighbor] | onward
                    layer[cell] = common
                layers.append(layer)
            # C1: forced detour >= 2; C3: forced detour >= 4.
            critical.append((layers[1], layers[3]))
        self.profiles = (reachable, critical)
        return self.profiles

    def exposure(self, me, enemy):
        reachable, critical = self.wall_profiles()
        legal = reachable[0][me] & reachable[1][enemy]
        result = []
        for player, pawn in enumerate((me, enemy)):
            moderate = _POPCOUNT(critical[player][0][pawn] & legal)
            severe = _POPCOUNT(critical[player][1][pawn] & legal)
            result.append(1.5 * min(3, moderate) + 3.0 * min(2, severe))
        return result


class _BudgetExpired(Exception):
    pass


class _Search:
    def __init__(self, observation, deadline):
        self.deadline = deadline
        self.completed_depth = 0
        self.root_value = None
        self.best_response = None
        self.principal = {}
        self.goals = (observation['you']['goal_row'], observation['opponent']['goal_row'])
        graph = list(_BASE)
        bits = forbidden = 0
        for offset, name in ((0, 'horizontal'), (64, 'vertical')):
            for row, col in observation['walls'][name]:
                wall = offset + row * 8 + col
                bits |= 1 << wall
                forbidden |= _CONFLICTS[wall]
                for a, b in _EDGES[wall]:
                    graph[a] = tuple(n for n in graph[a] if n != b)
                    graph[b] = tuple(n for n in graph[b] if n != a)
        layout = _Layout(tuple(graph), bits, forbidden, self.goals)
        positions = [p['position'][0] * 9 + p['position'][1]
                     for p in (observation['you'], observation['opponent'])]
        # State: layout, our pawn, enemy pawn, our walls, enemy walls, to move.
        self.root = (layout, *positions, observation['you']['walls_remaining'],
                     observation['opponent']['walls_remaining'], 0)
        self.layouts = {bits: layout}
        self.generated = {}
        self.root_moves = [r * 9 + c for r, c in observation['legal_moves']]
        self.root_walls = {((0 if w['orientation'] == 'H' else 64) + w['row'] * 8 + w['col'])
                           for w in observation['legal_walls']}

    def check_time(self):
        if perf_counter() >= self.deadline:
            raise _BudgetExpired

    def with_wall(self, layout, wall):
        bits = layout.bits | (1 << wall)
        cached = self.layouts.get(bits)
        if cached is not None:
            return cached
        graph = list(layout.graph)
        for a, b in _EDGES[wall]:
            graph[a] = tuple(n for n in graph[a] if n != b)
            graph[b] = tuple(n for n in graph[b] if n != a)
        result = _Layout(tuple(graph), bits, layout.forbidden | _CONFLICTS[wall], self.goals)
        self.layouts[bits] = result
        return result

    def evaluate(self, state, include_risk=True):
        layout, me, enemy, mine, theirs, player = state
        d0, d1 = layout.distances
        if me // 9 == self.goals[0]:
            return 10000.0
        if enemy // 9 == self.goals[1]:
            return -10000.0
        my_moves = _moves(layout.graph, me, enemy)
        enemy_moves = _moves(layout.graph, enemy, me)
        # Exact jump options for the next step; BFS estimates the rest.
        own = 1 + min((d0[cell] for cell in my_moves), default=d0[me])
        other = 1 + min((d1[cell] for cell in enemy_moves), default=d1[enemy])
        flexibility = (sum(d0[cell] == own - 1 for cell in my_moves)
                       - sum(d1[cell] == other - 1 for cell in enemy_moves))
        own_risk, enemy_risk = layout.exposure(me, enemy) if include_risk and (mine or theirs) else (0, 0)
        risk = (enemy_risk if mine else 0) - (own_risk if theirs else 0)
        return (4.0 * (other - own) + _RESERVE_VALUE[mine] - _RESERVE_VALUE[theirs]
                + risk + 0.08 * flexibility)

    def children(self, state):
        self.check_time()
        layout, me, enemy, mine, theirs, player = state
        key = (layout.bits, me, enemy, mine, theirs, player)
        cached = self.generated.get(key)
        if cached is not None:
            return cached
        sign = 1 if player == 0 else -1
        children = []
        moves = self.root_moves if state is self.root else _moves(
            layout.graph, me if player == 0 else enemy, enemy if player == 0 else me)
        for target in moves:
            child = (layout, target if player == 0 else me,
                     enemy if player == 0 else target, mine, theirs, 1 - player)
            children.append((sign * self.evaluate(child, include_risk=False), ('move', target), child))
        walls_left = mine if player == 0 else theirs
        if walls_left:
            for wall in range(128):
                if wall % 16 == 0:
                    self.check_time()
                if layout.forbidden & (1 << wall):
                    continue
                if state is self.root and wall not in self.root_walls:
                    continue
                after = self.with_wall(layout, wall)
                if after.distances[0][me] < 0 or after.distances[1][enemy] < 0:
                    continue
                child = (after, me, enemy, mine - (player == 0),
                         theirs - (player == 1), 1 - player)
                children.append((sign * self.evaluate(child, include_risk=False), ('wall', wall), child))
        # Stable sorting prefers a move over an equally good wall.
        children.sort(key=lambda c: c[0], reverse=True)
        self.generated[key] = children
        return children

    def shortlist(self, state, children, wall_limit):
        moves = [child for child in children if child[1][0] == 'move']
        walls = [child for child in children if child[1][0] == 'wall'][:wall_limit]
        result = moves + walls
        if state is self.root and self.root[4]:
            # Include prophylactic walls that obstruct the opponent's most
            # damaging wall, even if they provide no immediate distance gain.
            layout, me, enemy = state[:3]
            threats = sorted(
                (child for child in children if child[1][0] == 'wall'),
                key=lambda child: ((child[2][0].distances[0][me] - layout.distances[0][me])
                                   - (child[2][0].distances[1][enemy] - layout.distances[1][enemy])),
                reverse=True)[:3]
            for threat in threats:
                conflict = _CONFLICTS[threat[1][1]]
                protective = next((child for child in children if child[1][0] == 'wall'
                                   and conflict & (1 << child[1][1])), None)
                if protective is not None and protective not in result:
                    result.append(protective)
        result.sort(key=lambda c: c[0], reverse=True)
        return result

    def value(self, state, depth, alpha, beta):
        self.check_time()
        score = self.evaluate(state, include_risk=False)
        if abs(score) >= 9000 or depth == 0:
            return self.evaluate(state) if depth == 0 and abs(score) < 9000 else score
        player = state[5]
        children = self.children(state)
        if not children:
            return score
        key = (state[0].bits, *state[1:], depth)
        if depth == 1:
            # All legal replies were generated. Spend the more expensive
            # bottleneck calculation on promising walls plus every move.
            finalists = self.shortlist(state, children, 8)
            sign = 1 if player == 0 else -1
            scored = [(sign * self.evaluate(child), action) for _, action, child in finalists]
            scored.sort(key=lambda c: c[0], reverse=True)
            self.principal[key] = scored[0][1]
            return sign * scored[0][0]
        selected = self.shortlist(state, children, 4)
        best = -float('inf') if player == 0 else float('inf')
        for _, action, child in selected:
            candidate = self.value(child, depth - 1, alpha, beta)
            if ((player == 0 and candidate > best)
                    or (player == 1 and candidate < best)):
                self.principal[key] = action
            if player == 0:
                best = max(best, candidate)
                alpha = max(alpha, best)
            else:
                best = min(best, candidate)
                beta = min(beta, best)
            if alpha >= beta:
                break
        return best

    def choose(self, max_depth=6):
        candidates = self.children(self.root)
        selected = self.shortlist(self.root, candidates, 12)
        selected = [(self.evaluate(child), action, child) for _, action, child in selected]
        selected.sort(key=lambda c: c[0], reverse=True)
        best_action = selected[0][1]
        self.completed_depth = 1
        self.root_value = selected[0][0]
        # Use a fully completed iteration only. If time runs out, the previous
        # result remains valid; a partial search must not bias the first move.
        for depth in range(2, max_depth + 1):
            scored = []
            try:
                for ordering, action, child in selected:
                    score = self.value(child, depth - 1, -float('inf'), float('inf'))
                    scored.append((score, ordering, action, child))
            except _BudgetExpired:
                break
            scored.sort(key=lambda c: c[0], reverse=True)
            best_action = scored[0][2]
            self.completed_depth = depth
            self.root_value = scored[0][0]
            child = scored[0][3]
            self.best_response = self.principal.get((child[0].bits, *child[1:], depth - 1))
            if abs(scored[0][0]) >= 9000:
                break
            # Retain alternatives for deeper search instead of committing to
            # just one short route that the opponent could seal later.
            selected = [(score, action, child) for score, _, action, child in scored[:6]]
        return best_action


def _action_dict(action):
    if action is None:
        return None
    kind, value = action
    if kind == 'pass':
        return {'type': 'pass'}
    if kind == 'move':
        return {'type': 'move', 'to': [value // 9, value % 9]}
    return {'type': 'wall', 'orientation': 'H' if value < 64 else 'V',
            'row': (value % 64) // 8, 'col': value % 8}


def analyze_exhaustive(observation, include_table=True):
    """Evaluate EVERY legal action and EVERY legal opponent reply, depth 2.

    No shortlist, alpha-beta cutoff, or search deadline is used. This is an
    untimed web/analysis mode; it can exceed the Arena's one-second limit.
    Pass is included when permitted by the caller (Arena allows it, the web
    UI does not). A win/draw ends that branch immediately by the game rules.
    """
    started = perf_counter()
    search = _Search(observation, float('inf'))
    allow_pass = observation.get('allow_pass', True)

    def actions(state):
        children = list(search.children(state))
        if allow_pass:
            child = (*state[:5], 1 - state[5])
            sign = 1 if state[5] == 0 else -1
            children.append((sign * search.evaluate(child, include_risk=False), ('pass', 0), child))
        return children

    roots = actions(search.root)
    remaining = observation.get('max_turns', 500) - observation.get('turn', 0)
    best_action = ('pass', 0)
    best_value = -float('inf')
    best_reply = None
    reply_count = leaf_count = terminal_roots = 0
    table = []
    move_count = sum(action[0] == 'move' for _, action, _ in roots)
    wall_count = sum(action[0] == 'wall' for _, action, _ in roots)

    for _, action, child in roots:
        score = search.evaluate(child, include_risk=False)
        worst_reply = None
        replies_seen = 0
        if abs(score) >= 9000 or remaining <= 1:
            if abs(score) < 9000:
                score = 0.0
            terminal_roots += 1
            leaf_count += 1
        else:
            replies = actions(child)
            score = float('inf')
            for _, reply, leaf in replies:
                # Intentionally no break even after a forced loss: coverage
                # statistics mean all legal replies were actually evaluated.
                value = search.evaluate(leaf)
                if remaining <= 2 and abs(value) < 9000:
                    value = 0.0
                replies_seen += 1
                leaf_count += 1
                if value < score:
                    score, worst_reply = value, reply
            if not replies:
                score = search.evaluate(child)
                leaf_count += 1
        reply_count += replies_seen
        if score > best_value:
            best_value, best_action, best_reply = score, action, worst_reply
        if include_table:
            table.append({'action': _action_dict(action), 'score': score,
                          'reply_count': replies_seen, 'worst_reply': _action_dict(worst_reply)})

    report = {
        'action': _action_dict(best_action), 'predicted_reply': _action_dict(best_reply),
        'search_mode': 'exhaustive', 'complete': True, 'depth': 2,
        'score': best_value, 'seconds': perf_counter() - started,
        'root_actions': len(roots), 'root_evaluated': len(roots),
        'root_moves': move_count, 'root_walls': wall_count,
        'root_passes': len(roots) - move_count - wall_count,
        'reply_actions': reply_count, 'leaf_evaluations': leaf_count,
        'terminal_roots': terminal_roots,
    }
    if include_table:
        report['action_table'] = table
    return report


def analyze_action(observation, time_limit=0.60, max_depth=6):
    """Return the action and a predicted adversarial reply for inspection.

    A reply is the strongest one found within the search, not a claim that the
    actual opponent will choose it. Depth >= 2 includes an opponent response.
    """
    start = perf_counter()
    moves = observation['legal_moves']
    if not moves:
        return {'action': {'type': 'pass'}, 'predicted_reply': None, 'depth': 0}
    goal = observation['you']['goal_row']
    for move in moves:
        if move[0] == goal:
            return {'action': {'type': 'move', 'to': list(move)},
                    'predicted_reply': None, 'depth': 0, 'score': 10000.0}
    distances = observation['distances']
    fallback = min(moves, key=lambda p: distances[p[0]][p[1]]
                   if distances[p[0]][p[1]] >= 0 else 1000)
    search = _Search(observation, start + time_limit)
    try:
        action = search.choose(max_depth)
    except _BudgetExpired:
        return {'action': {'type': 'move', 'to': list(fallback)},
                'predicted_reply': None, 'depth': 0}
    return {'action': _action_dict(action),
            'predicted_reply': _action_dict(search.best_response),
            'depth': search.completed_depth, 'score': search.root_value,
            'seconds': perf_counter() - start,
            'wall_exposure': dict(zip(('you', 'opponent'),
                                     search.root[0].exposure(search.root[1], search.root[2])))}


class _AlphaSearch(_Search):
    """Full legal action set; discard branches only with alpha-beta bounds.

    Moving replies are tested before generating walls, so a disproven branch
    does not pay for 128 hypothetical board layouts. Wall layouts, distance
    maps and transpositions are shared within a decision.
    """
    WIN = 100000.0

    def __init__(self, observation, deadline, recent=()):
        super().__init__(observation, deadline)
        self.tt = {}
        self.hints = {}
        self.nodes = 0
        self.cutoffs = 0
        self.recent = recent
        self.hard_model = observation.get('opponent_model') == 'wrongway-hard'
        self.remaining = observation.get('max_turns', 500) - observation.get('turn', 0)

    def evaluate(self, state, include_risk=False):
        layout, me, enemy, mine, theirs, player = state
        d0, d1 = layout.distances
        if me // 9 == self.goals[0]:
            return self.WIN
        if enemy // 9 == self.goals[1]:
            return -self.WIN
        own, other = d0[me], d1[enemy]
        # A surplus of walls is useful only while there is enough opponent
        # route left to delay. This prevents hoarding ten walls in a lost race.
        own_reserve = 300.0 * mine * other / (mine + other + 2.0)
        enemy_reserve = 300.0 * theirs * own / (theirs + own + 2.0)
        score = (100.0 * (other - own) + own_reserve - enemy_reserve
                 + (43.0 if player == 0 else -43.0))
        if mine == 0 and theirs == 0:
            a = 1 + min(d0[p] for p in _moves(layout.graph, me, enemy))
            b = 1 + min(d1[p] for p in _moves(layout.graph, enemy, me))
            ahead = a <= b if player == 0 else a < b
            score += 700.0 if ahead else -700.0
        return score

    def move_children(self, state):
        layout, me, enemy, mine, theirs, player = state
        targets = self.root_moves if state is self.root else _moves(
            layout.graph, me if player == 0 else enemy, enemy if player == 0 else me)
        distance = layout.distances[player]
        if self.hard_model and player == 1 and any(distance[p] < distance[enemy] for p in targets):
            targets = [p for p in targets if distance[p] <= distance[enemy]]
        for target in sorted(targets, key=distance.__getitem__):
            yield ('move', target), (layout, target if player == 0 else me,
                                    enemy if player == 0 else target, mine, theirs, 1 - player)

    def ordered(self, state):
        key = (state[0].bits, *state[1:])
        hint = self.hints.get(key)
        moves = list(self.move_children(state))
        if hint is not None and hint[0] == 'move':
            found = next((c for c in moves if c[0] == hint), None)
            if found is not None:
                yield found
        for action, child in moves:
            if action != hint:
                yield action, child
        # This part is never reached if a moving reply already closes the
        # alpha-beta window. All legal walls remain available otherwise.
        walls = [(action, child) for _, action, child in self.children(state) if action[0] == 'wall']
        if self.hard_model and state[5] == 1:
            layout, me, enemy = state[:3]
            walls = [(action, child) for action, child in walls
                     if child[0].distances[0][me] - layout.distances[0][me]
                     > child[0].distances[1][enemy] - layout.distances[1][enemy]]
        if hint is not None and hint[0] == 'wall':
            found = next((c for c in walls if c[0] == hint), None)
            if found is not None:
                yield found
        for action, child in walls:
            if action != hint:
                yield action, child

    def ab(self, state, depth, alpha, beta, ply):
        self.check_time()
        self.nodes += 1
        value = self.evaluate(state)
        if abs(value) >= self.WIN / 2:
            return value - ply if value > 0 else value + ply
        if ply >= self.remaining:
            return 0.0
        player = state[5]
        pawn, other = (state[1], state[2]) if player == 0 else (state[2], state[1])
        targets = _moves(state[0].graph, pawn, other)
        if any(cell // 9 == self.goals[player] for cell in targets):
            return self.WIN - ply - 1 if player == 0 else -self.WIN + ply + 1
        if depth == 0:
            return value
        key = (state[0].bits, *state[1:], depth)
        original_alpha, original_beta = alpha, beta
        cached = self.tt.get(key)
        if cached is not None:
            score, bound = cached
            if bound == 'exact':
                return score
            if bound == 'lower':
                alpha = max(alpha, score)
            else:
                beta = min(beta, score)
            if alpha >= beta:
                return score
        best = -float('inf') if player == 0 else float('inf')
        chosen = None
        for action, child in self.ordered(state):
            score = self.ab(child, depth - 1, alpha, beta, ply + 1)
            if chosen is None or (score > best if player == 0 else score < best):
                best, chosen = score, action
            if player == 0:
                alpha = max(alpha, best)
            else:
                beta = min(beta, best)
            if alpha >= beta:
                self.cutoffs += 1
                break
        if chosen is None:
            best = value
        else:
            self.hints[(state[0].bits, *state[1:])] = chosen
        bound = 'upper' if best <= original_alpha else 'lower' if best >= original_beta else 'exact'
        self.tt[key] = (best, bound)
        return best

    def root_bonus(self, action, child):
        if action[0] != 'move':
            return 0.0
        target = action[1]
        current = self.root[1]
        distance = self.root[0].distances[0]
        bonus = 40.0 if distance[target] < distance[current] else 0.0
        if target in self.recent:
            bonus -= 60.0 / (self.recent.index(target) + 1)
        return bonus

    def choose_alpha(self, max_depth):
        roots = [(ordering + self.root_bonus(action, child), action, child)
                 for ordering, action, child in self.children(self.root)]
        roots.sort(key=lambda c: c[0], reverse=True)
        best_action = roots[0][1]
        self.completed_depth = 0
        root_count = len(roots)
        for depth in range(1, max_depth + 1):
            self.tt.clear()  # Mate scores depend on ply within this iteration.
            scored = []
            alpha = -float('inf')
            try:
                for ordering, action, child in roots:
                    bonus = self.root_bonus(action, child)
                    score = self.ab(child, depth - 1, alpha - bonus, float('inf'), 1) + bonus
                    scored.append((score, ordering, action, child))
                    alpha = max(alpha, score)
            except _BudgetExpired:
                break
            # A fail-low alpha-beta bound may equal the incumbent's score.
            # Keep the first exact incumbent on ties, never replace it using
            # a secondary heuristic from a branch that was only bounded.
            scored.sort(key=lambda c: c[0], reverse=True)
            best_action = scored[0][2]
            self.completed_depth = depth
            self.root_value = scored[0][0]
            child = scored[0][3]
            self.best_response = self.hints.get((child[0].bits, *child[1:]))
            roots = [(score, action, child) for score, _, action, child in scored]
            if abs(scored[0][0]) >= self.WIN / 2:
                break
        return best_action, root_count


def analyze_alpha(observation, time_limit=0.65, max_depth=12, recent=()):
    started = perf_counter()
    moves = observation['legal_moves']
    for move in moves:
        if move[0] == observation['you']['goal_row']:
            return {'action': {'type': 'move', 'to': list(move)}, 'predicted_reply': None,
                    'depth': 0, 'score': _AlphaSearch.WIN, 'search_mode': 'bounded', 'seconds': 0.0}
    if not moves:
        return {'action': {'type': 'pass'}, 'predicted_reply': None, 'depth': 0, 'search_mode': 'bounded'}
    distance = observation['distances']
    fallback = min(moves, key=lambda p: distance[p[0]][p[1]] if distance[p[0]][p[1]] >= 0 else 1000)
    search = _AlphaSearch(observation, started + time_limit, recent)
    try:
        action, root_count = search.choose_alpha(max_depth)
    except _BudgetExpired:
        return {'action': {'type': 'move', 'to': list(fallback)}, 'predicted_reply': None,
                'depth': 0, 'search_mode': 'bounded', 'seconds': perf_counter() - started}
    return {'action': _action_dict(action), 'predicted_reply': _action_dict(search.best_response),
            'depth': search.completed_depth, 'score': search.root_value,
            'search_mode': 'bounded', 'algorithm': 'alpha-beta', 'root_actions': root_count,
            'nodes': search.nodes, 'cutoffs': search.cutoffs, 'seconds': perf_counter() - started}


_RECENT_POSITIONS = []


def choose_action(observation):
    # Arena does not supply this optional web-only setting, so submissions
    # keep the usual 0.60 s budget. The untimed website can opt into more.
    if observation.get('search_mode') == 'exhaustive':
        report = analyze_exhaustive(observation, include_table=False)
    else:
        budget = max(0.05, min(10.0, float(observation.get('agent_time_limit', 0.65))))
        current = observation['you']['position'][0] * 9 + observation['you']['position'][1]
        if observation.get('turn', 0) <= 1:
            _RECENT_POSITIONS.clear()
        recent = [p for p in reversed(_RECENT_POSITIONS) if p != current][:4]
        report = analyze_alpha(observation, time_limit=budget, recent=recent)
        _RECENT_POSITIONS.append(current)
        del _RECENT_POSITIONS[:-8]
    action = dict(report['action'])
    if observation.get('_bridge_analysis'):
        # Private bridge telemetry is never added to normal Arena actions.
        action['_analysis'] = {key: value for key, value in report.items() if key != 'action'}
    return action
