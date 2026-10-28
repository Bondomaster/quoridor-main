"""Wrongway / Quoridor agent. Python 3.9+, standard library only.

Bitboard BFS, legal pawn jumps, selective iterative-deepening alpha-beta.
Submit this file alone, renamed to your student ID.
"""

from time import perf_counter
from collections import deque


_ALL = (1 << 81) - 1
_BITS = tuple(1 << i for i in range(81))
_TOP = (1 << 9) - 1
_BOTTOM = _TOP << 72
_LEFT = sum(_BITS[r * 9] for r in range(9))
_RIGHT = _LEFT << 8
_OPEN = (_ALL ^ _TOP, _ALL ^ _BOTTOM, _ALL ^ _LEFT, _ALL ^ _RIGHT)
_GOALS = (_TOP, _BOTTOM)
_RESERVE = tuple(80 * n for n in range(11))
_WALLS = []
_CONFLICT = []
for _orientation in ("H", "V"):
    for _r in range(8):
        for _c in range(8):
            _a = _r * 9 + _c
            if _orientation == "H":
                _cuts = (_BITS[_a + 9] | _BITS[_a + 10],
                         _BITS[_a] | _BITS[_a + 1], 0, 0)
                _edges = ((_a, _a + 9), (_a + 1, _a + 10))
            else:
                _cuts = (0, 0, _BITS[_a + 1] | _BITS[_a + 10],
                         _BITS[_a] | _BITS[_a + 9])
                _edges = ((_a, _a + 1), (_a + 9, _a + 10))
            _WALLS.append((_orientation, _r, _c, _cuts, _edges))
for _i, (_o, _r, _c, _, _) in enumerate(_WALLS):
    _conflict = 1 << _i
    for _j, (_oo, _rr, _cc, _, _) in enumerate(_WALLS):
        if ((_o != _oo and _r == _rr and _c == _cc)
                or (_o == _oo == "H" and _r == _rr and abs(_c - _cc) <= 1)
                or (_o == _oo == "V" and _c == _cc and abs(_r - _rr) <= 1)):
            _conflict |= 1 << _j
    _CONFLICT.append(_conflict)


def _expand(bits, masks):
    up, down, left, right = masks
    return (((bits & up) >> 9) | ((bits & down) << 9)
            | ((bits & left) >> 1) | ((bits & right) << 1))


def _distance_map(masks, goal):
    distance = [99] * 81
    frontier = seen = goal
    depth = 0
    while frontier:
        cells = frontier
        while cells:
            bit = cells & -cells
            distance[bit.bit_length() - 1] = depth
            cells ^= bit
        frontier = _expand(frontier, masks) & ~seen
        seen |= frontier
        depth += 1
    return bytes(distance)


def _moves(position, opponent, masks):
    destinations = []
    bit = _BITS[position]
    for direction, delta in enumerate((-9, 9, -1, 1)):
        if not masks[direction] & bit:
            continue
        adjacent = position + delta
        if adjacent != opponent:
            destinations.append(adjacent)
        elif masks[direction] & _BITS[opponent]:
            destinations.append(opponent + delta)
        else:
            for side in ((2, 3) if direction < 2 else (0, 1)):
                if masks[side] & _BITS[opponent]:
                    destinations.append(opponent + (-9, 9, -1, 1)[side])
    return destinations


class _Board:
    __slots__ = ("key", "masks", "forbidden", "distances", "paths")

    def __init__(self, key=0, masks=_OPEN, forbidden=0):
        self.paths = {}
        self.key = key
        self.masks = masks
        self.forbidden = forbidden
        self.distances = (_distance_map(masks, _TOP), _distance_map(masks, _BOTTOM))


class _TimeUp(Exception):
    pass



_last_search = {}
_race_cache = None


def _solve_race(board, deadline):
    """Retrograde solution of the pawn-only game, including passes and cycles."""
    count = 2 * 81 * 81
    parents = [[] for _ in range(count)]
    remaining = [0] * count
    outcomes = [0] * count
    lengths = [0] * count
    pending = deque()
    for p0 in range(81):
        if p0 % 8 == 0 and perf_counter() >= deadline:
            raise _TimeUp
        for p1 in range(81):
            if p0 == p1:
                continue
            for turn in (0, 1):
                index = turn * 6561 + p0 * 81 + p1
                if p0 < 9 or p1 >= 72:
                    winner = 0 if p0 < 9 else 1
                    outcomes[index] = 1 if turn == winner else -1
                    pending.append(index)
                    continue
                own, enemy = (p0, p1) if turn == 0 else (p1, p0)
                moves = _moves(own, enemy, board.masks) + [own]
                remaining[index] = len(moves)
                for destination in moves:
                    child = (6561 + destination * 81 + p1 if turn == 0
                             else p0 * 81 + destination)
                    parents[child].append(index)
    processed = 0
    while pending:
        index = pending.popleft()
        processed += 1
        if processed % 256 == 0 and perf_counter() >= deadline:
            raise _TimeUp
        for parent in parents[index]:
            if outcomes[parent]:
                continue
            if outcomes[index] == -1:
                outcomes[parent] = 1
                lengths[parent] = lengths[index] + 1
                pending.append(parent)
            else:
                remaining[parent] -= 1
                lengths[parent] = max(lengths[parent], lengths[index] + 1)
                if remaining[parent] == 0:
                    outcomes[parent] = -1
                    pending.append(parent)
    return outcomes, lengths


def _race_action(board, positions, turn, plies, deadline):
    global _race_cache
    if _race_cache is None or _race_cache[0] != board.key:
        _race_cache = (board.key, _solve_race(board, deadline))
    outcomes, lengths = _race_cache[1]
    own, enemy = positions[turn], positions[1 - turn]
    best, chosen = None, own
    for destination in _moves(own, enemy, board.masks) + [own]:
        p0, p1 = positions
        if turn == 0:
            p0 = destination
        else:
            p1 = destination
        index = (1 - turn) * 6561 + p0 * 81 + p1
        outcome, length = -outcomes[index], lengths[index] + 1
        if length > plies:
            outcome = 0
        progress = -board.distances[turn][destination]
        rank = (outcome, -length if outcome > 0 else length if outcome < 0 else progress,
                progress, destination != own)
        if best is None or rank > best:
            best, chosen = rank, destination
    if chosen == own:
        return {"type": "pass"}
    return {"type": "move", "to": [chosen // 9, chosen % 9]}



# Search configuration is frozen into each tested submission.
_BUDGET = 0.60
_WALL_VALUE = 45
_WALL_WIDTH = 10
_PLAN_WALLS = True
_MATE = 100000
_EDGE_WALLS = {}
_NEAR_WALLS = [0] * 81
for _w, (_, _, _, _, _edges) in enumerate(_WALLS):
    for _a, _b in _edges:
        _EDGE_WALLS[_a, _b] = _EDGE_WALLS.get((_a, _b), 0) | (1 << _w)
        _EDGE_WALLS[_b, _a] = _EDGE_WALLS[_a, _b]
        _NEAR_WALLS[_a] |= 1 << _w
        _NEAR_WALLS[_b] |= 1 << _w
_ADJ = []
for _p in range(81):
    _ADJ.append(tuple((d, _p + delta) for d, delta in enumerate((-9, 9, -1, 1))
                      if _OPEN[d] & _BITS[_p]))


def _route(board, side, position):
    key = side * 81 + position
    cached = board.paths.get(key)
    if cached is not None:
        return cached
    dm = board.distances[side]
    result = 0
    p = position
    while 0 < dm[p] < 99:
        for direction, neighbor in _ADJ[p]:
            if board.masks[direction] & _BITS[p] and dm[neighbor] == dm[p] - 1:
                result |= _EDGE_WALLS[p, neighbor]
                p = neighbor
                break
        else:
            break
    board.paths[key] = result
    return result


class _Search:
    def __init__(self, deadline, max_plies, boards=None, table=None):
        self.deadline = deadline
        self.max_plies = max_plies
        self.boards = {} if boards is None else boards
        self.table = {} if table is None else table
        self.nodes = 0
        self.killers = [[-1, -1] for _ in range(80)]
        self.history = [[0] * 210 for _ in range(2)]
        self.path = set()
        self.loop_hits = 0

    def check_time(self):
        if perf_counter() >= self.deadline:
            raise _TimeUp

    def wall_board(self, board, wall):
        key = board.key | (1 << wall)
        result = self.boards.get(key)
        if result is None:
            cuts = _WALLS[wall][3]
            result = _Board(key, tuple(m & ~c for m, c in zip(board.masks, cuts)),
                            board.forbidden | _CONFLICT[wall])
            if len(self.boards) < 20000:
                self.boards[key] = result
        return result

    @staticmethod
    def winning_reply(board, positions, turn):
        p, op = positions[turn], positions[1-turn]
        if board.distances[turn][p] > 2:
            return False
        goal = 0 if turn == 0 else 8
        return any(q // 9 == goal for q in _moves(p, op, board.masks))

    @staticmethod
    def evaluate(board, positions, stocks, turn):
        p0, p1 = positions
        a, b = board.distances[0][p0], board.distances[1][p1]
        if abs(p0 // 9 - p1 // 9) + abs(p0 % 9 - p1 % 9) == 1:
            a = 1 + min((board.distances[0][p] for p in _moves(p0, p1, board.masks)), default=98)
            b = 1 + min((board.distances[1][p] for p in _moves(p1, p0, board.masks)), default=98)
        w0, w1 = stocks
        gap = w0 - w1
        value = 100 * (b - a) + _WALL_VALUE * gap + 8 * gap * abs(gap)
        value += 20 * ((w0 > 0) - (w1 > 0))
        # Race bonuses remain heuristic: only the retrograde solver certifies wins.
        margin = b - a + (0.5 if turn == 0 else -0.5)
        if not w1 and margin > 0:
            value += 1800 + min(1000, int(margin * 100))
        if not w0 and margin < 0:
            value -= 1800 + min(1000, int(-margin * 100))
        return (value if turn == 0 else -value) + 45

    def successor(self, board, positions, stocks, turn, action):
        if action == 209:
            return board, positions, stocks
        if action < 81:
            if action not in _moves(positions[turn], positions[1-turn], board.masks):
                return None
            pp = list(positions)
            pp[turn] = action
            return board, tuple(pp), stocks
        wall = action - 81
        if not stocks[turn] or board.forbidden & (1 << wall):
            return None
        child = self.wall_board(board, wall)
        if child.distances[0][positions[0]] == 99 or child.distances[1][positions[1]] == 99:
            return None
        ww = list(stocks)
        ww[turn] -= 1
        return child, positions, tuple(ww)

    def actions(self, board, positions, stocks, turn, depth, ply, preferred=-1, root_walls=None):
        self.check_time()
        own, enemy = positions[turn], positions[1-turn]
        goal = 0 if turn == 0 else 8
        pawn_moves = _moves(own, enemy, board.masks)
        for p in pawn_moves:
            if p // 9 == goal:
                return [(100000, p, board, tuple(p if t == turn else positions[t] for t in (0,1)), stocks)]
        result = []
        dm = board.distances[turn]
        for p in pawn_moves:
            pp = list(positions)
            pp[turn] = p
            pp = tuple(pp)
            score = 100 * (dm[own] - dm[p]) + self.history[turn][p]
            if self.winning_reply(board, pp, 1-turn):
                score -= 200000
            if p in self.killers[ply]:
                score += 60
            result.append((score, p, board, pp, stocks))
        # Waiting is useful at pawn contact, but generally wastes a racing tempo.
        if (not pawn_moves or (abs(own//9-enemy//9)+abs(own%9-enemy%9) <= 2)):
            score = -120
            if self.winning_reply(board, positions, 1-turn):
                score -= 200000
            result.append((score, 209, board, positions, stocks))
        if stocks[turn]:
            if root_walls is not None:
                wall_ids = root_walls
            else:
                bits = (_route(board, 1-turn, enemy) | _NEAR_WALLS[own] | _NEAR_WALLS[enemy]) & ~board.forbidden
                wall_ids = []
                while bits:
                    bit = bits & -bits
                    wall_ids.append(bit.bit_length()-1)
                    bits ^= bit
            d_own, d_enemy = dm[own], board.distances[1-turn][enemy]
            walls = []
            ww = list(stocks)
            ww[turn] -= 1
            ww = tuple(ww)
            threatened = self.winning_reply(board, positions, 1-turn)
            for i, wall in enumerate(wall_ids):
                if i % 8 == 0:
                    self.check_time()
                if board.forbidden & (1 << wall):
                    continue
                child = self.wall_board(board, wall)
                after_own = child.distances[turn][own]
                after_enemy = child.distances[1-turn][enemy]
                if after_own == 99 or after_enemy == 99:
                    continue
                gain = after_enemy - d_enemy - (after_own - d_own)
                defense = threatened and not self.winning_reply(child, positions, 1-turn)
                # shortcut: selective walls omit some neutral preparations; widen the league before changing this filter.
                if (not defense and 81+wall != preferred and not (
                        gain >= 2 or (gain >= 1 and (_PLAN_WALLS or d_enemy <= d_own+1)))):
                    continue
                score = 100 * gain - _WALL_VALUE + self.history[turn][81+wall]
                if defense:
                    score += 5000
                elif self.winning_reply(child, positions, 1-turn):
                    score -= 200000
                walls.append((score, 81+wall, child, positions, ww))
            walls.sort(key=lambda item:item[0], reverse=True)
            if ply:
                limit = _WALL_WIDTH if depth >= 3 else max(6, _WALL_WIDTH-3)
                kept = walls[:limit]
                kept.extend(x for x in walls[limit:] if x[1] == preferred or x[0] > 4000)
                walls = kept
            result.extend(walls)
        if not result:
            result.append((-1000, 209, board, positions, stocks))
        result.sort(key=lambda item:(item[1]==preferred,item[0]),reverse=True)
        return result

    def negamax(self, board, positions, stocks, turn, depth, alpha, beta, ply):
        self.nodes += 1
        if self.nodes % 32 == 0:
            self.check_time()
        if positions[1-turn] // 9 == (8 if turn == 0 else 0):
            return -_MATE + ply
        left = self.max_plies-ply
        if left <= 0:
            return 0
        if self.winning_reply(board, positions, turn):
            return _MATE-ply-1
        if depth <= 0:
            return self.evaluate(board, positions, stocks, turn)
        key = (board.key,positions,stocks,turn,min(left,64))
        # Repeated positions are heuristic leaves, never a fabricated terminal draw.
        loop_key = (board.key,positions,stocks,turn)
        if loop_key in self.path:
            self.loop_hits += 1
            return self.evaluate(board,positions,stocks,turn)-8
        original_alpha = alpha
        loop_start = self.loop_hits
        entry = self.table.get(key)
        preferred = -1
        if entry is not None:
            saved_depth,value,flag,preferred = entry
            if value > 90000:
                value -= ply
            elif value < -90000:
                value += ply
            if saved_depth >= depth:
                if flag == 0:
                    return value
                if flag == 1:
                    alpha=max(alpha,value)
                else:
                    beta=min(beta,value)
                if alpha>=beta:
                    return value
        self.path.add(loop_key)
        best,best_action = -100001,-1
        try:
            # Try the TT reply before scoring all walls: many cutoffs need no generation.
            first = None
            if preferred >= 0:
                first = self.successor(board,positions,stocks,turn,preferred)
            if first is not None:
                ch,pp,ww = first
                value = -self.negamax(ch,pp,ww,1-turn,depth-1,-beta,-alpha,ply+1)
                best,best_action=value,preferred
                alpha=max(alpha,value)
            if alpha < beta:
                options=self.actions(board,positions,stocks,turn,depth,ply,preferred)
                for _,action,ch,pp,ww in options:
                    if first is not None and action==preferred:
                        continue
                    if best_action==-1:
                        value=-self.negamax(ch,pp,ww,1-turn,depth-1,-beta,-alpha,ply+1)
                    else:
                        value=-self.negamax(ch,pp,ww,1-turn,depth-1,-alpha-1,-alpha,ply+1)
                        if alpha < value < beta:
                            value=-self.negamax(ch,pp,ww,1-turn,depth-1,-beta,-alpha,ply+1)
                    if value>best:
                        best,best_action=value,action
                    alpha=max(alpha,value)
                    if alpha>=beta:
                        break
            if alpha>=beta and best_action>=0:
                if best_action < 81:
                    self.killers[ply]=[best_action,self.killers[ply][0]]
                self.history[turn][best_action]=min(150,self.history[turn][best_action]+depth*depth)
            flag=2 if best<=original_alpha else 1 if best>=beta else 0
            stored=best+ply if best>90000 else best-ply if best< -90000 else best
            if len(self.table)<100000 and self.loop_hits == loop_start:
                self.table[key]=(depth,stored,flag,best_action)
            return best
        finally:
            self.path.remove(loop_key)


_board_cache = {}
_tt_cache = {}
_root_counts = {}
_choice_counts = {}
_previous_turn = -1


def choose_action(observation):
    global _board_cache, _tt_cache, _previous_turn
    started=perf_counter()
    moves=observation['legal_moves']
    you=observation['you']
    for p in moves:
        if p[0]==you['goal_row']:
            return {'type':'move','to':list(p)}
    turn=0 if you['goal_row']==0 else 1
    enemy=observation['opponent']
    positions=[0,0];stocks=[0,0]
    for side,pawn in ((turn,you),(1-turn,enemy)):
        positions[side]=pawn['position'][0]*9+pawn['position'][1]
        stocks[side]=pawn['walls_remaining']
    positions,stocks=tuple(positions),tuple(stocks)
    masks=list(_OPEN);used=forbidden=0
    for name,offset in (('horizontal',0),('vertical',64)):
        for r,c in observation['walls'][name]:
            wall=offset+r*8+c
            used |= 1<<wall;forbidden |= _CONFLICT[wall]
            masks=[m & ~cut for m,cut in zip(masks,_WALLS[wall][3])]
    if observation['turn']<=_previous_turn:
        _board_cache={};_tt_cache={};_root_counts.clear();_choice_counts.clear()
    _previous_turn=observation['turn']
    # Discard hypothetical walls incompatible with the real board between turns.
    if len(_board_cache)>6000:
        _board_cache={k:b for k,b in _board_cache.items() if k & used==used}
        if len(_board_cache)>14000:
            _board_cache={}
    if len(_tt_cache)>70000:
        _tt_cache={k:v for k,v in _tt_cache.items() if k[0] & used==used}
        if len(_tt_cache)>70000:
            _tt_cache={}
    board=_board_cache.get(used)
    if board is None:
        board=_Board(used,tuple(masks),forbidden)
        _board_cache[used]=board
    legal_walls=sorted((64 if w['orientation']=='V' else 0)+w['row']*8+w['col']
                       for w in observation['legal_walls'])
    legal_codes={p[0]*9+p[1] for p in moves}|{81+w for w in legal_walls}|{209}
    ordered=sorted(moves,key=lambda p:observation['distances'][p[0]][p[1]])
    best_code=ordered[0][0]*9+ordered[0][1] if ordered else 209
    own,opponent=positions[turn],positions[1-turn]
    goal=8 if turn==0 else 0
    safe=False
    for p in ordered:
        destination=p[0]*9+p[1]
        if not any(q//9==goal for q in _moves(opponent,destination,board.masks)):
            best_code=destination;safe=True;break
    if not safe:
        if not any(q//9==goal for q in _moves(opponent,own,board.masks)):
            best_code=209
        else:
            for wall in legal_walls:
                mm=tuple(m & ~cut for m,cut in zip(board.masks,_WALLS[wall][3]))
                if not any(q//9==goal for q in _moves(opponent,own,mm)):
                    best_code=81+wall;break
    remaining=observation['max_turns']-observation['turn']
    search=_Search(started+_BUDGET,remaining,_board_cache,_tt_cache)
    completed_depth=0;score=None
    root_key=(used,positions,stocks,turn)
    try:
        if not stocks[1-turn]:
            action=_race_action(board,positions,turn,remaining,search.deadline)
            outcomes,lengths=_race_cache[1]
            index=turn*6561+positions[0]*81+positions[1]
            if not stocks[turn] or (outcomes[index]==1 and lengths[index]<=remaining):
                _last_search.clear()
                _last_search.update(solver='exact_race',elapsed=perf_counter()-started)
                return action
        root_key=(used,positions,stocks,turn)
        count=_root_counts.get(root_key,0)
        _root_counts[root_key]=count+1
        options=search.actions(board,positions,stocks,turn,1,0,-1,legal_walls)
        for depth in range(1,min(32,remaining)+1):
            alpha=-100001;iteration=best_code
            scored=[]
            search.path={root_key}
            for _,action,ch,pp,ww in options:
                search.check_time()
                # Repeated waits have a small root cost, not a false repetition-draw rule.
                penalty=min(120,12*_choice_counts.get((root_key,action),0))
                if action==209:
                    penalty+=min(120,count*15)
                threshold=alpha if abs(alpha)>90000 else alpha+penalty
                value=-search.negamax(ch,pp,ww,1-turn,depth-1,-100500,-threshold,1)
                adjusted=value-penalty if abs(value)<90000 else value
                scored.append((adjusted,action,ch,pp,ww))
                if adjusted>alpha:
                    alpha,iteration=adjusted,action
            if iteration in legal_codes:
                best_code=iteration
            completed_depth,score=depth,alpha
            options=sorted(scored,key=lambda x:x[0],reverse=True)
            if abs(alpha)>90000:
                break
    except _TimeUp:
        pass
    _choice_counts[root_key,best_code]=_choice_counts.get((root_key,best_code),0)+1
    _last_search.clear()
    _last_search.update(depth=completed_depth,score=score,nodes=search.nodes,
                        boards=len(_board_cache),tt=len(_tt_cache),elapsed=perf_counter()-started)
    if best_code==209:
        return {'type':'pass'}
    if best_code<81:
        return {'type':'move','to':[best_code//9,best_code%9]}
    orientation,row,col=_WALLS[best_code-81][:3]
    return {'type':'wall','orientation':orientation,'row':row,'col':col}
