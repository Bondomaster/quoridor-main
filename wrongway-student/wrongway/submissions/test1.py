"""Wrongway agent - test1.py
Cải tiến so với base:
  - Race Solver: giải ngược (retrograde) thế cờ chỉ có quân (không tường) để
    biết chắc ai THẮNG/THUA/HOÀ khi chơi hoàn hảo → "nhìn thấy thế thắng".
  - Threat Detection: phát hiện đối thủ sắp về đích, ưu tiên chặn ngay.
  - Forced-Win Extraction: khi search trả về score >= WIN, chọn đường thắng
    nhanh nhất (ply nhỏ nhất).
  - Giữ nguyên toàn bộ Negamax + Alpha-Beta + Iterative Deepening + TT.
"""

import random
import time

INF_D   = 99
WIN     = 10_000.0
NEAR_WIN = 900.0
TIME_BUDGET = 0.65   # giây (giới hạn arena = 1 giây)
MAX_DEPTH   = 10
CAND_EDGES  = 7

# Trọng số hàm lượng giá
W1_DIST  = 2.0
W2_WALL  = 1.60
W3_TEMPO = 0.45
W4_MOB   = 0.15

# ── bitboard constants ────────────────────────────────────────────────────────
BIT  = [1 << i for i in range(82)]
FULL = (1 << 81) - 1
_ROW = [((1 << 9) - 1) << (9 * r) for r in range(9)]
_COL = [sum(1 << (9 * r + c) for r in range(9)) for c in range(9)]
BASE_D = FULL & ~_ROW[8]
BASE_U = FULL & ~_ROW[0]
BASE_R = FULL & ~_COL[8]
BASE_L = FULL & ~_COL[0]
GM = (_ROW[0], _ROW[8])

WALLS = []
for _o in (0, 1):
    for _r in range(8):
        for _c in range(8):
            _a = 9 * _r + _c
            if _o == 0:
                _cA   = BIT[_a]     | BIT[_a + 1]
                _cB   = BIT[_a + 9] | BIT[_a + 10]
                _same = BIT[_a] | BIT[_a + 1] | (BIT[_a - 1] if _c > 0 else 0)
            else:
                _cA   = BIT[_a]     | BIT[_a + 9]
                _cB   = BIT[_a + 1] | BIT[_a + 10]
                _same = BIT[_a] | BIT[_a + 9] | (BIT[_a - 9] if _r > 0 else 0)
            WALLS.append((_o, _r, _c, _a, FULL & ~_cA, FULL & ~_cB,
                          BIT[_a], _same, BIT[_a]))


class _Timeout(Exception):
    pass


_deadline   = 0.0
_cache      = {}
_cand_cache = {}
_tt         = {}

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

_rng   = random.Random(25127275)
Z_ME   = [_rng.getrandbits(64) for _ in range(81)]
Z_OPP  = [_rng.getrandbits(64) for _ in range(81)]
Z_WALL = [_rng.getrandbits(64) for _ in range(128)]
Z_WME  = [_rng.getrandbits(64) for _ in range(11)]
Z_WOPP = [_rng.getrandbits(64) for _ in range(11)]
Z_SIDE = [_rng.getrandbits(64) for _ in range(2)]


# ── khoảng cách / đường đi ───────────────────────────────────────────────────
def dist(pos, gi, hm, vm, D, U, R, L):
    key = (hm, vm, pos, gi)
    v = _cache.get(key)
    if v is not None:
        return v
    S   = GM[gi]
    bit = BIT[pos]
    if S & bit:
        v = 0
    else:
        k = 0
        while True:
            N = S | ((S >> 9) & D) | ((S << 9) & U) | ((S >> 1) & R) | ((S << 1) & L)
            k += 1
            if N & bit:
                v = k
                break
            if N == S:
                v = INF_D
                break
            S = N
    _cache[key] = v
    return v


def path_edges(pos, gi, D, U, R, L):
    S      = GM[gi]
    bit    = BIT[pos]
    layers = [S]
    while not (S & bit):
        N = S | ((S >> 9) & D) | ((S << 9) & U) | ((S >> 1) & R) | ((S << 1) & L)
        if N == S:
            return []
        layers.append(N)
        S = N
    d     = len(layers) - 1
    edges = []
    cur   = pos
    while d > 0:
        T = layers[d - 1]
        b = BIT[cur]
        if (D & b) and (T & BIT[cur + 9]):
            nxt = cur + 9
        elif (U & b) and (T & BIT[cur - 9]):
            nxt = cur - 9
        elif (R & b) and (T & BIT[cur + 1]):
            nxt = cur + 1
        else:
            nxt = cur - 1
        edges.append((cur, nxt))
        cur = nxt
        d  -= 1
    return edges


def wall_cands(opp, go, hm, vm, D, U, R, L):
    key = (hm, vm, opp, go)
    v   = _cand_cache.get(key)
    if v is not None:
        return v
    out  = []
    seen = set()

    for x, y in path_edges(opp, go, D, U, R, L)[:CAND_EDGES]:
        if x - y == 9 or y - x == 9:
            r = min(x, y) // 9
            c = x % 9
            for cc in (c, c - 1):
                if 0 <= cc <= 7 and r <= 7:
                    wi = r * 8 + cc
                    if wi not in seen:
                        seen.add(wi)
                        out.append(wi)
        else:
            r = x // 9
            c = min(x, y) % 9
            for rr in (r, r - 1):
                if 0 <= rr <= 7 and c <= 7:
                    wi = 64 + rr * 8 + c
                    if wi not in seen:
                        seen.add(wi)
                        out.append(wi)

    opp_r, opp_c = opp // 9, opp % 9
    for wi in range(128):
        if wi in seen:
            continue
        o, r, c, a, _, _, selfm, _, _ = WALLS[wi]
        if abs(r - opp_r) + abs(c - opp_c) > 3:
            continue
        if o == 0:
            if (c > 0 and (hm & BIT[a - 1])) or (c < 7 and (hm & BIT[a + 1])):
                seen.add(wi)
                out.append(wi)
        else:
            if (r > 0 and (vm & BIT[a - 9])) or (r < 7 and (vm & BIT[a + 9])):
                seen.add(wi)
                out.append(wi)

    _cand_cache[key] = out
    return out


# ── sinh nước đi ─────────────────────────────────────────────────────────────
def pawn_moves(me, opp, D, U, R, L):
    out = []
    bm  = BIT[me]
    if D & bm:
        a = me + 9
        if a != opp:
            out.append(a)
        elif D & BIT[a]:
            out.append(a + 9)
        else:
            if L & BIT[a]: out.append(a - 1)
            if R & BIT[a]: out.append(a + 1)
    if U & bm:
        a = me - 9
        if a != opp:
            out.append(a)
        elif U & BIT[a]:
            out.append(a - 9)
        else:
            if L & BIT[a]: out.append(a - 1)
            if R & BIT[a]: out.append(a + 1)
    if R & bm:
        a = me + 1
        if a != opp:
            out.append(a)
        elif R & BIT[a]:
            out.append(a + 1)
        else:
            if U & BIT[a]: out.append(a - 9)
            if D & BIT[a]: out.append(a + 9)
    if L & bm:
        a = me - 1
        if a != opp:
            out.append(a)
        elif L & BIT[a]:
            out.append(a - 1)
        else:
            if U & BIT[a]: out.append(a - 9)
            if D & BIT[a]: out.append(a + 9)
    return out


# ── Retrograde Race Solver ────────────────────────────────────────────────────
# Kết quả: WIN(1), LOSE(-1), DRAW(0) cho mỗi trạng thái (me, opp, turn)
# khi không còn tường nào – giải ngược từ các terminal state.
_race_cache: dict | None = None   # (D,U,R,L) -> (outcomes, dtm)

def _solve_race(D, U, R, L):
    """Retrograde quy hoạch động lùi cho thế cờ chỉ có quân (không tường).
    Trả về (outcomes, dtm):
      outcomes[gi*81*81 + me*81 + opp]: 1=thắng, -1=thua, 0=hoà
      dtm     [gi*81*81 + me*81 + opp]: số bước đến kết thúc (DTM)
    gi=0 nghĩa là bên cần về hàng 0 đang tới lượt, gi=1 về hàng 8.
    """
    N       = 81
    SIZE    = 2 * N * N
    outcomes = [0]  * SIZE
    dtm      = [0]  * SIZE
    pending  = []

    # Terminal: me đã về đích → WIN cho bên đó
    for gi in (0, 1):
        goal_row = gi * 8   # gi=0 → hàng 0; gi=1 → hàng 8
        for me in range(N):
            if me // 9 == goal_row:
                for opp in range(N):
                    if opp == me:
                        continue
                    idx = gi * N * N + me * N + opp
                    outcomes[idx] = 1    # me ở đích → thắng
                    dtm[idx]      = 0
                    pending.append(idx)

    # Xây parent map
    parents = [[] for _ in range(SIZE)]
    child_count = [0] * SIZE
    for gi in (0, 1):
        goal_row = gi * 8
        for me in range(N):
            for opp in range(N):
                if me == opp:
                    continue
                if me // 9 == goal_row:
                    continue  # terminal, đã xử lý
                idx   = gi * N * N + me * N + opp
                moves = pawn_moves(me, opp, D, U, R, L)
                if not moves:
                    # không đi được → hoà / thua (coi như pass)
                    child_count[idx] = 1
                else:
                    child_count[idx] = len(moves)
                for dest in (moves or [me]):   # pass = đứng yên
                    # con của (gi, me, opp) chạy sang (1-gi, opp, dest)
                    child_idx = (1 - gi) * N * N + opp * N + dest
                    parents[child_idx].append(idx)

    # BFS ngược
    from collections import deque
    q = deque(pending)
    visited = set(pending)
    while q:
        idx = q.popleft()
        res = outcomes[idx]
        d   = dtm[idx]
        for par in parents[idx]:
            if par in visited:
                continue
            if outcomes[par] != 0:
                continue
            if res == -1:
                # con là thua → cha thắng
                outcomes[par] = 1
                dtm[par]      = d + 1
                visited.add(par)
                q.append(par)
            else:
                # con là thắng → giảm số lựa chọn của cha
                child_count[par] -= 1
                if dtm[par] == 0:
                    dtm[par] = d + 1
                else:
                    dtm[par] = max(dtm[par], d + 1)
                if child_count[par] <= 0:
                    outcomes[par] = -1
                    visited.add(par)
                    q.append(par)

    return outcomes, dtm


def _race_result(me, opp, gi, D, U, R, L):
    """Tra bảng race: (outcome, dtm). outcome: 1=thắng, -1=thua, 0=hoà."""
    global _race_cache
    key = (D, U, R, L)
    if _race_cache is None or _race_cache[0] != key:
        outcomes, dtm = _solve_race(D, U, R, L)
        _race_cache = (key, outcomes, dtm)
    outcomes = _race_cache[1]
    dtm      = _race_cache[2]
    idx = gi * 81 * 81 + me * 81 + opp
    return outcomes[idx], dtm[idx]


# ── Threat Detection ──────────────────────────────────────────────────────────
def _opponent_threat(opp, gi, hm, vm, D, U, R, L):
    """True nếu đối thủ có thể về đích trong <= 2 bước (bỏ qua quân cờ của ta)."""
    opp_goal = 1 - gi
    d = dist(opp, opp_goal, hm, vm, D, U, R, L)
    return d <= 2


# ── hàm lượng giá ─────────────────────────────────────────────────────────────
def evaluate(me, opp, wme, wopp, gi, hm, vm, D, U, R, L):
    dm = dist(me, gi,     hm, vm, D, U, R, L)
    do = dist(opp, 1 - gi, hm, vm, D, U, R, L)

    if dm == 0:     return WIN
    if do <= 1:     return -NEAR_WIN
    if dm <= 2 and dm < do:
        return NEAR_WIN - dm * 20.0

    # Race bonus khi không còn tường
    if wme == 0 and wopp == 0:
        outcome, dtm_v = _race_result(me, opp, gi, D, U, R, L)
        if outcome == 1:
            return WIN - dtm_v          # thắng → điểm gần WIN, dtm nhỏ = tốt hơn
        if outcome == -1:
            return -(WIN - dtm_v)       # thua

    bm     = BIT[me]
    me_mob = ((D & bm) != 0) + ((U & bm) != 0) + ((R & bm) != 0) + ((L & bm) != 0)
    bo     = BIT[opp]
    opp_mob = ((D & bo) != 0) + ((U & bo) != 0) + ((R & bo) != 0) + ((L & bo) != 0)

    score = (W1_DIST  * (do - dm) +
             W2_WALL  * (wme - wopp) +
             W3_TEMPO +
             W4_MOB   * (me_mob - opp_mob))
    return score


# ── sinh con ──────────────────────────────────────────────────────────────────
def kids_pawn(me, opp, wme, wopp, gi, hm, vm, D, U, R, L, moves):
    out  = []
    goal = GM[gi]
    for dest in moves:
        if goal & BIT[dest]:
            out.append((WIN, 0, dest, hm, vm, D, U, R, L, dest, wme))
            continue
        e = evaluate(dest, opp, wme, wopp, gi, hm, vm, D, U, R, L)
        out.append((e, 0, dest, hm, vm, D, U, R, L, dest, wme))
    return out


def kids_wall(me, opp, wme, wopp, gi, hm, vm, D, U, R, L):
    go  = 1 - gi
    do0 = dist(opp, go, hm, vm, D, U, R, L)
    dm0 = dist(me,  gi, hm, vm, D, U, R, L)

    # Nếu đối thủ sắp về đích (<=3 bước), luôn xét tường bất kể gain
    urgent   = do0 <= 3
    min_gain = 1 if urgent else (2 if (dm0 < do0 and do0 > 3) else 1)

    for wi in wall_cands(opp, go, hm, vm, D, U, R, L):
        o, r, c, a, nA, nB, selfm, same, other = WALLS[wi]
        if o == 0:
            if (hm & same) or (vm & other):
                continue
            hm2, vm2 = hm | selfm, vm
            D2, U2, R2, L2 = D & nA, U & nB, R, L
        else:
            if (vm & same) or (hm & other):
                continue
            vm2, hm2 = vm | selfm, hm
            R2, L2, D2, U2 = R & nA, L & nB, D, U

        do = dist(opp, go, hm2, vm2, D2, U2, R2, L2)
        if do >= INF_D or do <= do0:
            continue
        dm = dist(me, gi, hm2, vm2, D2, U2, R2, L2)
        if dm >= INF_D:
            continue

        gain = (do - do0) - (dm - dm0)
        if not urgent and gain < min_gain and do0 > 2:
            continue

        e = evaluate(me, opp, wme - 1, wopp, gi, hm2, vm2, D2, U2, R2, L2)
        yield (e, 1, wi, hm2, vm2, D2, U2, R2, L2, me, wme - 1)


# ── Negamax + Alpha-Beta + TT ─────────────────────────────────────────────────
def negamax(me, opp, wme, wopp, gi, hm, vm, D, U, R, L,
            depth, alpha, beta, ply, zkey):
    if time.perf_counter() > _deadline:
        raise _Timeout

    entry   = _tt.get(zkey)
    tt_move = None
    if entry is not None:
        edepth, eflag, evalue, emove = entry
        if edepth >= depth:
            if eflag == TT_EXACT:                        return evalue
            elif eflag == TT_LOWER and evalue >= beta:   return evalue
            elif eflag == TT_UPPER and evalue <= alpha:  return evalue
        tt_move = emove

    moves = pawn_moves(me, opp, D, U, R, L)
    goal  = GM[gi]
    for dest in moves:
        if goal & BIT[dest]:
            return WIN - ply        # thắng → trả ngay, ply nhỏ = thắng nhanh hơn

    if depth <= 0:
        return evaluate(me, opp, wme, wopp, gi, hm, vm, D, U, R, L)

    pk   = kids_pawn(me, opp, wme, wopp, gi, hm, vm, D, U, R, L, moves)
    kids = pk
    if wme > 0:
        kids = pk + list(kids_wall(me, opp, wme, wopp, gi, hm, vm, D, U, R, L))

    if not kids:
        return 0.0

    if tt_move is not None:
        kids.sort(key=lambda k: 0 if (k[1], k[2]) == tt_move else 1 - k[0] / 10000.0)
    else:
        kids.sort(key=lambda k: -k[0])

    orig_alpha = alpha
    best_val   = -2 * WIN
    best_move  = (kids[0][1], kids[0][2])

    for i, k in enumerate(kids):
        nxt_zkey = zkey ^ Z_SIDE[gi] ^ Z_SIDE[1 - gi]
        if k[1] == 0:
            nxt_zkey ^= Z_ME[me] ^ Z_ME[k[9]]
        else:
            nxt_zkey ^= Z_WALL[k[2]] ^ Z_WME[wme] ^ Z_WME[wme - 1]

        d = depth - 1
        if i >= 6 and depth >= 3 and k[1] == 1:
            d = depth - 2

        v = -negamax(opp, k[9], wopp, k[10], 1 - gi, k[3], k[4],
                     k[5], k[6], k[7], k[8], d, -beta, -alpha, ply + 1, nxt_zkey)

        if d < depth - 1 and v > alpha:
            v = -negamax(opp, k[9], wopp, k[10], 1 - gi, k[3], k[4],
                         k[5], k[6], k[7], k[8], depth - 1, -beta, -alpha, ply + 1, nxt_zkey)

        if v > best_val:
            best_val  = v
            best_move = (k[1], k[2])
            if v > alpha:
                alpha = v
                if alpha >= beta:
                    break

    flag = TT_UPPER if best_val <= orig_alpha else (TT_LOWER if best_val >= beta else TT_EXACT)
    _tt[zkey] = (depth, flag, best_val, best_move)
    return best_val


# ── Root search ───────────────────────────────────────────────────────────────
_prev_best = None


def _think(obs):
    global _prev_best
    you, other = obs["you"], obs["opponent"]
    me  = you["position"][0]  * 9 + you["position"][1]
    opp = other["position"][0] * 9 + other["position"][1]
    gi  = 0 if you["goal_row"] == 0 else 1
    wme, wopp = you["walls_remaining"], other["walls_remaining"]

    hm = vm = 0
    D, U, R, L = BASE_D, BASE_U, BASE_R, BASE_L
    zkey = Z_ME[me] ^ Z_OPP[opp] ^ Z_WME[wme] ^ Z_WOPP[wopp] ^ Z_SIDE[gi]

    for r, c in obs["walls"]["horizontal"]:
        wi    = r * 8 + c
        w     = WALLS[wi]
        hm   |= w[6]
        D    &= w[4]
        U    &= w[5]
        zkey ^= Z_WALL[wi]

    for r, c in obs["walls"]["vertical"]:
        wi    = 64 + r * 8 + c
        w     = WALLS[wi]
        vm   |= w[6]
        R    &= w[4]
        L    &= w[5]
        zkey ^= Z_WALL[wi]

    legal_moves = sorted(r * 9 + c for r, c in obs["legal_moves"])
    legal_walls = set()
    for w in obs["legal_walls"]:
        legal_walls.add((0 if w["orientation"] == "H" else 64) + w["row"] * 8 + w["col"])

    # Thắng ngay nếu có thể
    goal = GM[gi]
    for m in legal_moves:
        if goal & BIT[m]:
            return {"type": "move", "to": [m // 9, m % 9]}

    # ── Race check (khi không còn tường) ────────────────────────────────────
    if wme == 0 and wopp == 0:
        outcome, dtm_v = _race_result(me, opp, gi, D, U, R, L)
        if outcome == 1:
            # Chắc thắng → chọn nước đi tốt nhất theo DTM (tối thiểu dtm)
            best_m, best_d = None, 9999
            for m in legal_moves:
                o2, d2 = _race_result(opp, m, 1 - gi, D, U, R, L)
                # sau khi ta đi m, đối thủ sẽ ở (opp, m, 1-gi) → ta muốn opp thua
                if o2 == -1 and d2 < best_d:
                    best_d, best_m = d2, m
            if best_m is not None:
                return {"type": "move", "to": [best_m // 9, best_m % 9]}

    if len(_cache) > 300000:
        _cache.clear()
    if len(_cand_cache) > 150000:
        _cand_cache.clear()
    _prev_best = None
    _tt.clear()

    kids = kids_pawn(me, opp, wme, wopp, gi, hm, vm, D, U, R, L, legal_moves)
    if wme > 0:
        kids += [k for k in kids_wall(me, opp, wme, wopp, gi, hm, vm, D, U, R, L)
                 if k[2] in legal_walls]
    if not kids:
        return None
    kids.sort(key=lambda k: -k[0])
    best = kids[0]

    for depth in range(2, MAX_DEPTH + 1):
        order = list(kids)
        if _prev_best is not None:
            order.sort(key=lambda k: 0 if (k[1], k[2]) == _prev_best else 1)
        try:
            alpha, beta = -2 * WIN, 2 * WIN
            cur, cur_v  = None, -2 * WIN
            for k in order:
                nxt_zkey = zkey ^ Z_SIDE[gi] ^ Z_SIDE[1 - gi]
                if k[1] == 0:
                    nxt_zkey ^= Z_ME[me] ^ Z_ME[k[9]]
                else:
                    nxt_zkey ^= Z_WALL[k[2]] ^ Z_WME[wme] ^ Z_WME[wme - 1]

                v = -negamax(opp, k[9], wopp, k[10], 1 - gi, k[3], k[4],
                             k[5], k[6], k[7], k[8], depth - 1, -beta, -alpha, 1, nxt_zkey)
                if v > cur_v:
                    cur_v, cur = v, k
                    if v > alpha:
                        alpha = v
            if cur is not None:
                best = cur
                _prev_best = (best[1], best[2])
            if cur_v >= WIN - 50 or cur_v <= -WIN + 50:
                break
        except _Timeout:
            break

    if best[1] == 0:
        return {"type": "move", "to": [best[2] // 9, best[2] % 9]}
    w = WALLS[best[2]]
    return {"type": "wall", "orientation": "H" if w[0] == 0 else "V",
            "row": w[1], "col": w[2]}


def _fallback(obs):
    moves = obs.get("legal_moves") or []
    if not moves:
        return {"type": "pass"}
    d    = obs.get("distances")
    best, bd = moves[0], 10 ** 9
    for m in moves:
        v = d[m[0]][m[1]] if d else 0
        if 0 <= v < bd:
            best, bd = m, v
    return {"type": "move", "to": list(best)}


def _valid(obs, a):
    if not isinstance(a, dict):
        return False
    if a.get("type") == "move":
        return list(a["to"]) in [list(m) for m in obs["legal_moves"]]
    if a.get("type") == "wall":
        return any(w["orientation"] == a["orientation"] and
                   w["row"] == a["row"] and w["col"] == a["col"]
                   for w in obs["legal_walls"])
    return a.get("type") == "pass"


def choose_action(observation):
    global _deadline, _prev_best
    if observation.get("turn", 1) <= 1:
        _prev_best = None
    _deadline = time.perf_counter() + TIME_BUDGET
    try:
        action = _think(observation)
    except Exception:
        action = None
    if action is None or not _valid(observation, action):
        action = _fallback(observation)
    return action

    return action