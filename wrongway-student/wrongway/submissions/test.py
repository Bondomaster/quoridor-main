"""Wrongway agent (một file, chỉ dùng thư viện chuẩn).

Ý tưởng:
  * Bàn cờ 9x9 được biểu diễn bằng số nguyên 81 bit (bitboard). Cạnh bị tường
    chặn được lưu thành 4 "mặt nạ" (xuống, lên, phải, trái).
  * Khoảng cách ngắn nhất tới đích tính bằng loang (flood fill) song song trên
    bitboard, nhanh hơn BFS thường nhiều lần và có cache.
  * Chọn nước đi bằng Negamax + cắt tỉa Alpha-Beta, đào sâu dần (iterative
    deepening) trong giới hạn thời gian.
  * Chỉ xét những tường nằm trên một đường ngắn nhất của đối thủ và làm đường
    đó dài thêm (các tường khác không thể tăng khoảng cách).
  * Hàm lượng giá: (đường đối thủ - đường của mình) + chênh lệch số tường
    + lợi thế đi trước.
"""
       
import random
import time

INF_D = 99          # khoảng cách "vô cực"   
WIN = 10000.0
NEAR_WIN = 900.0
TIME_BUDGET = 0.65  # giây (giới hạn của arena là 1 giây)
MAX_DEPTH = 10
CAND_EDGES = 7      # số cạnh đầu của đường đối thủ được xét để đặt tường

# Trọng số hàm lượng giá: Eval = W1*(do - dm) + W2*(wme - wopp) + W3*Tempo + W4*Mobility
W1_DIST = 2.0       # chênh lệch đường tới đích
W2_WALL = 1.60      # bảo tồn tường (tường là tài nguyên hữu hạn, rất quý)
W3_TEMPO = 0.45     # lợi thế nhịp đi trước
W4_MOB = 0.15       # độ cơ động (số hướng đi mở)

BIT = [1 << i for i in range(82)]
FULL = (1 << 81) - 1
_ROW = [((1 << 9) - 1) << (9 * r) for r in range(9)]
_COL = [sum(1 << (9 * r + c) for r in range(9)) for c in range(9)]
BASE_D = FULL & ~_ROW[8]   # ô có thể đi xuống (+9)
BASE_U = FULL & ~_ROW[0]   # ô có thể đi lên (-9)
BASE_R = FULL & ~_COL[8]   # ô có thể đi sang phải (+1)
BASE_L = FULL & ~_COL[0]   # ô có thể đi sang trái (-1)
GM = (_ROW[0], _ROW[8])    # mặt nạ hàng đích: gi=0 -> hàng 0, gi=1 -> hàng 8

# WALLS[o*64 + r*8 + c] = (o, r, c, a, giữ_A, giữ_B, bit_neo, xung_đột_cùng, xung_đột_khác)
WALLS = []
for _o in (0, 1):
    for _r in range(8):
        for _c in range(8):
            _a = 9 * _r + _c
            if _o == 0:      # tường ngang: chặn D (xuống) và U (lên)
                _cA = BIT[_a] | BIT[_a + 1]
                _cB = BIT[_a + 9] | BIT[_a + 10]
                _same = BIT[_a] | BIT[_a + 1] | (BIT[_a - 1] if _c > 0 else 0)
            else:            # tường dọc: chặn R (phải) và L (trái)
                _cA = BIT[_a] | BIT[_a + 9]
                _cB = BIT[_a + 1] | BIT[_a + 10]
                _same = BIT[_a] | BIT[_a + 9] | (BIT[_a - 9] if _r > 0 else 0)
            WALLS.append((_o, _r, _c, _a, FULL & ~_cA, FULL & ~_cB,
                          BIT[_a], _same, BIT[_a]))


class _Timeout(Exception):
    pass


_deadline = 0.0
_cache = {}
_cand_cache = {}
_tt = {}

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

# Bảng Zobrist Hashing
_rng = random.Random(25127275)
Z_ME = [_rng.getrandbits(64) for _ in range(81)]
Z_OPP = [_rng.getrandbits(64) for _ in range(81)]
Z_WALL = [_rng.getrandbits(64) for _ in range(128)]
Z_WME = [_rng.getrandbits(64) for _ in range(11)]
Z_WOPP = [_rng.getrandbits(64) for _ in range(11)]
Z_SIDE = [_rng.getrandbits(64) for _ in range(2)]


# --------------------------------------------------------------------------
# Khoảng cách / đường đi
# --------------------------------------------------------------------------
def dist(pos, gi, hm, vm, D, U, R, L):
    """Số bước ngắn nhất từ ô pos tới hàng đích gi (bỏ qua quân cờ)."""
    key = (hm, vm, pos, gi)
    v = _cache.get(key)
    if v is not None:
        return v
    S = GM[gi]
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
    """Một đường ngắn nhất, dạng danh sách cạnh (ô_từ, ô_tới)."""
    S = GM[gi]
    bit = BIT[pos]
    layers = [S]
    while not (S & bit):
        N = S | ((S >> 9) & D) | ((S << 9) & U) | ((S >> 1) & R) | ((S << 1) & L)
        if N == S:
            return []
        layers.append(N)
        S = N
    d = len(layers) - 1
    edges = []
    cur = pos
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
        d -= 1
    return edges


def wall_cands(opp, go, hm, vm, D, U, R, L):
    """Các tường (chỉ số) chặn một cạnh trên đường đi hoặc tạo chuỗi tường nối tiếp."""
    key = (hm, vm, opp, go)
    v = _cand_cache.get(key)
    if v is not None:
        return v
    out = []
    seen = set()

    # 1. Các tường cắt qua đường đi ngắn nhất của đối thủ
    for x, y in path_edges(opp, go, D, U, R, L)[:CAND_EDGES]:
        if x - y == 9 or y - x == 9:          # đi dọc -> tường ngang
            r = min(x, y) // 9
            c = x % 9
            for cc in (c, c - 1):
                if 0 <= cc <= 7 and r <= 7:
                    wi = r * 8 + cc
                    if wi not in seen:
                        seen.add(wi)
                        out.append(wi)
        else:                                  # đi ngang -> tường dọc
            r = x // 9
            c = min(x, y) % 9
            for rr in (r, r - 1):
                if 0 <= rr <= 7 and c <= 7:
                    wi = 64 + rr * 8 + c
                    if wi not in seen:
                        seen.add(wi)
                        out.append(wi)

    # 2. Các tường nối tiếp tạo chuỗi với tường đã có quanh đối thủ
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


# --------------------------------------------------------------------------
# Sinh nước đi
# --------------------------------------------------------------------------
def pawn_moves(me, opp, D, U, R, L):
    """Nước đi quân cờ (có nhảy / đi chéo), giống luật của arena."""
    out = []
    bm = BIT[me]
    if D & bm:
        a = me + 9
        if a != opp:
            out.append(a)
        elif D & BIT[a]:
            out.append(a + 9)
        else:
            if L & BIT[a]:
                out.append(a - 1)
            if R & BIT[a]:
                out.append(a + 1)
    if U & bm:
        a = me - 9
        if a != opp:
            out.append(a)
        elif U & BIT[a]:
            out.append(a - 9)
        else:
            if L & BIT[a]:
                out.append(a - 1)
            if R & BIT[a]:
                out.append(a + 1)
    if R & bm:
        a = me + 1
        if a != opp:
            out.append(a)
        elif R & BIT[a]:
            out.append(a + 1)
        else:
            if U & BIT[a]:
                out.append(a - 9)
            if D & BIT[a]:
                out.append(a + 9)
    if L & bm:
        a = me - 1
        if a != opp:
            out.append(a)
        elif L & BIT[a]:
            out.append(a - 1)
        else:
            if U & BIT[a]:
                out.append(a - 9)
            if D & BIT[a]:
                out.append(a + 9)
    return out


def evaluate(me, opp, wme, wopp, gi, hm, vm, D, U, R, L):
    """Hàm lượng giá đa tiêu chí kết hợp khoảng cách, tường, tempo và mobility."""
    dm = dist(me, gi, hm, vm, D, U, R, L)
    do = dist(opp, 1 - gi, hm, vm, D, U, R, L)

    if dm == 0:
        return WIN
    if do <= 1:
        return -NEAR_WIN
    if dm <= 2 and dm < do:
        return NEAR_WIN - dm * 20.0

    bm = BIT[me]
    me_mob = ((D & bm) != 0) + ((U & bm) != 0) + ((R & bm) != 0) + ((L & bm) != 0)
    bo = BIT[opp]
    opp_mob = ((D & bo) != 0) + ((U & bo) != 0) + ((R & bo) != 0) + ((L & bo) != 0)

    score = (W1_DIST * (do - dm) +
             W2_WALL * (wme - wopp) +
             W3_TEMPO +
             W4_MOB * (me_mob - opp_mob))
    return score


def kids_pawn(me, opp, wme, wopp, gi, hm, vm, D, U, R, L, moves):
    """Con của nút khi đi quân. Sắp xếp ưu tiên giảm khoảng cách D_me."""
    out = []
    goal = GM[gi]
    for dest in moves:
        if goal & BIT[dest]:
            out.append((WIN, 0, dest, hm, vm, D, U, R, L, dest, wme))
            continue
        e = evaluate(dest, opp, wme, wopp, gi, hm, vm, D, U, R, L)
        out.append((e, 0, dest, hm, vm, D, U, R, L, dest, wme))
    return out


def kids_wall(me, opp, wme, wopp, gi, hm, vm, D, U, R, L):
    """Sinh các nước đặt tường có chọn lọc (gain > 0, không spam lãng phí)."""
    go = 1 - gi
    do0 = dist(opp, go, hm, vm, D, U, R, L)
    dm0 = dist(me, gi, hm, vm, D, U, R, L)

    # Tiết kiệm tường: nếu ta đang dẫn trước và đối thủ còn xa, chỉ đặt tường có gain >= 2
    min_gain = 2 if (dm0 < do0 and do0 > 3) else 1

    for wi in wall_cands(opp, go, hm, vm, D, U, R, L):
        o, r, c, a, nA, nB, selfm, same, other = WALLS[wi]
        if o == 0:
            if (hm & same) or (vm & other):
                continue
            hm2 = hm | selfm
            vm2 = vm
            D2 = D & nA
            U2 = U & nB
            R2 = R
            L2 = L
        else:
            if (vm & same) or (hm & other):
                continue
            vm2 = vm | selfm
            hm2 = hm
            R2 = R & nA
            L2 = L & nB
            D2 = D
            U2 = U
        do = dist(opp, go, hm2, vm2, D2, U2, R2, L2)
        if do >= INF_D or do <= do0:
            continue
        dm = dist(me, gi, hm2, vm2, D2, U2, R2, L2)
        if dm >= INF_D:
            continue

        gain = (do - do0) - (dm - dm0)
        if gain < min_gain and do0 > 2:
            continue

        e = evaluate(me, opp, wme - 1, wopp, gi, hm2, vm2, D2, U2, R2, L2)
        yield (e, 1, wi, hm2, vm2, D2, U2, R2, L2, me, wme - 1)


# --------------------------------------------------------------------------
# Tìm kiếm
# --------------------------------------------------------------------------
def negamax(me, opp, wme, wopp, gi, hm, vm, D, U, R, L, depth, alpha, beta, ply, zkey):
    """Giá trị của thế cờ theo góc nhìn bên đang tới lượt với TT + Alpha-Beta."""
    if time.perf_counter() > _deadline:
        raise _Timeout

    # Tra cứu bảng chuyển vị (Transposition Table)
    entry = _tt.get(zkey)
    tt_move = None
    if entry is not None:
        edepth, eflag, evalue, emove = entry
        if edepth >= depth:
            if eflag == TT_EXACT:
                return evalue
            elif eflag == TT_LOWER and evalue >= beta:
                return evalue
            elif eflag == TT_UPPER and evalue <= alpha:
                return evalue
        tt_move = emove

    moves = pawn_moves(me, opp, D, U, R, L)
    goal = GM[gi]
    for dest in moves:
        if goal & BIT[dest]:
            return WIN - ply

    if depth <= 0:
        return evaluate(me, opp, wme, wopp, gi, hm, vm, D, U, R, L)

    pk = kids_pawn(me, opp, wme, wopp, gi, hm, vm, D, U, R, L, moves)
    kids = pk
    if wme > 0:
        kids = pk + list(kids_wall(me, opp, wme, wopp, gi, hm, vm, D, U, R, L))

    if not kids:
        return 0.0

    # Move Ordering: TT move trước tiên, tiếp theo là điểm lượng giá tĩnh
    if tt_move is not None:
        kids.sort(key=lambda k: 0 if (k[1], k[2]) == tt_move else 1 - k[0] / 10000.0)
    else:
        kids.sort(key=lambda k: -k[0])

    orig_alpha = alpha
    best_val = -2 * WIN
    best_move = (kids[0][1], kids[0][2])

    for i, k in enumerate(kids):
        nxt_zkey = zkey ^ Z_SIDE[gi] ^ Z_SIDE[1 - gi]
        if k[1] == 0:
            nxt_zkey ^= Z_ME[me] ^ Z_ME[k[9]]
        else:
            nxt_zkey ^= Z_WALL[k[2]] ^ Z_WME[wme] ^ Z_WME[wme - 1]

        # Late Move Reduction (LMR) cho các nước tường ở cuối
        d = depth - 1
        if i >= 6 and depth >= 3 and k[1] == 1:
            d = depth - 2

        v = -negamax(opp, k[9], wopp, k[10], 1 - gi, k[3], k[4],
                     k[5], k[6], k[7], k[8], d, -beta, -alpha, ply + 1, nxt_zkey)

        if d < depth - 1 and v > alpha:
            v = -negamax(opp, k[9], wopp, k[10], 1 - gi, k[3], k[4],
                         k[5], k[6], k[7], k[8], depth - 1, -beta, -alpha, ply + 1, nxt_zkey)

        if v > best_val:
            best_val = v
            best_move = (k[1], k[2])
            if v > alpha:
                alpha = v
                if alpha >= beta:
                    break

    if best_val <= orig_alpha:
        flag = TT_UPPER
    elif best_val >= beta:
        flag = TT_LOWER
    else:
        flag = TT_EXACT
    _tt[zkey] = (depth, flag, best_val, best_move)
    return best_val


_prev_best = None


def _think(obs):
    global _prev_best
    you, other = obs["you"], obs["opponent"]
    me = you["position"][0] * 9 + you["position"][1]
    opp = other["position"][0] * 9 + other["position"][1]
    gi = 0 if you["goal_row"] == 0 else 1
    wme, wopp = you["walls_remaining"], other["walls_remaining"]

    hm = vm = 0
    D, U, R, L = BASE_D, BASE_U, BASE_R, BASE_L
    zkey = Z_ME[me] ^ Z_OPP[opp] ^ Z_WME[wme] ^ Z_WOPP[wopp] ^ Z_SIDE[gi]

    for r, c in obs["walls"]["horizontal"]:
        wi = r * 8 + c
        w = WALLS[wi]
        hm |= w[6]
        D &= w[4]
        U &= w[5]
        zkey ^= Z_WALL[wi]

    for r, c in obs["walls"]["vertical"]:
        wi = 64 + r * 8 + c
        w = WALLS[wi]
        vm |= w[6]
        R &= w[4]
        L &= w[5]
        zkey ^= Z_WALL[wi]

    legal_moves = sorted(r * 9 + c for r, c in obs["legal_moves"])
    legal_walls = set()
    for w in obs["legal_walls"]:
        legal_walls.add((0 if w["orientation"] == "H" else 64) + w["row"] * 8 + w["col"])

    goal = GM[gi]
    for m in legal_moves:                      # thắng ngay nếu có thể
        if goal & BIT[m]:
            return {"type": "move", "to": [m // 9, m % 9]}

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
            cur, cur_v = None, -2 * WIN
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
    return {"type": "wall", "orientation": "H" if w[0] == 0 else "V", "row": w[1], "col": w[2]}


def _fallback(obs):
    moves = obs.get("legal_moves") or []
    if not moves:
        return {"type": "pass"}
    d = obs.get("distances")
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
        return any(w["orientation"] == a["orientation"] and w["row"] == a["row"]
                   and w["col"] == a["col"] for w in obs["legal_walls"])
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