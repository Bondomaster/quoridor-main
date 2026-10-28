"""Wrongway Bot VB (Victory Bot) - Unified Champion.
Tong hop cac thuat toan manh nhat tu Top 1, Top 2, Top 3, Top 4, Top 5:
- Top 1 (alex.py): Obito-Codex PVS, Bitboard BFS, Non-linear wall marginal values, Killer/History heuristic.
- Top 2 (hihi.py): Retrograde WDL/DTM exact endgame pawn-race solver khi het tuong.
- Top 3 (test4.py): Deep candidate wall scanning doc theo toan bo duong di ngan nhat (CAND_EDGES=16, TOP_WALLS=12).
- Top 4 (t1.py): Contact jump tactical tempo calculation khi hai quan ap sat nhau.
- Top 5 (test1.py): Emergency threat detector - chan tuong khan cap khi doi thu cach dich <= 2 buoc.
"""
import gc
from time import perf_counter

# Tắt bộ dọn rác tự động để tránh khựng ~20ms ngẫu nhiên giữa lúc tìm kiếm
# (chương trình không tạo vòng tham chiếu; vẫn dọn định kỳ ở đầu lượt).
gc.disable()

# ---------------------------------------------------------------------------
# Hằng số bitboard
# ---------------------------------------------------------------------------
SIZE = 9
ALL = (1 << 81) - 1
ROW = [((1 << 9) - 1) << (9 * r) for r in range(9)]
COL = [sum(1 << (9 * r + c) for r in range(9)) for c in range(9)]
ANCHOR = sum(1 << (9 * r + c) for r in range(8) for c in range(8))  # neo tường hợp lệ
U0 = ALL & ~ROW[0]   # ô có thể đi lên (chưa có tường)
D0 = ALL & ~ROW[8]   # ô có thể đi xuống
L0 = ALL & ~COL[0]   # ô có thể sang trái
R0 = ALL & ~COL[8]   # ô có thể sang phải
INF = 1000
_RC_R = [i // 9 for i in range(81)]   # hàng của ô theo chỉ số bit
_RC_C = [i % 9 for i in range(81)]    # cột của ô theo chỉ số bit

# Mã hóa nước đi bằng số nguyên: 0..80 = đi tới ô, 100+a = tường H tại neo a,
# 200+a = tường V tại neo a, PASS = đứng yên.
H_BASE = 100
V_BASE = 200
PASS = 999

# ---------------------------------------------------------------------------
# Tham số đánh giá (đơn vị: 10 = một bước đi)
# ---------------------------------------------------------------------------
STEP = 10           # giá trị 1 bước chênh lệch quãng đường
TEMPO = 5           # lợi thế của bên đang tới lượt
RACE_BONUS = 60     # cuộc đua đã "chắc chắn" (một bên hết tường)
# Giá trị của k bức tường còn lại (đã tinh chỉnh bằng tự đấu): mỗi tường ~1.2
# bước, riêng 3 tường cuối cùng quý hơn (2.4 / 1.8 / 1.4 bước) vì cuối ván, bên
# còn tường có thể chặn quyết định.  WALL_VALUE[k] = tổng giá trị của k tường.
_WALL_MARGINAL = [26, 20, 16, 13, 12, 12, 12, 12, 12, 12]
WALL_VALUE = [sum(_WALL_MARGINAL[:k]) for k in range(11)]
TIME_LIMIT = 0.84   # một deadline chung, chừa 0.16s cho IPC và dao động lịch CPU
TOP_WALLS = 12       # số tường tối đa xét tại một nút trong cây
ITER_FACTOR = 1.0   # bắt đầu độ sâu mới nếu còn ít nhất bằng thời gian lượt lặp vừa rồi
WIN = 100000

# Bộ nhớ dùng chung giữa các lượt trong một ván
_TT = {}
_STATE = {"last_turn": -1, "margin": 0.0, "moves": 0, "goals": None}
_HIST = set()        # các thế cờ đã xuất hiện trong ván (để nhận ra lặp lại)
# Root-only preferences: they never change the game rules or TT evaluations.
ROOT_PASS_PENALTY = 4        # 0.4 step, only while pawns are far apart
ROOT_REPEAT_PENALTY = 2      # mild preference against a recent same-wall retreat
ROOT_RECENT_WINDOW = 6
_ACTIVE_RECENT = []          # (turn, pawn bit, horizontal walls, vertical walls)
_ROOT_GOAL = [0]     # hàng đích của "mình" ở lượt hiện tại (để tính điểm hòa)
CONTEMPT = 35        # hòa chỉ được 1 điểm (thắng 3): coi hòa là hơi xấu cho mình
_BLOCK = [False]     # bật kiểm tra "chặn cửa" khi trên bàn đã đủ nhiều tường
_BCACHE = {}
BLOCK_MIN_WALLS = 8
BLOCK_MIN_S = 15    # chỉ kiểm tra khi một bên dẫn rõ rệt


MATE_ZONE = WIN - 1000


class _TimeUp(Exception):
    pass


def _to_tt(score, ply):
    """Điểm thắng/thua phụ thuộc độ sâu: lưu dạng tương đối để tái sử dụng đúng."""
    if score > MATE_ZONE:
        return score + ply
    if score < -MATE_ZONE:
        return score - ply
    return score


def _draw(goal):
    """Điểm của thế HÒA theo góc nhìn bên có hàng đích `goal` đang tới lượt.

    Thể thức 3-1-0: hòa kém hơn một ván cân bằng, nên với "mình" hòa = -CONTEMPT,
    với đối thủ hòa = +CONTEMPT (giữ tính đối xứng của negamax).
    """
    return -CONTEMPT if goal == _ROOT_GOAL[0] else CONTEMPT


# ---------------------------------------------------------------------------
# Các hàm cơ bản trên bitboard
# ---------------------------------------------------------------------------
def _dist(start, goal, U, D, L, R):
    """BFS bitboard: số bước ít nhất từ ô `start` (1 bit) tới hàng `goal`."""
    if start & goal:
        return 0
    seen = front = start
    d = 0
    while True:
        d += 1
        front = (((front & U) >> 9) | ((front & D) << 9)
                 | ((front & L) >> 1) | ((front & R) << 1)) & ~seen
        if not front:
            return INF
        if front & goal:
            return d
        seen |= front


def _back_layers(goal, target, U, D, L, R, extra):
    """BFS ngược từ hàng đích. layers[k] = các ô cách đích đúng k bước.

    Dừng sau khi chạm `target` thêm `extra` lớp. Trả về (layers, k_target).
    """
    layers = [goal]
    seen = front = goal
    hit = 0 if goal & target else -1
    while hit < 0 or len(layers) <= hit + extra:
        front = (((front & U) >> 9) | ((front & D) << 9)
                 | ((front & L) >> 1) | ((front & R) << 1)) & ~seen
        if not front:
            break
        seen |= front
        layers.append(front)
        if hit < 0 and front & target:
            hit = len(layers) - 1
    return layers, hit


def _sp_edges(pos, layers, d, U, D, L, R):
    """Các cạnh nằm trên ÍT NHẤT MỘT đường đi ngắn nhất từ pos tới đích.

    Trả về 4 mặt nạ: ô xuất phát của cạnh đi lên / xuống / trái / phải.
    """
    up = dn = lt = rt = 0
    cur = pos
    for k in range(d, 0, -1):
        nxt = layers[k - 1]
        u = cur & U & (nxt << 9)
        w = cur & D & (nxt >> 9)
        l = cur & L & (nxt << 1)
        r = cur & R & (nxt >> 1)
        up |= u
        dn |= w
        lt |= l
        rt |= r
        cur = (u >> 9) | (w << 9) | (l >> 1) | (r << 1)
    return up, dn, lt, rt


def _pawn_moves(a, b, U, D, L, R):
    """Các ô quân `a` đi được (đúng luật arena: nhảy thẳng, chéo khi bị chặn)."""
    res = []
    if a & U:
        t = a >> 9
        if t != b:
            res.append(t)
        elif b & U:
            res.append(b >> 9)
        else:
            if b & L:
                res.append(b >> 1)
            if b & R:
                res.append(b << 1)
    if a & D:
        t = a << 9
        if t != b:
            res.append(t)
        elif b & D:
            res.append(b << 9)
        else:
            if b & L:
                res.append(b >> 1)
            if b & R:
                res.append(b << 1)
    if a & L:
        t = a >> 1
        if t != b:
            res.append(t)
        elif b & L:
            res.append(b >> 1)
        else:
            if b & U:
                res.append(b >> 9)
            if b & D:
                res.append(b << 9)
    if a & R:
        t = a << 1
        if t != b:
            res.append(t)
        elif b & R:
            res.append(b << 1)
        else:
            if b & U:
                res.append(b >> 9)
            if b & D:
                res.append(b << 9)
    return res


def _static(dm, do, wm, wo):
    """Điểm thế cờ theo góc nhìn bên ĐANG TỚI LƯỢT.

    dm/do: quãng đường của mình/đối thủ; wm/wo: số tường còn lại.
    """
    s = STEP * (do - dm) + TEMPO + WALL_VALUE[wm] - WALL_VALUE[wo]
    if wo == 0 and dm <= do:
        s += RACE_BONUS        # đối thủ không thể làm mình chậm lại nữa
    elif wm == 0 and dm > do:
        s -= RACE_BONUS        # mình không thể làm đối thủ chậm lại nữa
    return s


def _jumps(o, U, D, L, R):
    """Các bước "nhảy qua" quân o: list (ô đứng cạnh o, các ô đáp xuống)."""
    js = []
    if o & D:                          # đứng dưới o, nhảy lên
        dest = (o >> 9) if o & U else (((o >> 1) if o & L else 0) | ((o << 1) if o & R else 0))
        if dest:
            js.append((o << 9, dest))
    if o & U:                          # đứng trên o, nhảy xuống
        dest = (o << 9) if o & D else (((o >> 1) if o & L else 0) | ((o << 1) if o & R else 0))
        if dest:
            js.append((o >> 9, dest))
    if o & L:                          # đứng bên trái o, nhảy sang phải
        dest = (o << 1) if o & R else (((o >> 9) if o & U else 0) | ((o << 9) if o & D else 0))
        if dest:
            js.append((o >> 1, dest))
    if o & R:                          # đứng bên phải o, nhảy sang trái
        dest = (o >> 1) if o & L else (((o >> 9) if o & U else 0) | ((o << 9) if o & D else 0))
        if dest:
            js.append((o << 1, dest))
    return js


def _reach_mask(o, goal, U, D, L, R):
    """Tập ô có thể tới hàng `goal` khi quân o đứng yên mãi (được nhảy qua o).

    BFS ngược từ đích, không đi vào ô o; cạnh "nhảy" m -> dest được xét ngược:
    nếu dest tới được đích thì m cũng tới được.
    """
    js = _jumps(o, U, D, L, R)
    seen = goal | o
    front = goal & ~o
    while front:
        nxt = (((front & U) >> 9) | ((front & D) << 9)
               | ((front & L) >> 1) | ((front & R) << 1)) & ~seen
        for m, dest in js:
            if (front & dest) and not (seen & m):
                nxt |= m
        seen |= nxt
        front = nxt
    return seen & ~o


def _blocked(a, o, goal, U, D, L, R, hw, vw, d):
    """True nếu quân a KHÔNG thể tới đích khi quân o đứng yên mãi.

    Luật cho phép đứng yên, nên nếu o đứng đúng "cửa ải" duy nhất thì bên kia
    chỉ có thể hòa -> nhận ra thế "chặn cửa". Kết quả được nhớ theo (o, tường).
    """
    key = (o, hw, vw, goal)
    m = _BCACHE.get(key)
    if m is None:
        # Lọc nhanh: còn đường ngắn nhất (d bước) không đi qua ô o -> không bị chặn.
        if a & goal:
            return False
        seen = a | o
        front = a
        for _ in range(d):
            front = (((front & U) >> 9) | ((front & D) << 9)
                     | ((front & L) >> 1) | ((front & R) << 1)) & ~seen
            if front & goal:
                return False
            if not front:
                break
            seen |= front
        m = _reach_mask(o, goal, U, D, L, R)
        _BCACHE[key] = m
    return not (m & a)


def _may_block(a, o, goal_row, d):
    """Điều kiện cần để o chặn được a: o nằm trên một đường ngắn nhất của a.

    Kiểm tra nhanh bằng khoảng cách Manhattan (cận dưới của quãng đường thật).
    """
    ia = a.bit_length() - 1
    io = o.bit_length() - 1
    ra = _RC_R[ia]
    ro = _RC_R[io]
    dr = ra - ro if ra > ro else ro - ra
    dc = _RC_C[ia] - _RC_C[io]
    if dc < 0:
        dc = -dc
    dg = ro - goal_row if ro > goal_row else goal_row - ro
    return dr + dc + dg <= d


def _evalb(dm, do, wm, wo, me, op, gm, go, U, D, L, R, hw, vw):
    """Đánh giá tĩnh có xét thế "chặn cửa" (bên đang thắng bị chặn -> hòa)."""
    adjacent = (((me & U) >> 9) | ((me & D) << 9) |
                ((me & L) >> 1) | ((me & R) << 1)) & op
    jm, jo = dm, do
    if adjacent:
        # Codex's one-step correction is conditional on real contact; topology
        # BFS alone misses a useful straight jump or forced diagonal.
        ma = _pawn_moves(me, op, U, D, L, R)
        mo = _pawn_moves(op, me, U, D, L, R)
        if ma: jm = 1 + min(_dist(x, gm, U, D, L, R) for x in ma)
        if mo: jo = 1 + min(_dist(x, go, U, D, L, R) for x in mo)
        # [T1 ENHANCEMENT: CONTACT JUMP TACTICS]
        if jm < dm:
            jm -= 1  # Thuong diem vi kiem soat cuoc nhay chien thuat
    s = _static(jm, jo, wm, wo)
    if _BLOCK[0] and (s > BLOCK_MIN_S or s < -BLOCK_MIN_S):
        if s > 0:
            if (_may_block(me, op, 0 if gm & 1 else 8, dm)
                    and _blocked(me, op, gm, U, D, L, R, hw, vw, dm)):
                return _draw(gm)
        elif s < 0:
            if (_may_block(op, me, 0 if go & 1 else 8, do)
                    and _blocked(op, me, go, U, D, L, R, hw, vw, do)):
                return _draw(gm)
    return s


def _wall_list(hw, vw):
    """Đổi mặt nạ neo tường sang danh sách bit (dùng khi cần)."""
    out = []
    for mask, base in ((hw, H_BASE), (vw, V_BASE)):
        while mask:
            low = mask & -mask
            out.append(base + low.bit_length() - 1)
            mask ^= low
    return out


# ---------------------------------------------------------------------------
# Sinh nước đi cho một nút
#   Mỗi "con" là tuple:
#   (điểm sắp xếp, mã nước, a', U', D', L', R', hw', vw', wa', d_a', d_b')
#   Điểm sắp xếp = đánh giá tĩnh sau nước đi, theo góc nhìn bên vừa đi.
# ---------------------------------------------------------------------------
def _gen_pawns(a, b, wa, wb, ga, gb, U, D, L, R, hw, vw, lay_m, dm, db):
    """Các nước di chuyển quân. Trả về (mã nước thắng ngay hoặc None, list con)."""
    out = []
    nlay = len(lay_m)
    lo = dm - 2 if dm >= 2 else 0
    hi = dm + 3 if dm + 3 <= nlay else nlay
    for x in _pawn_moves(a, b, U, D, L, R):
        if x & ga:
            return x.bit_length() - 1, None
        nd = -1
        for k in range(lo, hi):
            if x & lay_m[k]:
                nd = k
                break
        if nd < 0:
            nd = _dist(x, ga, U, D, L, R)
        out.append((-_evalb(db, nd, wb, wa, b, x, gb, ga, U, D, L, R, hw, vw), x.bit_length() - 1,
                    x, U, D, L, R, hw, vw, wa, nd, db))
    return None, out


def _gen_walls(a, b, wa, wb, ga, gb, U, D, L, R, hw, vw, lay_m, dm, top, defensive):
    """Các bức tường đáng xét, đã sắp xếp giảm dần theo lợi ích, tối đa `top`.

    Ứng viên = tường chặn ít nhất một cạnh trên một đường ngắn nhất của đối thủ.
    `defensive`: thêm tường chiếm chỗ những tường đối thủ có thể dùng để chặn mình.
    """
    mu, md, ml, mr = _sp_edges(a, lay_m, dm, U, D, L, R)
    lay_o, do = _back_layers(gb, b, U, D, L, R, 0)
    ou, od, ol, orr = _sp_edges(b, lay_o, do, U, D, L, R)
    avail_h = ANCHOR & ~(hw | (hw << 1) | (hw >> 1) | vw)
    avail_v = ANCHOR & ~(vw | (vw << 9) | (vw >> 9) | hw)
    cand_h = ((ou >> 9) | (ou >> 10) | od | (od >> 1)) & avail_h
    cand_v = (orr | (orr >> 9) | (ol >> 1) | (ol >> 10)) & avail_v
    if defensive:
        # Tường "phòng thủ": chiếm trước chỗ (chồng lấn / cắt ngang) của những tường
        # mà đối thủ có thể dùng để chặn đường ngắn nhất của mình.
        threat_h = ((mu >> 9) | (mu >> 10) | md | (md >> 1)) & ANCHOR
        threat_v = (mr | (mr >> 9) | (ml >> 1) | (ml >> 10)) & ANCHOR
        cand_h |= ((threat_h << 1) | (threat_h >> 1) | threat_v) & avail_h
        cand_v |= ((threat_v << 9) | (threat_v >> 9) | threat_h) & avail_v
        if top is not None and wb:
            # Include the dangerous closure itself to measure how much a
            # zero-gain crossing wall can prevent on the following turn.
            cand_h |= threat_h & avail_h
            cand_v |= threat_v & avail_v
    contact = ((((a & U) >> 9) | ((a & D) << 9) |
                ((a & L) >> 1) | ((a & R) << 1)) & b)
    wa2 = wa - 1
    walls = []
    for is_h, mask in ((1, cand_h), (0, cand_v)):
        while mask:
            w = mask & -mask
            mask ^= w
            if is_h:
                e1 = w | (w << 1)
                e2 = (w << 9) | (w << 10)
                D2 = D & ~e1
                U2 = U & ~e2
                L2 = L
                R2 = R
                touch_me = (md & e1) or (mu & e2)
                touch_op = (od & e1) or (ou & e2)
                hw2 = hw | w
                vw2 = vw
                code = H_BASE + w.bit_length() - 1
            else:
                e1 = w | (w << 9)
                e2 = (w << 1) | (w << 10)
                R2 = R & ~e1
                L2 = L & ~e2
                U2 = U
                D2 = D
                touch_me = (mr & e1) or (ml & e2)
                touch_op = (orr & e1) or (ol & e2)
                hw2 = hw
                vw2 = vw | w
                code = V_BASE + w.bit_length() - 1
            if touch_op:
                ndo = _dist(b, gb, U2, D2, L2, R2)
                if ndo >= INF:
                    continue
            else:
                ndo = do
            if touch_me:
                ndm = _dist(a, ga, U2, D2, L2, R2)
                if ndm >= INF:
                    continue
            else:
                ndm = dm
            score = (-_evalb(ndo, ndm, wb, wa2, b, a, gb, ga,
                             U2, D2, L2, R2, hw2, vw2)
                     if contact else -_static(ndo, ndm, wb, wa2))
            walls.append((score, code,
                          a, U2, D2, L2, R2, hw2, vw2, wa2, ndm, ndo))
    walls.sort(reverse=True)
    if top is not None and len(walls) > top:
        selected = walls[:top]
        if defensive and wb:
            dangers = sorted(((k[10] - dm - (k[11] - do), k[1])
                              for k in walls
                              if k[10] - dm - (k[11] - do) >= 3), reverse=True)[:3]
            protect = {}
            for damage, threat in dangers:
                base = V_BASE if threat >= V_BASE else H_BASE
                index = threat - base
                codes = ((H_BASE + index, H_BASE + index - 1,
                          H_BASE + index + 1, V_BASE + index)
                         if base == H_BASE else
                         (V_BASE + index, V_BASE + index - 9,
                          V_BASE + index + 9, H_BASE + index))
                for code in codes:
                    protect[code] = max(damage, protect.get(code, 0))
            present = {k[1] for k in selected}
            defenses = [k for k in walls if k[1] in protect and k[1] not in present]
            defenses.sort(key=lambda k: (protect[k[1]], k[0]), reverse=True)
            selected.extend(defenses[:3])
        walls = selected
    return walls


def _wall_child(code, a, b, wa, wb, ga, gb, U, D, L, R, hw, vw):
    """Dựng con cho một mã tường cụ thể (từ bảng chuyển vị / killer); None nếu sai luật."""
    if wa <= 0:
        return None
    if code >= V_BASE:
        w = 1 << (code - V_BASE)
        if not (w & ANCHOR) or (w & (vw | (vw << 9) | (vw >> 9) | hw)):
            return None
        R2 = R & ~(w | (w << 9))
        L2 = L & ~((w << 1) | (w << 10))
        U2, D2, hw2, vw2 = U, D, hw, vw | w
    else:
        w = 1 << (code - H_BASE)
        if not (w & ANCHOR) or (w & (hw | (hw << 1) | (hw >> 1) | vw)):
            return None
        D2 = D & ~(w | (w << 1))
        U2 = U & ~((w << 9) | (w << 10))
        L2, R2, hw2, vw2 = L, R, hw | w, vw
    ndo = _dist(b, gb, U2, D2, L2, R2)
    if ndo >= INF:
        return None
    ndm = _dist(a, ga, U2, D2, L2, R2)
    if ndm >= INF:
        return None
    return (0, code, a, U2, D2, L2, R2, hw2, vw2, wa - 1, ndm, ndo)


# ---------------------------------------------------------------------------
# Tìm kiếm Negamax + Alpha-Beta (PVS, bảng chuyển vị, killer move)
# ---------------------------------------------------------------------------
class _Searcher:
    def __init__(self, deadline, turn, max_turns):
        self.deadline = deadline
        self.turn = turn
        self.max_turns = max_turns
        self.nodes = 0
        self.killers = [[None, None] for _ in range(128)]
        self.path = set()

    def search(self, a, b, wa, wb, ga, gb, U, D, L, R, hw, vw, da, db,
               depth, alpha, beta, ply):
        self.nodes += 1
        if (self.nodes & 7) == 0 and perf_counter() > self.deadline:
            raise _TimeUp()
        # Reaching the goal wins even on the last permitted action.
        if a & ga:
            return WIN - ply
        if b & gb:
            return -WIN + ply
        remaining = self.max_turns - self.turn - ply
        if remaining <= 0:
            return _draw(ga)               # hết giới hạn lượt -> hòa
        if depth <= 0:
            # A wall-only distance can miss an immediate straight/diagonal jump.
            if da <= 2 and any(x & ga for x in _pawn_moves(a, b, U, D, L, R)):
                return WIN - ply - 1
            if remaining == 1:
                return _draw(ga)
            return _evalb(da, db, wa, wb, a, b, ga, gb, U, D, L, R, hw, vw)

        # The rules have no repetition draw. Finite search and the TT handle
        # cycles; history is used only to break root ties. A capped horizon is
        # exact here because the maximum search depth is below 64 plies.
        key = (a, b, wa, wb, hw, vw, ga, min(remaining, 64), min(ply, 3))
        entry = _TT.get(key)
        tt_move = None
        if entry is not None:
            e_depth, e_score, e_flag, tt_move = entry
            if e_score > MATE_ZONE:
                e_score -= ply
            elif e_score < -MATE_ZONE:
                e_score += ply
            if e_depth >= depth:
                if e_flag == 0:
                    return e_score
                if e_flag == 1:
                    if e_score >= beta:
                        return e_score
                elif e_score <= alpha:
                    return e_score

        lay_m, dm = _back_layers(ga, a, U, D, L, R, 2)
        win, pawns = _gen_pawns(a, b, wa, wb, ga, gb, U, D, L, R, hw, vw, lay_m, dm, db)
        if win is not None:
            return WIN - ply - 1           # về đích ngay
        pass_sc = -_evalb(db, da, wb, wa, b, a, gb, ga, U, D, L, R, hw, vw)

        if depth == 1:
            if remaining == 1:
                # No action reaches the goal (checked above), so every legal
                # action exhausts the turn limit and the result is a draw.
                best = _draw(ga)
                _TT[key] = (1, best, 0, PASS)
                return best
            # Nút sát lá: giá trị = nước tốt nhất theo đánh giá tĩnh.
            # Check the opponent's immediate goal after each candidate when
            # it is close enough to win with a normal step or a jump.
            threat = db <= 2
            best = (self._leaf_child(b, a, wb, wa, gb, ga, U, D, L, R,
                                     hw, vw, db, da, ply + 1)
                    if threat else pass_sc)
            best_move = PASS
            for k in pawns:
                value = (self._leaf_child(b, k[2], wb, k[9], gb, ga,
                                         k[3], k[4], k[5], k[6], k[7], k[8],
                                         k[11], k[10], ply + 1)
                         if threat else k[0])
                if value > best:
                    best = value
                    best_move = k[1]
            if best < beta and wa > 0:
                walls = _gen_walls(a, b, wa, wb, ga, gb, U, D, L, R, hw, vw,
                                   lay_m, dm, None if threat else 1, threat)
                for k in walls:
                    value = (self._leaf_child(b, a, wb, k[9], gb, ga,
                                             k[3], k[4], k[5], k[6], k[7], k[8],
                                             k[11], k[10], ply + 1)
                             if threat else k[0])
                    if value > best:
                        best = value
                        best_move = k[1]
                    if best >= beta:
                        break
            flag = 1 if best >= beta else (2 if best <= alpha else 0)
            _TT[key] = (1, _to_tt(best, ply), flag, best_move)
            return best

        pawns.sort(reverse=True)
        self.path.add(key)
        orig_alpha = alpha
        best = -WIN * 2
        best_move = None
        nd = depth - 1
        np1 = ply + 1
        tried = set()
        first = True
        killers = self.killers[ply]
        cutoff = False
        stage = 0
        while stage < 5 and not cutoff:
            # Thứ tự xét: nước trong bảng chuyển vị -> killer -> di chuyển -> tường -> đứng yên
            if stage == 0:
                lst = []
                if tt_move is not None:
                    if tt_move == PASS:
                        lst = [(pass_sc, PASS, a, U, D, L, R, hw, vw, wa, da, db)]
                    elif tt_move < H_BASE:
                        lst = [k for k in pawns if k[1] == tt_move]
                    else:
                        c = _wall_child(tt_move, a, b, wa, wb, ga, gb, U, D, L, R, hw, vw)
                        if c is not None:
                            lst = [c]
            elif stage == 1:
                lst = []
                if wa > 0:
                    for km in killers:
                        if km is not None and km not in tried:
                            c = _wall_child(km, a, b, wa, wb, ga, gb, U, D, L, R, hw, vw)
                            if c is not None:
                                lst.append(c)
            elif stage == 2:
                lst = pawns
            elif stage == 3:
                lst = _gen_walls(a, b, wa, wb, ga, gb, U, D, L, R, hw, vw,
                                 lay_m, dm, TOP_WALLS, ply <= 2) if wa > 0 else ()
            else:
                lst = [(pass_sc, PASS, a, U, D, L, R, hw, vw, wa, da, db)]
            stage += 1
            for k in lst:
                code = k[1]
                if code in tried:
                    continue
                tried.add(code)
                if first:
                    sc = -self.search(b, k[2], wb, k[9], gb, ga, k[3], k[4], k[5], k[6],
                                      k[7], k[8], k[11], k[10], nd, -beta, -alpha, np1)
                    first = False
                else:
                    sc = -self.search(b, k[2], wb, k[9], gb, ga, k[3], k[4], k[5], k[6],
                                      k[7], k[8], k[11], k[10], nd, -alpha - 1, -alpha, np1)
                    if alpha < sc < beta:
                        sc = -self.search(b, k[2], wb, k[9], gb, ga, k[3], k[4], k[5],
                                          k[6], k[7], k[8], k[11], k[10], nd, -beta,
                                          -alpha, np1)
                if sc > best:
                    best = sc
                    best_move = code
                    if sc > alpha:
                        alpha = sc
                        if alpha >= beta:
                            cutoff = True
                            if code >= H_BASE and code != PASS and killers[0] != code:
                                killers[1] = killers[0]
                                killers[0] = code
                            break
        self.path.discard(key)
        if best <= orig_alpha:
            flag = 2
        elif best >= beta:
            flag = 1
        else:
            flag = 0
        _TT[key] = (depth, _to_tt(best, ply), flag, best_move)
        return best

    def _leaf_child(self, a, b, wa, wb, ga, gb, U, D, L, R,
                    hw, vw, da, db, ply):
        """Negated child value for the depth-one fast path."""
        if a & ga:
            return -WIN + ply
        if b & gb:
            return WIN - ply
        if self.turn + ply >= self.max_turns:
            return -_draw(ga)
        if da <= 2 and any(x & ga for x in _pawn_moves(a, b, U, D, L, R)):
            return -WIN + ply + 1
        return -_evalb(da, db, wa, wb, a, b, ga, gb, U, D, L, R, hw, vw)


def _root_bias(child, a, b):
    """Small action preference, not a repetition draw or a subtree value."""
    ia, ib = a.bit_length() - 1, b.bit_length() - 1
    contact = abs(_RC_R[ia] - _RC_R[ib]) + abs(_RC_C[ia] - _RC_C[ib]) <= 2
    if contact:
        # Waiting and revisiting squares can control a straight/diagonal jump.
        return 0
    code = child[1]
    if code == PASS:
        return -ROOT_PASS_PENALTY
    if code < H_BASE:
        if any(child[2] == pawn and child[7] == horizontal and child[8] == vertical
               for _, pawn, horizontal, vertical in _ACTIVE_RECENT):
            return -ROOT_REPEAT_PENALTY
    return 0


def _root_adjust(raw_score, bias):
    # Preserve proven outcome and mate distance exactly.
    return raw_score if abs(raw_score) > MATE_ZONE else raw_score + bias


def _root_raw_bound(bound, bias):
    # Mate values are separated from all heuristic values by a large unused
    # score band. Only heuristic alpha/beta bounds need the inverse offset.
    return bound if abs(bound) > MATE_ZONE else bound - bias


def _root_search(a, b, wa, wb, ga, gb, U, D, L, R, hw, vw, turn, max_turns, t0):
    """Đào sâu lặp ở gốc; trả về mã nước đi tốt nhất."""
    da = _dist(a, ga, U, D, L, R)
    db = _dist(b, gb, U, D, L, R)
    lay_m, dm = _back_layers(ga, a, U, D, L, R, 2)
    win, kids = _gen_pawns(a, b, wa, wb, ga, gb, U, D, L, R, hw, vw, lay_m, dm, db)
    if win is not None:
        return win, None
    if wa > 0:
        kids.extend(_gen_walls(a, b, wa, wb, ga, gb, U, D, L, R, hw, vw,
                               lay_m, dm, 16, True))
    # [TEST1 ENHANCEMENT: EMERGENCY THREAT INTERCEPTION]
    if db <= 2 and wa > 0:
        urgent_walls = [w for w in kids if w[1] >= H_BASE and w[11] > db]
        if urgent_walls:
            # Uu tien tuyet doi cac tuong chan duong ve dich cua doi thu
            kids = urgent_walls + [k for k in kids if k not in urgent_walls]
    kids.append((-_evalb(db, da, wb, wa, b, a, gb, ga, U, D, L, R, hw, vw) - 1, PASS,
                 a, U, D, L, R, hw, vw, wa, da, db))
    biases = {child[1]: _root_bias(child, a, b) for child in kids}
    def root_order(k, scores=None):
        child = (b, k[2], wb, k[9], k[7], k[8], gb)
        value = _root_adjust(k[0], biases[k[1]]) if scores is None else scores.get(k[1], -WIN * 3)
        return (-value, child in _HIST, k[1] == PASS)

    kids.sort(key=root_order)
    best_move = kids[0][1]
    _STATE["depth"] = 1
    _STATE["score"] = _root_adjust(kids[0][0], biases[kids[0][1]])
    if len(kids) == 1:
        return best_move, kids[0]

    deadline = t0 + TIME_LIMIT - _STATE["margin"]
    s = _Searcher(deadline, turn, max_turns)
    depth = 1
    scores = {}
    while depth < 60:
        depth += 1
        it_start = perf_counter()
        alpha = -WIN * 2
        beta = WIN * 2
        cur_best = None
        cur_best_sc = -WIN * 2
        new_scores = {}
        try:
            first = True
            for k in kids:
                bias = biases[k[1]]
                # Root heuristic value includes this action's small bias.
                # Translate every PVS window back to the child's raw scale;
                # TT values remain independent of root preferences/history.
                if first:
                    sc = _root_adjust(-s.search(b, k[2], wb, k[9], gb, ga, k[3], k[4], k[5], k[6],
                                   k[7], k[8], k[11], k[10], depth - 1,
                                   -_root_raw_bound(beta, bias), -_root_raw_bound(alpha, bias), 1), bias)
                    first = False
                else:
                    sc = _root_adjust(-s.search(b, k[2], wb, k[9], gb, ga, k[3], k[4], k[5], k[6],
                                   k[7], k[8], k[11], k[10], depth - 1,
                                   -_root_raw_bound(alpha + 1, bias), -_root_raw_bound(alpha, bias), 1), bias)
                    if sc > alpha:
                        # Đã chắc chắn tốt hơn nước tốt nhất hiện tại: ghi nhận
                        # ngay, phòng khi hết giờ trong lúc tính lại cho chính xác.
                        prev_best, prev_sc = cur_best, cur_best_sc
                        cur_best, cur_best_sc = k[1], sc
                        sc = _root_adjust(-s.search(b, k[2], wb, k[9], gb, ga, k[3], k[4], k[5],
                                       k[6], k[7], k[8], k[11], k[10], depth - 1,
                                       -_root_raw_bound(beta, bias), -_root_raw_bound(alpha, bias), 1), bias)
                        cur_best, cur_best_sc = prev_best, prev_sc
                new_scores[k[1]] = sc
                if sc > cur_best_sc:
                    cur_best_sc = sc
                    cur_best = k[1]
                    if sc > alpha:
                        alpha = sc
        except _TimeUp:
            # Lượt lặp dở dang: nước đầu tiên (tốt nhất cũ) đã xét xong thì dùng
            # kết quả tốt nhất tìm được trong lượt lặp này.
            if cur_best is not None:
                best_move = cur_best
            break
        best_move = cur_best
        scores = new_scores
        _STATE["depth"] = depth
        _STATE["nodes"] = s.nodes
        _STATE["score"] = cur_best_sc
        raw_best = _root_raw_bound(cur_best_sc, biases[best_move])
        _STATE["raw_score"] = raw_best
        # Lượt lặp sau: nước tốt nhất lên đầu, còn lại theo điểm.
        kids.sort(key=lambda k: root_order(k, scores))
        if abs(raw_best) > MATE_ZONE and WIN - abs(raw_best) <= depth:
            break  # thắng/thua chắc chắn đã được chứng minh trong tầm tìm kiếm
        now = perf_counter()
        if now + (now - it_start) * ITER_FACTOR > deadline:
            break  # không đủ thời gian cho độ sâu tiếp theo
    for k in kids:
        if k[1] == best_move:
            return best_move, k
    return best_move, None


# ---------------------------------------------------------------------------
# Giao tiếp với arena
# ---------------------------------------------------------------------------
def _bit(rc):
    return 1 << (rc[0] * 9 + rc[1])


def _fallback(observation):
    """Phương án dự phòng: đi theo bản đồ khoảng cách của arena."""
    moves = observation.get("legal_moves") or []
    dist = observation.get("distances")
    if not moves:
        return {"type": "pass"}
    best = moves[0]
    best_d = INF
    for m in moves:
        d = dist[m[0]][m[1]] if dist else 0
        if 0 <= d < best_d:
            best, best_d = m, d
    return {"type": "move", "to": [int(best[0]), int(best[1])]}


def _decide(observation, t0):
    you = observation["you"]
    opp = observation["opponent"]
    a = _bit(you["position"])
    b = _bit(opp["position"])
    ga = ROW[you["goal_row"]]
    gb = ROW[opp["goal_row"]]
    wa = you["walls_remaining"]
    wb = opp["walls_remaining"]
    U, D, L, R = U0, D0, L0, R0
    hw = vw = 0
    for r, c in observation["walls"]["horizontal"]:
        w = 1 << (r * 9 + c)
        hw |= w
        D &= ~(w | (w << 1))
        U &= ~((w << 9) | (w << 10))
    for r, c in observation["walls"]["vertical"]:
        w = 1 << (r * 9 + c)
        vw |= w
        R &= ~(w | (w << 9))
        L &= ~((w << 1) | (w << 10))
    wa = max(0, min(10, wa))
    wb = max(0, min(10, wb))

    turn = observation.get("turn", 0)
    goals = (ga, gb)
    if turn <= _STATE["last_turn"] or _STATE["goals"] != goals:
        _TT.clear()          # ván mới
        _HIST.clear()
        _ACTIVE_RECENT.clear()
        _STATE["margin"] = 0.0
    elif len(_TT) > 150000:
        _TT.clear()          # bảng quá lớn
    _STATE["last_turn"] = turn
    _STATE["goals"] = goals
    max_turns = observation.get("max_turns", 500)

    check_blocks = (bin(hw).count("1") + bin(vw).count("1")) >= BLOCK_MIN_WALLS
    if check_blocks != _BLOCK[0]:
        # Entries made while the optional blocking evaluator was disabled
        # must not be treated as exact after that evaluator becomes active.
        _TT.clear()
    _BLOCK[0] = check_blocks
    _BCACHE.clear()
    _ROOT_GOAL[0] = ga
    _HIST.add((a, b, wa, wb, hw, vw, ga))
    # Forget old graph/turn contexts. A fresh wall layout can make a retreat
    # necessary, and exact-endgame turns must not leave stale search history.
    _ACTIVE_RECENT[:] = [entry for entry in _ACTIVE_RECENT
                         if entry[2:] == (hw, vw)
                         and 0 < turn - entry[0] <= 2 * ROOT_RECENT_WINDOW]
    _ACTIVE_RECENT.append((turn, a, hw, vw))
    del _ACTIVE_RECENT[:-ROOT_RECENT_WINDOW]
    code, kid = _root_search(a, b, wa, wb, ga, gb, U, D, L, R, hw, vw, turn,
                             max_turns, t0)
    if kid is not None:
        # ghi nhớ thế cờ sau nước đi của mình (đối thủ tới lượt)
        _HIST.add((b, kid[2], wb, kid[9], kid[7], kid[8], gb))
    if code == PASS:
        return {"type": "pass"}
    if code < H_BASE:
        return {"type": "move", "to": [code // 9, code % 9]}
    if code < V_BASE:
        idx = code - H_BASE
        return {"type": "wall", "orientation": "H", "row": idx // 9, "col": idx % 9}
    idx = code - V_BASE
    return {"type": "wall", "orientation": "V", "row": idx // 9, "col": idx % 9}


def _is_legal(action, observation):
    kind = action.get("type")
    if kind == "pass":
        return True
    if kind == "move":
        return list(action["to"]) in [list(m) for m in observation["legal_moves"]]
    if kind == "wall":
        for w in observation["legal_walls"]:
            if (w["orientation"] == action["orientation"] and w["row"] == action["row"]
                    and w["col"] == action["col"]):
                return True
    return False


def choose_action(observation):
    t0 = perf_counter()
    action = None
    try:
        _STATE["moves"] += 1
        if _STATE["moves"] % 25 == 0:
            gc.collect()     # dọn rác định kỳ lúc đầu lượt (GC tự động đã tắt)
        action = _decide(observation, t0)
        if not _is_legal(action, observation):
            action = None
    except Exception:
        action = None
    if action is None:
        action = _fallback(observation)
    # Nếu lượt này bị trễ (máy chấm đang bận), giảm thời gian các lượt sau.
    late = perf_counter() - (t0 + TIME_LIMIT - _STATE["margin"])
    if late > 0.02:
        _STATE["margin"] = min(0.4, _STATE["margin"] + 2 * late)
    return action


WALL_VALUE = [round(value * 1.0) for value in WALL_VALUE]
ITER_FACTOR = 0.3


"""Incremental reverse BFS cache, inlined by build_fusion.py.

The graph and goal, rather than the pawn square, identify a distance map.
No pawn occupancy is included: these are topology distances, not game values.
"""

_PATH_CACHE = {}
_PATH_HITS = 0
_PATH_MISSES = 0


def _cached_layers(goal, target, U, D, L, R, extra):
    global _PATH_HITS, _PATH_MISSES
    key = (goal, U, D, L, R)
    entry = _PATH_CACHE.get(key)
    if entry is None:
        _PATH_MISSES += 1
        # Bound retained memory across games and wall branches.
        if len(_PATH_CACHE) >= 16000:
            _PATH_CACHE.clear()
        entry = [[goal], goal, {}]
        _PATH_CACHE[key] = entry
    else:
        _PATH_HITS += 1
    layers, seen, distances = entry
    hit = distances.get(target)
    if hit is None:
        hit = next((i for i, front in enumerate(layers) if front & target), -1)
    while (hit < 0 or len(layers) <= hit + extra) and layers[-1]:
        front = layers[-1]
        following = (((front & U) >> 9) | ((front & D) << 9)
                     | ((front & L) >> 1) | ((front & R) << 1)) & ~seen
        layers.append(following)
        seen |= following
        if hit < 0 and following & target:
            hit = len(layers) - 1
    entry[1] = seen
    distances[target] = hit
    # Keep the historical API: -1 means no route in backward layers.
    return layers, hit


def _dist(start, goal, U, D, L, R):
    distance = _cached_layers(goal, start, U, D, L, R, 0)[1]
    return distance if distance >= 0 else INF


def _back_layers(goal, target, U, D, L, R, extra):
    return _cached_layers(goal, target, U, D, L, R, extra)


"""Exact, deadline-limited pawn-only Wrongway solver (standard library only).

``solve_action(observation, deadline)`` returns a legal pawn action or ``None``.
The caller can then use its ordinary search. A returned move proves a win when
the opponent has no walls and we still have spare walls. With neither side
holding walls, it preserves the best result under the actual turn limit.
"""

from collections import deque as _eg_deque
from time import perf_counter as _eg_now


_EG_N = 9
_EG_CELLS = 81
_EG_SPAN = _EG_CELLS * _EG_CELLS
_EG_CACHE = None
_EG_VISITS = {}
_EG_CONTEXT = None
_EG_LAST_TURN = -1


class _EGTimeout(Exception):
    pass


def _eg_moves(own, enemy, graph):
    moves = []
    for neighbor in graph[own]:
        if neighbor != enemy:
            moves.append(neighbor)
            continue
        row, col = divmod(enemy, _EG_N)
        own_row, own_col = divmod(own, _EG_N)
        dr, dc = row - own_row, col - own_col
        nr, nc = row + dr, col + dc
        beyond = nr * _EG_N + nc
        if (0 <= nr < _EG_N and 0 <= nc < _EG_N
                and beyond in graph[enemy]):
            moves.append(beyond)
        else:
            for side in graph[enemy]:
                sr, sc = divmod(side, _EG_N)
                if (sr - row) * dr + (sc - col) * dc == 0:
                    moves.append(side)
    return tuple(sorted(set(moves)))


def _eg_graph(walls):
    graph = []
    for cell in range(_EG_CELLS):
        row, col = divmod(cell, _EG_N)
        graph.append([nr * _EG_N + nc
                      for nr, nc in ((row - 1, col), (row, col - 1),
                                     (row, col + 1), (row + 1, col))
                      if 0 <= nr < _EG_N and 0 <= nc < _EG_N])
    for orientation, name in ((0, "horizontal"), (1, "vertical")):
        for row, col in walls[name]:
            first = row * _EG_N + col
            edges = (((first, first + _EG_N),
                      (first + 1, first + _EG_N + 1)) if orientation == 0
                     else ((first, first + 1),
                           (first + _EG_N, first + _EG_N + 1)))
            for a, b in edges:
                if b in graph[a]:
                    graph[a].remove(b)
                    graph[b].remove(a)
    return tuple(tuple(neighbors) for neighbors in graph)


def _eg_distances(graph, goal):
    distances = [999] * _EG_CELLS
    pending = _eg_deque(range(goal * _EG_N, (goal + 1) * _EG_N))
    for cell in pending:
        distances[cell] = 0
    while pending:
        cell = pending.popleft()
        distance = distances[cell] + 1
        for neighbor in graph[cell]:
            if distances[neighbor] == 999:
                distances[neighbor] = distance
                pending.append(neighbor)
    return distances


class _PawnEndgame:
    __slots__ = ("key", "graph", "goals", "maps", "result", "distance")

    def __init__(self, key, walls, goals, deadline):
        self.key, self.goals = key, goals
        self.graph = graph = _eg_graph(walls)
        self.maps = maps = tuple(_eg_distances(graph, goal) for goal in goals)
        count = 2 * _EG_SPAN
        result, distance = [0] * count, [0] * count
        remaining, longest = [0] * count, [0] * count
        previous = [[] for _ in range(count)]
        pending = _eg_deque()
        for p0 in range(_EG_CELLS):
            if _eg_now() >= deadline:
                raise _EGTimeout
            if maps[0][p0] == 999:
                continue
            for p1 in range(_EG_CELLS):
                if p0 == p1 or maps[1][p1] == 999:
                    continue
                winner = (0 if p0 // _EG_N == goals[0] else
                          1 if p1 // _EG_N == goals[1] else None)
                pair = p0 * _EG_CELLS + p1
                for turn in (0, 1):
                    state = turn * _EG_SPAN + pair
                    if winner is not None:
                        result[state] = 1 if winner == turn else -1
                        pending.append(state)
                        continue
                    own, enemy = (p0, p1) if turn == 0 else (p1, p0)
                    destinations = _eg_moves(own, enemy, graph) + (own,)
                    remaining[state] = len(destinations)
                    for destination in destinations:
                        next_pair = (destination * _EG_CELLS + p1 if turn == 0
                                     else p0 * _EG_CELLS + destination)
                        previous[(1 - turn) * _EG_SPAN + next_pair].append(state)
        processed = 0
        while pending:
            state = pending.popleft()
            processed += 1
            if processed % 128 == 0 and _eg_now() >= deadline:
                raise _EGTimeout
            for parent in previous[state]:
                if result[parent]:
                    continue
                if result[state] == -1:
                    result[parent] = 1
                    distance[parent] = distance[state] + 1
                    pending.append(parent)
                else:
                    remaining[parent] -= 1
                    longest[parent] = max(longest[parent], distance[state])
                    if remaining[parent] == 0:
                        result[parent] = -1
                        distance[parent] = longest[parent] + 1
                        pending.append(parent)
        if _eg_now() >= deadline:
            raise _EGTimeout
        self.result, self.distance = result, distance

    def probe(self, p0, p1, turn):
        state = turn * _EG_SPAN + p0 * _EG_CELLS + p1
        return self.result[state], self.distance[state]

    def action(self, p0, p1, turn, turns_left, legal, visits):
        own, enemy = (p0, p1) if turn == 0 else (p1, p0)
        best, best_key = None, None
        for destination in _eg_moves(own, enemy, self.graph) + (own,):
            if destination != own and destination not in legal:
                continue
            np0, np1 = ((destination, p1) if turn == 0 else (p0, destination))
            outcome, distance = self.probe(np0, np1, 1 - turn)
            outcome, distance = -outcome, distance + 1
            if distance > turns_left:
                outcome = 0
            pace = -distance if outcome == 1 else distance if outcome == -1 else 0
            progress = -self.maps[turn][destination]
            pressure = 0.0
            if outcome == 0 and turns_left >= 2:
                replies = _eg_moves(enemy, destination, self.graph) + (enemy,)
                opportunities = 0
                for reply in replies:
                    ap0, ap1 = ((destination, reply) if turn == 0
                                 else (reply, destination))
                    reply_result, reply_distance = self.probe(ap0, ap1, turn)
                    opportunities += (reply_result == 1
                                      and reply_distance <= turns_left - 2)
                pressure = opportunities / len(replies)
            key = (outcome, pace, progress, pressure,
                   -visits.get((destination, enemy), 0), destination != own)
            if best_key is None or key > best_key:
                best, best_key = destination, key
        return best


def solve_action(observation, deadline):
    """Use exact pawn play when safe; ``deadline`` is an absolute perf_counter.

    Cache at most one completed wall layout. Timeout leaves the caller enough
    control to use its own legal fallback; a partially built table is discarded.
    PASS is included because it can change jump tempo and force a draw.
    """
    global _EG_CACHE, _EG_CONTEXT, _EG_LAST_TURN, _EG_VISITS
    if observation["opponent"]["walls_remaining"] != 0 or _eg_now() >= deadline:
        return None
    turn = observation["player_id"]
    goal = observation["you"]["goal_row"]
    for move in observation["legal_moves"]:
        if move[0] == goal:
            return {"type": "move", "to": list(move)}
    turns_left = observation["max_turns"] - observation["turn"]
    if turns_left <= 0:
        return None
    positions, goals = [0, 0], [0, 0]
    for player, info in ((turn, observation["you"]),
                         (1 - turn, observation["opponent"])):
        positions[player] = info["position"][0] * _EG_N + info["position"][1]
        goals[player] = info["goal_row"]
    goals = tuple(goals)
    walls = observation["walls"]
    key = (tuple(sorted(map(tuple, walls["horizontal"]))),
           tuple(sorted(map(tuple, walls["vertical"]))), goals)
    try:
        if _EG_CACHE is None or _EG_CACHE.key != key:
            solved = _PawnEndgame(key, walls, goals, deadline)
            _EG_CACHE = solved
    except _EGTimeout:
        return None
    p0, p1 = positions
    outcome, distance = _EG_CACHE.probe(p0, p1, turn)
    if (observation["you"]["walls_remaining"] != 0
            and not (outcome == 1 and distance <= turns_left)):
        return None
    context = (key, turn)
    if context != _EG_CONTEXT or observation["turn"] <= _EG_LAST_TURN:
        _EG_VISITS = {}
    _EG_CONTEXT, _EG_LAST_TURN = context, observation["turn"]
    own, enemy = positions[turn], positions[1 - turn]
    pair = (own, enemy)
    _EG_VISITS[pair] = min(32, _EG_VISITS.get(pair, 0) + 1)
    if len(_EG_VISITS) > 256:
        _EG_VISITS = {pair: _EG_VISITS[pair]}
    legal = {row * _EG_N + col for row, col in observation["legal_moves"]}
    destination = _EG_CACHE.action(p0, p1, turn, turns_left, legal, _EG_VISITS)
    if destination is None:
        return None
    return ({"type": "pass"} if destination == own else
            {"type": "move", "to": [destination // _EG_N, destination % _EG_N]})



_FUSION_DECIDE = _decide


def _decide(observation, t0):
    # Endgame construction and ordinary search share ONE action deadline.
    if observation["opponent"]["walls_remaining"] == 0 or (observation["you"]["walls_remaining"] == 0 and observation["opponent"]["walls_remaining"] <= 1):
        exact_deadline = min(t0 + TIME_LIMIT - _STATE["margin"] - 0.10,
                             perf_counter() + 0.24)
        action = solve_action(observation, exact_deadline)
        if action is not None and _is_legal(action, observation):
            _STATE["mode"] = "exact_endgame"
            return action
    _STATE["mode"] = "search"
    return _FUSION_DECIDE(observation, t0)

