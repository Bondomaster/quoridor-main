#!/usr/bin/env python3
"""Wrongway Nova: successor to the supplied 25127055.py engine.
Standard library, single-file submission. Keeps bitboard BFS and selective PVS.
Fixes hypothetical wall evaluation, incomplete-iteration selection and terminal
horizon handling. Uses exact retrograde WDL/DTM at pawn-only roots; search leaf
race values remain explicitly heuristic. Does not claim universal dominance.
"""
from __future__ import annotations

import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

# ============================================================================
# 1. BOARD CONSTANTS & TOPOLOGY
# ============================================================================
R = 9
C = 9
N = 81
W1 = 8
M = 64
TOTAL_MOVES = 209
PASS = 209

INF = 999
WIN = 10 ** 7
MATE_THRESHOLD = WIN - 10000
RACE_WIN_BONUS = 20000

ROW0_MASK = 0x1FF
ROW8_MASK = 0x1FF << 72
GOAL_MASK = (ROW0_MASK, ROW8_MASK)

INIT_OPEN_DN = (1 << 72) - 1
INIT_OPEN_RT = 0
for _r in range(R):
    for _c in range(C - 1):
        INIT_OPEN_RT |= 1 << (_r * C + _c)
INIT_FREE_WMASK = (1 << (2 * M)) - 1

PBIT = tuple(1 << u for u in range(N))
ROW_OF = tuple(u // C for u in range(N))
COL_OF = tuple(u % C for u in range(N))
CEN_BONUS = tuple(4 - abs((u % C) - 4) for u in range(N))
DIST_TO_GOAL_ROW = (
    tuple(u // C for u in range(N)),
    tuple((R - 1) - (u // C) for u in range(N)),
)

MASK_UP = tuple((1 << (u - C)) if u >= C else 0 for u in range(N))
MASK_DN = tuple((1 << u) if u < N - C else 0 for u in range(N))
MASK_LT = tuple((1 << (u - 1)) if (u % C) > 0 else 0 for u in range(N))
MASK_RT = tuple((1 << u) if (u % C) < C - 1 else 0 for u in range(N))

ADJ_CELL_MASK = [0] * N
for _u in range(N):
    _m = 0
    if _u >= C:
        _m |= 1 << (_u - C)
    if _u < N - C:
        _m |= 1 << (_u + C)
    if (_u % C) > 0:
        _m |= 1 << (_u - 1)
    if (_u % C) < C - 1:
        _m |= 1 << (_u + 1)
    ADJ_CELL_MASK[_u] = _m
ADJ_CELL_MASK = tuple(ADJ_CELL_MASK)

WALL_DN_MASK = [0] * (2 * M)
WALL_RT_MASK = [0] * (2 * M)
CONFLICT_MASK = [0] * (2 * M)
CROSS_WALL = [0] * (2 * M)
OVERLAP_WALLS = [[] for _ in range(2 * M)]
WBIT_TO_IDX = {1 << w: w for w in range(2 * M)}
CBIT_TO_IDX = {1 << u: u for u in range(N)}

for _r in range(W1):
    for _c in range(W1):
        _idx = _r * W1 + _c
        _u = _r * C + _c
        WALL_DN_MASK[_idx] = (1 << _u) | (1 << (_u + 1))
        WALL_RT_MASK[M + _idx] = (1 << _u) | (1 << (_u + C))

        CROSS_WALL[_idx] = M + _idx
        CROSS_WALL[M + _idx] = _idx

        _cm_h = (1 << _idx) | (1 << (M + _idx))
        if _c > 0:
            _cm_h |= 1 << (_idx - 1)
            OVERLAP_WALLS[_idx].append(_idx - 1)
        if _c < W1 - 1:
            _cm_h |= 1 << (_idx + 1)
            OVERLAP_WALLS[_idx].append(_idx + 1)
        CONFLICT_MASK[_idx] = _cm_h

        _cm_v = (1 << (M + _idx)) | (1 << _idx)
        if _r > 0:
            _cm_v |= 1 << (M + _idx - W1)
            OVERLAP_WALLS[M + _idx].append(M + _idx - W1)
        if _r < W1 - 1:
            _cm_v |= 1 << (M + _idx + W1)
            OVERLAP_WALLS[M + _idx].append(M + _idx + W1)
        CONFLICT_MASK[M + _idx] = _cm_v

WALL_DN_MASK = tuple(WALL_DN_MASK)
WALL_RT_MASK = tuple(WALL_RT_MASK)
CONFLICT_MASK = tuple(CONFLICT_MASK)
CROSS_WALL = tuple(CROSS_WALL)
OVERLAP_WALLS = tuple(tuple(x) for x in OVERLAP_WALLS)

DN_BIT_DIRECT = {}
DN_BIT_CAND = {}
for _r in range(R - 1):
    for _c in range(C):
        _u = _r * C + _c
        _dir = 0
        _cand = 0
        for _wc in (_c - 1, _c):
            if 0 <= _wc <= W1 - 1:
                _idx = _r * W1 + _wc
                _dir |= 1 << _idx
                _cand |= (1 << _idx) | (1 << (M + _idx))
                if _r > 0:
                    _cand |= 1 << (M + (_r - 1) * W1 + _wc)
        DN_BIT_DIRECT[1 << _u] = _dir
        DN_BIT_CAND[1 << _u] = _cand

RT_BIT_DIRECT = {}
RT_BIT_CAND = {}
for _r in range(R):
    for _c in range(C - 1):
        _u = _r * C + _c
        _dir = 0
        _cand = 0
        for _wr in (_r - 1, _r):
            if 0 <= _wr <= W1 - 1:
                _idx = _wr * W1 + _c
                _dir |= 1 << (M + _idx)
                _cand |= (1 << (M + _idx)) | (1 << _idx)
                if _c > 0:
                    _cand |= 1 << (_wr * W1 + (_c - 1))
        RT_BIT_DIRECT[1 << _u] = _dir
        RT_BIT_CAND[1 << _u] = _cand

# ============================================================================
# STRUCTURAL WALL MASKS
# ============================================================================
_BORDER_MASK = 0
for _w in range(M):
    _r, _c = divmod(_w, W1)
    if _c == 0 or _c == W1 - 1:
        _BORDER_MASK |= 1 << _w
for _w in range(M, 2 * M):
    _r, _c = divmod(_w - M, W1)
    if _r == 0 or _r == W1 - 1:
        _BORDER_MASK |= 1 << _w
BORDER_MASK = _BORDER_MASK

_TOUCH_MASK = [0] * (2 * M)
for _w in range(M):
    _r, _c = divmod(_w, W1)
    if _c >= 2:
        _TOUCH_MASK[_w] |= 1 << (_r * W1 + (_c - 2))
    if _c <= W1 - 3:
        _TOUCH_MASK[_w] |= 1 << (_r * W1 + (_c + 2))
    for _vr in (_r - 1, _r, _r + 1):
        for _vc in (_c - 1, _c, _c + 1):
            if 0 <= _vr < W1 and 0 <= _vc < W1:
                if (_vr, _vc) != (_r, _c):
                    _TOUCH_MASK[_w] |= 1 << (M + _vr * W1 + _vc)

for _w in range(M, 2 * M):
    _idx = _w - M
    _r, _c = divmod(_idx, W1)
    if _r >= 2:
        _TOUCH_MASK[_w] |= 1 << (M + (_r - 2) * W1 + _c)
    if _r <= W1 - 3:
        _TOUCH_MASK[_w] |= 1 << (M + (_r + 2) * W1 + _c)
    for _hr in (_r - 1, _r, _r + 1):
        for _hc in (_c - 1, _c, _c + 1):
            if 0 <= _hr < W1 and 0 <= _hc < W1:
                if (_hr, _hc) != (_r, _c):
                    _TOUCH_MASK[_w] |= 1 << (_hr * W1 + _hc)
TOUCH_MASK = tuple(_TOUCH_MASK)


def wall_value(w: int) -> int:
    """Game-theoretic calibrated quadratic wall valuation.
    Prevents premature wall exhaustion while retaining strategic initiative.
    """
    if w <= 0:
        return 0
    if w > 12:
        w = 12
    return 110 * w - 2 * w * w


WALL_VALUE_TABLE = tuple(wall_value(w) for w in range(16))

# Deterministic 64-bit Zobrist hash keys (Seed = 12345)
_rnd = random.Random(12345)
_r64 = lambda: _rnd.getrandbits(64)
ZW = tuple(_r64() for _ in range(2 * M))
ZP = tuple(tuple(_r64() for _ in range(N)) for _ in range(2))
ZL = tuple(tuple(_r64() for _ in range(32)) for _ in range(2))
ZT = _r64()


# ============================================================================
# 2. TIME MANAGEMENT CONSTANTS & HARD TIMEOUT
# ============================================================================
DEFAULT_T_SOFT = 0.22
DEFAULT_T_HARD = 0.32
TOURNAMENT_T_SOFT = 0.52
TOURNAMENT_T_HARD = 0.72
CHECK_MASK = 127
ABORT_RATIO = 0.48


def get_time_budgets() -> tuple:
    """Dynamically determine soft and hard time budgets from environment or safe defaults."""
    env_val = os.environ.get("WRONGWAY_TIMEOUT")
    if env_val:
        try:
            val = float(env_val)
            if val > 0:
                t_soft = min(0.45, max(0.12, val * 0.45))
                t_hard = min(0.70, max(0.20, val * 0.70))
                return t_soft, t_hard
        except ValueError:
            pass
    return DEFAULT_T_SOFT, DEFAULT_T_HARD


T_SOFT, T_HARD = get_time_budgets()


class HardTimeout(Exception):
    """Raised inside search when time exceeds hard deadline."""
    pass


Timeout = HardTimeout


# ============================================================================
# 3. WAVEFRONT BFS & DUAL-EXTREMAL SHORTEST-PATH TOPOLOGY
# ============================================================================
def fast_bitboard_dist(open_dn: int, open_rt: int, pbit: int, goal_mask: int) -> int:
    """Wavefront bitboard BFS returning shortest path distance to goal_mask, or INF."""
    if pbit & goal_mask:
        return 0
    wave = pbit
    vis = wave
    d = 0
    while wave:
        d += 1
        wave = (((wave & open_dn) << 9) | ((wave >> 9) & open_dn) |
                ((wave & open_rt) << 1) | ((wave >> 1) & open_rt)) & ~vis
        if wave & goal_mask:
            return d
        vis |= wave
    return INF


def fast_bitboard_dist_masked(open_dn: int, open_rt: int, pbit: int, goal_mask: int, blocked_pbit: int = 0) -> int:
    """Wavefront bitboard BFS returning shortest path distance to goal_mask, masking blocked_pbit."""
    if pbit & goal_mask:
        return 0
    wave = pbit
    vis = wave | blocked_pbit
    d = 0
    while wave:
        d += 1
        wave = (((wave & open_dn) << 9) | ((wave >> 9) & open_dn) |
                ((wave & open_rt) << 1) | ((wave >> 1) & open_rt)) & ~vis
        if wave & goal_mask:
            return d
        vis |= wave
    return INF


def compute_dag_node(open_dn: int, open_rt: int, pos: int, s: int) -> Tuple[int, int, int, int, int, int, int]:
    """Computes backward shortest-path layers from goal row to pawn pos and extracts
    candidate walls, direct shortest-path walls, and dual-path intersection cut mask.
    Returns:
      (d, tunnel, sp_deg, first_step, cand_wmask, direct_wmask, cut_isect)
    """
    pbit = PBIT[pos]
    gmask = GOAL_MASK[s]
    if pbit & gmask:
        return (0, 0, 1, 0, 0, 0, 0)
    wave = gmask
    vis = wave
    layers = [wave]
    while True:
        wave = (((wave & open_dn) << 9) | ((wave >> 9) & open_dn) |
                ((wave & open_rt) << 1) | ((wave >> 1) & open_rt)) & ~vis
        if not wave:
            return (INF, 0, 0, 0, 0, 0, 0)
        if wave & pbit:
            break
        vis |= wave
        layers.append(wave)

    d = len(layers)
    blocked_dn = INIT_OPEN_DN ^ open_dn
    blocked_rt = INIT_OPEN_RT ^ open_rt
    wall_adj = blocked_dn | (blocked_dn << 9) | blocked_rt | (blocked_rt << 1)
    no_rt = ~open_rt
    no_dn = ~open_dn
    chute_mask = ((no_rt & (no_rt << 1)) | (no_dn & (no_dn << 9))) & wall_adj

    cur = pbit
    tunnel = 0
    dn_cand = 0
    rt_cand = 0
    dn_dir = 0
    rt_dir = 0
    first_step = 0

    for i in range(d - 1, -1, -1):
        prev = layers[i]
        up_e = (cur >> 9) & open_dn & prev
        dn_e = ((cur & open_dn) << 9) & prev
        lt_e = (cur >> 1) & open_rt & prev
        rt_e = ((cur & open_rt) << 1) & prev

        nxt = up_e | dn_e | lt_e | rt_e
        step_from_pawn = d - 1 - i
        if step_from_pawn < 4 or ((cur | nxt) & wall_adj):
            de = up_e | (dn_e >> 9)
            re = lt_e | (rt_e >> 1)
            dn_cand |= de
            rt_cand |= re
            if step_from_pawn < 3 or ((cur | nxt) & wall_adj):
                dn_dir |= de
                rt_dir |= re

        cur = nxt
        if i == d - 1:
            first_step = cur
        if step_from_pawn < 4 and (cur & (cur - 1)) == 0:
            if cur & chute_mask:
                tunnel += 2
            elif cur & wall_adj:
                tunnel += 1

    if tunnel > 4:
        tunnel = 4
    sp_deg = 2 if (first_step & (first_step - 1)) else 1

    cand_wmask = 0
    while dn_cand:
        lb = dn_cand & -dn_cand
        dn_cand ^= lb
        cand_wmask |= DN_BIT_CAND[lb]
    while rt_cand:
        lb = rt_cand & -rt_cand
        rt_cand ^= lb
        cand_wmask |= RT_BIT_CAND[lb]

    direct_wmask = 0
    while dn_dir:
        lb = dn_dir & -dn_dir
        dn_dir ^= lb
        direct_wmask |= DN_BIT_DIRECT[lb]
    while rt_dir:
        lb = rt_dir & -rt_dir
        rt_dir ^= lb
        direct_wmask |= RT_BIT_DIRECT[lb]

    # Theorem 1 Dual-Extremal Intersection
    cur1 = pbit
    cut1 = 0
    cur2 = pbit
    cut2 = 0
    if s == 0:
        for i in range(d - 1, -1, -1):
            prev = layers[i]
            n1 = (cur1 >> 1) & open_rt & prev
            if n1:
                cut1 |= RT_BIT_DIRECT[n1]
                cur1 = n1
            else:
                n1 = (cur1 >> 9) & open_dn & prev
                if n1:
                    cut1 |= DN_BIT_DIRECT[n1]
                    cur1 = n1
                else:
                    n1 = ((cur1 & open_dn) << 9) & prev
                    if n1:
                        cut1 |= DN_BIT_DIRECT[cur1]
                        cur1 = n1
                    else:
                        n1 = ((cur1 & open_rt) << 1) & prev
                        if n1:
                            cut1 |= RT_BIT_DIRECT[cur1]
                            cur1 = n1

            n2 = ((cur2 & open_rt) << 1) & prev
            if n2:
                cut2 |= RT_BIT_DIRECT[cur2]
                cur2 = n2
            else:
                n2 = (cur2 >> 9) & open_dn & prev
                if n2:
                    cut2 |= DN_BIT_DIRECT[n2]
                    cur2 = n2
                else:
                    n2 = ((cur2 & open_dn) << 9) & prev
                    if n2:
                        cut2 |= DN_BIT_DIRECT[cur2]
                        cur2 = n2
                    else:
                        n2 = (cur2 >> 1) & open_rt & prev
                        if n2:
                            cut2 |= RT_BIT_DIRECT[n2]
                            cur2 = n2
    else:
        for i in range(d - 1, -1, -1):
            prev = layers[i]
            n1 = (cur1 >> 1) & open_rt & prev
            if n1:
                cut1 |= RT_BIT_DIRECT[n1]
                cur1 = n1
            else:
                n1 = ((cur1 & open_dn) << 9) & prev
                if n1:
                    cut1 |= DN_BIT_DIRECT[cur1]
                    cur1 = n1
                else:
                    n1 = (cur1 >> 9) & open_dn & prev
                    if n1:
                        cut1 |= DN_BIT_DIRECT[n1]
                        cur1 = n1
                    else:
                        n1 = ((cur1 & open_rt) << 1) & prev
                        if n1:
                            cut1 |= RT_BIT_DIRECT[cur1]
                            cur1 = n1

            n2 = ((cur2 & open_rt) << 1) & prev
            if n2:
                cut2 |= RT_BIT_DIRECT[cur2]
                cur2 = n2
            else:
                n2 = ((cur2 & open_dn) << 9) & prev
                if n2:
                    cut2 |= DN_BIT_DIRECT[cur2]
                    cur2 = n2
                else:
                    n2 = (cur2 >> 9) & open_dn & prev
                    if n2:
                        cut2 |= DN_BIT_DIRECT[n2]
                        cur2 = n2
                    else:
                        n2 = (cur2 >> 1) & open_rt & prev
                        if n2:
                            cut2 |= RT_BIT_DIRECT[n2]
                            cur2 = n2

    return (d, tunnel, sp_deg, first_step, cand_wmask, direct_wmask, cut1 & cut2)


# ============================================================================
# 4. PERSISTENT GLOBAL CONTEXT (_CTX) & OPPONENT MODELING
# ============================================================================
class OppModel:
    """Online opponent behavior classifier across turns (RUNNER, SPAMMER, IDLE, DEFAULT)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.style: str = "DEFAULT"
        self.opp_turns: int = 0
        self.opp_walls_used: int = 0
        self.opp_moves_used: int = 0
        self.opp_passes: int = 0
        self.opp_walls_remaining: int = 10
        self.recent_actions: List[str] = []

    def _reclassify(self):
        if self.opp_walls_remaining <= 0:
            if self.opp_passes >= 2 and self.opp_moves_used == 0:
                self.style = "IDLE"
            elif self.opp_moves_used >= 2:
                self.style = "RUNNER"
            else:
                self.style = "DEFAULT"
            return

        if self.opp_passes >= 2 and self.opp_passes >= self.opp_moves_used:
            self.style = "IDLE"
        elif (self.opp_walls_used >= 3 and self.opp_walls_used >= self.opp_moves_used) or (
            len(self.recent_actions) >= 3 and self.recent_actions[-3:] == ["wall", "wall", "wall"]
        ):
            self.style = "SPAMMER"
        elif self.opp_moves_used >= 5 and self.opp_walls_used == 0:
            self.style = "RUNNER"
        else:
            self.style = "DEFAULT"

    def update_move(self):
        self.opp_turns += 1
        self.opp_moves_used += 1
        self.recent_actions.append("move")
        if len(self.recent_actions) > 16:
            self.recent_actions.pop(0)
        self._reclassify()

    def update_wall(self):
        self.opp_turns += 1
        self.opp_walls_used += 1
        self.opp_walls_remaining = max(0, self.opp_walls_remaining - 1)
        self.recent_actions.append("wall")
        if len(self.recent_actions) > 16:
            self.recent_actions.pop(0)
        self._reclassify()

    def update_pass(self):
        self.opp_turns += 1
        self.opp_passes += 1
        self.recent_actions.append("pass")
        if len(self.recent_actions) > 16:
            self.recent_actions.pop(0)
        self._reclassify()

    def walls_exhausted(self):
        self.opp_walls_remaining = 0
        if self.style == "SPAMMER":
            self.style = "DEFAULT"
        self._reclassify()

    def update(self, prev: Any, curr: Any = None):
        if isinstance(prev, dict) and "type" in prev:
            for item in (prev, curr):
                if isinstance(item, dict) and "type" in item:
                    atype = item.get("type")
                    if atype == "move":
                        self.update_move()
                    elif atype == "wall":
                        self.update_wall()
                    elif atype == "pass":
                        self.update_pass()
            return

        if isinstance(prev, dict) and isinstance(curr, dict):
            prev_opp = prev.get("opponent", {})
            curr_opp = curr.get("opponent", {})
            prev_pos = prev_opp.get("position")
            curr_pos = curr_opp.get("position")
            prev_w = prev_opp.get("walls_remaining", 10)
            curr_w = curr_opp.get("walls_remaining", 10)
            self.opp_walls_remaining = int(curr_w)
            if curr_w < prev_w:
                self.update_wall()
            elif prev_pos is not None and curr_pos is not None and list(prev_pos) != list(curr_pos):
                self.update_move()
            else:
                self.update_pass()
            if self.opp_walls_remaining <= 0:
                self.walls_exhausted()


class GlobalContext:
    """Manages cross-turn persistent state, bounded caches, and game boundaries."""

    def __init__(self):
        self.tt: Dict[int, Tuple[int, int, int, Any]] = {}
        self.dcache: Dict[Tuple, int] = {}
        self.dagcache: Dict[Tuple, Tuple] = {}
        self.race_cache: Dict[Tuple, int] = {}

        self.game_keys_recent: List[int] = []
        self.game_keys_set: Set[int] = set()
        self.pos_history: List[Tuple[int, int]] = []

        self.last_turn: Optional[int] = None
        self.last_obs: Optional[Dict[str, Any]] = None
        self.match_count: int = 0
        self.opp_model: OppModel = OppModel()

    def reset_new_game(self, obs: Dict[str, Any]):
        """Cleanly reset caches when a new match begins."""
        self.game_keys_recent.clear()
        self.game_keys_set.clear()
        self.pos_history.clear()
        self.race_cache.clear()
        self.tt.clear()
        self.dcache.clear()
        self.dagcache.clear()
        if self.opp_model is not None and hasattr(self.opp_model, "reset"):
            self.opp_model.reset()
        opp = obs.get("opponent", {})
        if isinstance(opp, dict) and "walls_remaining" in opp:
            self.opp_model.opp_walls_remaining = int(opp["walls_remaining"])
            if self.opp_model.opp_walls_remaining <= 0:
                self.opp_model.walls_exhausted()
        self.last_turn = obs.get("turn", 0)
        self.last_obs = obs
        self.match_count += 1

    def update_turn(self, obs: Dict[str, Any]):
        """Detect new game boundary and track turn progression."""
        turn = obs.get("turn", 0)
        walls = obs.get("walls", {})
        walls_h = walls.get("horizontal", [])
        walls_v = walls.get("vertical", [])

        is_new_game = (
            self.last_turn is None
            or turn < self.last_turn
            or (turn <= 1 and len(walls_h) == 0 and len(walls_v) == 0)
        )

        if is_new_game:
            self.reset_new_game(obs)
        else:
            if self.last_obs is not None and self.opp_model is not None:
                self.opp_model.update(self.last_obs, obs)
            self.last_turn = turn
            self.last_obs = obs

    def reset_if_new_game(self, obs: Dict[str, Any]):
        self.update_turn(obs)

    def record_game_key(self, key: int):
        self.game_keys_recent.append(key)
        self.game_keys_set.add(key)
        if len(self.game_keys_recent) > 500:
            old_key = self.game_keys_recent.pop(0)
            if old_key not in self.game_keys_recent:
                self.game_keys_set.discard(old_key)

    def record_position(self, wmask: int, pos: int):
        self.pos_history.append((wmask, pos))
        if len(self.pos_history) > 40:
            self.pos_history.pop(0)

    def clean_cache(self, max_tt: int = 300000):
        if len(self.tt) >= max_tt:
            self.tt = {k: v for k, v in self.tt.items() if v[0] > 2}
            if len(self.tt) > 220000:
                self.tt = {k: v for k, v in self.tt.items() if v[0] > 3}
            if len(self.tt) > 220000:
                self.tt.clear()
        if len(self.dcache) > 200000:
            self.dcache.clear()
        if len(self.dagcache) > 150000:
            self.dagcache.clear()

    def maintain(self, max_tt: int = 300000):
        self.clean_cache(max_tt)


_CTX = GlobalContext()


# Pawn-only leaf scores are estimates; exact retrograde is used at the root.
def pawn_race_estimate(st, side):
    other = 1 - side
    ds = 1 + min((st.fast_d(m, side) for m in st.pawn_moves(side)), default=INF)
    do = 1 + min((st.fast_d(m, other) for m in st.pawn_moves(other)), default=INF)
    return 45 + 100 * (do - ds)


# ============================================================================
# 5. STATE CLASS CORE
# ============================================================================
class State:
    def __init__(self, walls: int = 10):
        self.R, self.C, self.N = R, C, N
        self.W1, self.M = W1, M
        self.goal = [0, R - 1]
        self.pos = [8 * C + 4, 0 * C + 4]
        self.wl = [walls, walls]
        self.turn = 0
        self.ply_count = 0
        self.max_turns = 500

        self.open_dn = INIT_OPEN_DN
        self.open_rt = INIT_OPEN_RT
        self.wmask = 0
        self.free_wmask = INIT_FREE_WMASK

        self.zw = ZW
        self.zp = ZP
        self.zl = ZL
        self.zt = ZT
        self.whash = 0
        self.hash = (
            self.zp[0][self.pos[0]] ^
            self.zp[1][self.pos[1]] ^
            self.zl[0][walls] ^
            self.zl[1][walls]
        )

        self.dcache = _CTX.dcache if _CTX is not None else {}
        self.dagcache = _CTX.dagcache if _CTX is not None else {}
        self.tt = _CTX.tt if _CTX is not None else {}

        self.nodes = 0
        self.deadline_soft = float('inf')
        self.deadline_hard = float('inf')
        self.danger_cache = {}
        self.risk_aware = False
        self.root_allowed = None
        self.mandatory_root = []
        self.search_timeout = False
        self.tt_horizon = 64
        self.horizon_sensitive = True
        self.k_root = 16
        self.k_inner = 10
        self.k_leaf = 6
        self.k_inner_asym = 12
        self.k_leaf_asym = 8
        self.max_threat_extensions = 2
        self.exploit_margin = 60
        self.opp_style = (
            _CTX.opp_model.style
            if (_CTX is not None and getattr(_CTX, "opp_model", None) is not None)
            else "DEFAULT"
        )

        self.killer1 = [-1] * 64
        self.killer2 = [-1] * 64
        self.history = [0] * (TOTAL_MOVES + 1)
        self.pos_hist = ([(0, self.pos[0])], [(0, self.pos[1])])
        self.path_keys = set()
        self.root_side = 0
        self.root_eval = 0
        self.root_move_scores: Dict[int, int] = {}
        self.current_depth_best_move = None
        self.current_depth_best_val = -WIN * 10

    @classmethod
    def from_observation(cls, obs: Dict[str, Any]) -> "State":
        st = cls()
        me = obs.get("player_id", 0)
        op = 1 - me
        st.root_side = me

        you = obs.get("you", {})
        opp = obs.get("opponent", {})

        yr, yc = you.get("position", (8, 4) if me == 0 else (0, 4))
        opr, opc = opp.get("position", (0, 4) if me == 0 else (8, 4))
        st.pos[me] = yr * C + yc
        st.pos[op] = opr * C + opc

        st.wl[me] = you.get("walls_remaining", 10)
        st.wl[op] = opp.get("walls_remaining", 10)

        st.turn = obs.get("current_player", me)
        st.ply_count = obs.get("turn", 0)
        st.max_turns = obs.get("max_turns", 500)
        st.goal = [0, R - 1]

        st.open_dn = INIT_OPEN_DN
        st.open_rt = INIT_OPEN_RT
        st.wmask = 0
        st.free_wmask = INIT_FREE_WMASK
        st.whash = 0

        walls = obs.get("walls", {})
        for r, c in walls.get("horizontal", []):
            w = r * W1 + c
            st.wmask |= (1 << w)
            st.open_dn ^= WALL_DN_MASK[w]
            st.whash ^= ZW[w]
            st.free_wmask &= ~CONFLICT_MASK[w]

        for r, c in walls.get("vertical", []):
            w = M + r * W1 + c
            st.wmask |= (1 << w)
            st.open_rt ^= WALL_RT_MASK[w]
            st.whash ^= ZW[w]
            st.free_wmask &= ~CONFLICT_MASK[w]

        st.hash = (
            st.zp[0][st.pos[0]] ^
            st.zp[1][st.pos[1]] ^
            st.zl[0][st.wl[0]] ^
            st.zl[1][st.wl[1]]
        )
        if st.turn == 1:
            st.hash ^= st.zt

        st.pos_hist = ([(st.wmask, st.pos[0])], [(st.wmask, st.pos[1])])
        return st

    def _place_wall(self, w: int):
        self.open_dn ^= WALL_DN_MASK[w]
        self.open_rt ^= WALL_RT_MASK[w]
        self.wmask ^= 1 << w
        self.free_wmask &= ~CONFLICT_MASK[w]
        self.whash ^= self.zw[w]

    def _wall(self, v: int, i: int, d: int) -> None:
        """Modify wall state in-place (for testing and manual placement)."""
        w = v * M + i
        self.open_dn ^= WALL_DN_MASK[w]
        self.open_rt ^= WALL_RT_MASK[w]
        self.wmask ^= 1 << w
        self.whash ^= self.zw[w]
        if d > 0:
            self.free_wmask &= ~CONFLICT_MASK[w]
        else:
            fw = INIT_FREE_WMASK
            wm = self.wmask
            while wm:
                lb = wm & -wm
                wm ^= lb
                fw &= ~CONFLICT_MASK[WBIT_TO_IDX[lb]]
            self.free_wmask = fw

    def fast_d(self, pos: int, s: int) -> int:
        key = (self.wmask, pos, s)
        d = self.dcache.get(key)
        if d is None:
            d = fast_bitboard_dist(self.open_dn, self.open_rt, PBIT[pos], GOAL_MASK[s])
            self.dcache[key] = d
        return d

    def get_dag(self, pos: int, s: int) -> Tuple[int, int, int, int, int, int, int]:
        key = (self.wmask, pos, s)
        info = self.dagcache.get(key)
        if info is None:
            info = compute_dag_node(self.open_dn, self.open_rt, pos, s)
            self.dagcache[key] = info
            self.dcache[key] = info[0]
        return info

    def is_win_cell(self, s: int, pos: int) -> bool:
        return ROW_OF[pos] == self.goal[s]

    def opp_forward_jump_exists(self, me_cand: int, op: int, o: int) -> bool:
        """Kiểm tra nếu nước đi quân ta tới me_cand tạo cơ hội cho đối thủ tại op nhảy tiến >= 2 bước."""
        if abs(ROW_OF[me_cand] - ROW_OF[op]) + abs(COL_OF[me_cand] - COL_OF[op]) != 1:
            return False
        d_op = self.fast_d(op, o)
        old_me = self.pos[1 - o]
        self.pos[1 - o] = me_cand
        op_moves = self.pawn_moves(o)
        self.pos[1 - o] = old_me
        for q in op_moves:
            if self.fast_d(q, o) <= d_op - 2:
                return True
        return False

    def is_corridor_blocked(self, s: int) -> bool:
        """Detects if player s is trapped or body-blocked in a narrow corridor by the opponent."""
        me = self.pos[s]
        op = self.pos[1 - s]
        mdist = abs(ROW_OF[me] - ROW_OF[op]) + abs(COL_OF[me] - COL_OF[op])
        if mdist > 2:
            return False

        deg_me = (
            ((self.open_dn & MASK_UP[me]) != 0) +
            ((self.open_dn & MASK_DN[me]) != 0) +
            ((self.open_rt & MASK_LT[me]) != 0) +
            ((self.open_rt & MASK_RT[me]) != 0)
        )
        deg_op = (
            ((self.open_dn & MASK_UP[op]) != 0) +
            ((self.open_dn & MASK_DN[op]) != 0) +
            ((self.open_rt & MASK_LT[op]) != 0) +
            ((self.open_rt & MASK_RT[op]) != 0)
        )

        if deg_me <= 1:
            d_masked = fast_bitboard_dist_masked(self.open_dn, self.open_rt, PBIT[me], GOAL_MASK[s], PBIT[op])
            if d_masked >= INF:
                return True

        if deg_op <= 1 and mdist == 1:
            dist_me_to_goal = DIST_TO_GOAL_ROW[s][me]
            dist_op_to_goal = DIST_TO_GOAL_ROW[s][op]
            if dist_op_to_goal < dist_me_to_goal:
                p_moves = self.pawn_moves(s)
                has_forward_jump = any(DIST_TO_GOAL_ROW[s][m] < dist_me_to_goal for m in p_moves)
                if not has_forward_jump:
                    return True

        if mdist <= 2 and deg_me <= 2:
            d_masked = fast_bitboard_dist_masked(self.open_dn, self.open_rt, PBIT[me], GOAL_MASK[s], PBIT[op])
            if d_masked >= INF:
                p_moves = self.pawn_moves(s)
                dist_me_to_goal = DIST_TO_GOAL_ROW[s][me]
                has_forward_jump = any(DIST_TO_GOAL_ROW[s][m] < dist_me_to_goal for m in p_moves)
                if not has_forward_jump:
                    return True

        return False

    def compute_eff_dist(self, s: int) -> int:
        """Computes effective shortest path distance to goal for player s,
        accounting for pawn body blocking in narrow corridors and jump tactics."""
        me = self.pos[s]
        op = self.pos[1 - s]
        d_me = self.fast_d(me, s)
        if d_me == 0:
            return 0

        if self.is_corridor_blocked(s):
            return 1000

        mdist = abs(ROW_OF[me] - ROW_OF[op]) + abs(COL_OF[me] - COL_OF[op])
        if mdist == 1:
            p_moves = self.pawn_moves(s)
            best_after = INF
            for m in p_moves:
                dm = self.fast_d(m, s)
                if dm < best_after:
                    best_after = dm
            if best_after < INF:
                return best_after + 1

        return d_me

    def wall_danger(self, side):
        """Largest legal one-wall net detour, less the opponent's spent turn.
        A positional risk estimate, not a forced-loss claim.
        """
        other = 1 - side
        if self.wl[other] <= 0:
            return 0
        key = (self.wmask, self.pos[0], self.pos[1], side)
        cached = self.danger_cache.get(key)
        if cached is not None:
            return cached
        own = self.get_dag(self.pos[side], side)
        opponent = self.get_dag(self.pos[other], other)
        candidates = own[6] & self.free_wmask
        danger = 0
        while candidates:
            if time.perf_counter() >= self.deadline_hard:
                raise HardTimeout()
            bit = candidates & -candidates
            candidates ^= bit
            wall = bit.bit_length() - 1
            ndn = self.open_dn ^ WALL_DN_MASK[wall]
            nrt = self.open_rt ^ WALL_RT_MASK[wall]
            a = fast_bitboard_dist(ndn, nrt, PBIT[self.pos[side]], GOAL_MASK[side])
            if a >= INF or a <= own[0] + 1:
                continue
            b = opponent[0]
            if opponent[6] & bit:
                b = fast_bitboard_dist(ndn, nrt, PBIT[self.pos[other]], GOAL_MASK[other])
            if b >= INF:
                continue
            danger = max(danger, a-own[0] - (b-opponent[0]) - 1)
        danger = min(8, danger)
        self.danger_cache[key] = danger
        return danger

    def leaf_eval(self, s: Optional[int] = None, p_moves: Optional[List[int]] = None) -> int:
        """Ultra-fast leaf evaluation using pure wavefront BFS (fast_d) without DAG extraction."""
        if s is None:
            s = self.turn
        o = 1 - s
        me, op = self.pos[s], self.pos[o]

        dMe = self.fast_d(me, s)
        dOp = self.fast_d(op, o)
        eff_dMe = dMe
        mdist = abs(ROW_OF[me] - ROW_OF[op]) + abs(COL_OF[me] - COL_OF[op])
        if mdist == 1:
            if p_moves is None:
                p_moves = self.pawn_moves(s)
            best_after = INF
            for m in p_moves:
                dm = self.fast_d(m, s)
                if dm < best_after:
                    best_after = dm
            if best_after < INF:
                eff_dMe = best_after + 1

        if eff_dMe <= 1:
            return WIN - 100

        wMe, wOp = self.wl[s], self.wl[o]
        if wMe == 0 and wOp == 0:
            return pawn_race_estimate(self, s)


        # Heuristic race bonus, never a proof of forced victory.
        if mdist >= 2:
            if wOp == 0 and eff_dMe <= dOp:
                return RACE_WIN_BONUS + 150 * (dOp - eff_dMe) + WALL_VALUE_TABLE[wMe] + (45 * self.wall_danger(o) if self.risk_aware else 0)
            if wMe == 0 and wOp > 0 and dOp < eff_dMe:
                return -RACE_WIN_BONUS + 150 * (dOp - eff_dMe) - WALL_VALUE_TABLE[wOp] - (45 * self.wall_danger(s) if self.risk_aware else 0)

        sc = 45 + 100 * (dOp - eff_dMe)
        sc += WALL_VALUE_TABLE[wMe] - WALL_VALUE_TABLE[wOp]
        uMe = wMe if wMe < dOp else dOp
        uOp = wOp if wOp < eff_dMe else eff_dMe
        sc += 26 * (uMe - uOp)

        if wMe == 0 and wOp > 0:
            sc -= 90
        elif wOp == 0 and wMe > 0:
            sc += 90

        # Row advancement tie-breaker: reward crossing intermediate rows toward goal
        sc += 4 * (DIST_TO_GOAL_ROW[o][op] - DIST_TO_GOAL_ROW[s][me])

        if mdist <= 2 and self.wmask != 0:
            deg_me = (
                ((self.open_dn & MASK_UP[me]) != 0) +
                ((self.open_dn & MASK_DN[me]) != 0) +
                ((self.open_rt & MASK_LT[me]) != 0) +
                ((self.open_rt & MASK_RT[me]) != 0)
            )
            if deg_me == 1 and (ADJ_CELL_MASK[me] & PBIT[op]):
                sc -= 65
            deg_op = (
                ((self.open_dn & MASK_UP[op]) != 0) +
                ((self.open_dn & MASK_DN[op]) != 0) +
                ((self.open_rt & MASK_LT[op]) != 0) +
                ((self.open_rt & MASK_RT[op]) != 0)
            )
            if deg_op == 1 and (ADJ_CELL_MASK[op] & PBIT[me]):
                sc += 65

        if wMe + wOp >= 12:
            sc += 2 * (CEN_BONUS[me] - CEN_BONUS[op])

        if self.risk_aware:
            sc += 45 * (self.wall_danger(o) - self.wall_danger(s))
        return sc

    def evaluate(self, s: Optional[int] = None, dag_s: Optional[Tuple] = None,
                 dag_o: Optional[Tuple] = None, p_moves: Optional[List[int]] = None) -> int:
        if s is None:
            s = self.turn
        o = 1 - s
        me, op = self.pos[s], self.pos[o]
        if dag_s is None:
            dag_s = self.get_dag(me, s)
        if dag_o is None:
            dag_o = self.get_dag(op, o)
        if p_moves is None:
            p_moves = self.pawn_moves(s)

        dMe, tunMe, degMe, firstMe, *rest = dag_s
        dOp, tunOp, degOp, *rest = dag_o
        wMe, wOp = self.wl[s], self.wl[o]

        eff_dMe = dMe
        mdist = abs(ROW_OF[me] - ROW_OF[op]) + abs(COL_OF[me] - COL_OF[op])
        if mdist == 1:
            best_after = INF
            for m in p_moves:
                if PBIT[m] & firstMe:
                    if dMe - 1 < best_after:
                        best_after = dMe - 1
                else:
                    dm = self.fast_d(m, s)
                    if dm < best_after:
                        best_after = dm
            if best_after < INF:
                eff_dMe = best_after + 1

        if eff_dMe <= 1:
            return WIN - 100

        if wMe == 0 and wOp == 0:
            return pawn_race_estimate(self, s)


        # Heuristic race bonus, never a proof of forced victory.
        if mdist >= 2:
            if wOp == 0 and eff_dMe <= dOp:
                return RACE_WIN_BONUS + 150 * (dOp - eff_dMe) + WALL_VALUE_TABLE[wMe]
            if wMe == 0 and wOp > 0 and dOp < eff_dMe:
                return -RACE_WIN_BONUS + 150 * (dOp - eff_dMe) - WALL_VALUE_TABLE[wOp]

        sc = 45 + 100 * (dOp - eff_dMe)
        sc += WALL_VALUE_TABLE[wMe] - WALL_VALUE_TABLE[wOp]
        uMe = wMe if wMe < dOp else dOp
        uOp = wOp if wOp < eff_dMe else eff_dMe
        sc += 26 * (uMe - uOp)

        if wMe == 0 and wOp > 0:
            sc -= 90
        elif wOp == 0 and wMe > 0:
            sc += 90

        sc += 4 * (DIST_TO_GOAL_ROW[o][op] - DIST_TO_GOAL_ROW[s][me])

        if wOp > 0:
            sc -= 6 * tunMe
            if degMe >= 2:
                sc += 10
        if wMe > 0:
            sc += 6 * tunOp
            if degOp >= 2:
                sc -= 10

        if mdist <= 2 and self.wmask != 0:
            if degMe == 1 and (ADJ_CELL_MASK[me] & PBIT[op]):
                sc -= 65
            if degOp == 1 and (ADJ_CELL_MASK[op] & PBIT[me]):
                sc += 65

        if wMe + wOp >= 12:
            sc += 2 * (CEN_BONUS[me] - CEN_BONUS[op])

        return sc

    def draw_value(self, s: int) -> int:
        re = getattr(self, "root_eval", 0)
        if re < -500:
            return 50
        return -150

    def draw_score(self) -> int:
        val = self.draw_value(self.root_side)
        return val if self.turn == self.root_side else -val

    def generate_moves(self, ply: int = 0, depth: int = 2) -> List[int]:
        """Generates ordered candidate moves including pawns, tactical walls, and conditional PASS."""
        s = self.turn
        o = 1 - s
        me, op = self.pos[s], self.pos[o]

        dag_s = self.get_dag(me, s)
        dag_o = self.get_dag(op, o)
        dMe, _, _, first_mask, *rest = dag_s

        p_moves = self.pawn_moves(s)
        goal_dist_s = DIST_TO_GOAL_ROW[s]
        scored_candidates = []

        # 1. Đánh giá và xếp thứ tự nước đi quân (kết hợp Heuristic Chống Nhảy Tiến)
        has_safe_fwd = False
        for m in p_moves:
            dm = (dag_s[0] - 1) if (PBIT[m] & first_mask) else self.fast_d(m, s)
            p_score = 100 * (dMe - dm) + 4 * (goal_dist_s[me] - goal_dist_s[m]) + CEN_BONUS[m]
            if self.opp_forward_jump_exists(m, op, o):
                p_score -= 500  # Phạt nặng nước dâng bước nhảy thẳng/tiến cho đối thủ
            else:
                if dm < dMe:
                    has_safe_fwd = True
            scored_candidates.append((p_score, m))

        # 2. Sinh tường chiến thuật (đánh giá trên cùng thang điểm với nước đi quân)
        mdist = abs(ROW_OF[me] - ROW_OF[op]) + abs(COL_OF[me] - COL_OF[op])
        pass_candidate = False
        if not p_moves:
            pass_candidate = True
        elif self.wl[s] == 0 and self.wl[o] == 0 and mdist <= 2 and not has_safe_fwd:
            pass_candidate = True

        if self.wl[s] > 0:
            if ply == 0:
                k = self.k_root
            elif self.wl[o] == 0:
                k = self.k_inner_asym if depth >= 2 else self.k_leaf_asym
            elif depth >= 4:
                k = 12
            elif depth >= 2:
                k = self.k_inner
            else:
                k = self.k_leaf
            walls, scored_w = self.tactical_walls(s, k, ply=ply, depth=depth, dag_s=dag_s, dag_o=dag_o)
            for sc, net, mv in scored_w:
                scored_candidates.append((sc, mv))

        scored_candidates.sort(reverse=True)
        moves = [item[1] for item in scored_candidates]

        if pass_candidate:
            moves.append(PASS)

        return moves

    def pawn_moves(self, s: int) -> List[int]:
        me, op = self.pos[s], self.pos[1 - s]
        open_dn, open_rt = self.open_dn, self.open_rt
        out = []
        if open_dn & MASK_UP[me]:
            x = me - C
            if x != op:
                out.append(x)
            elif open_dn & MASK_UP[op]:
                out.append(op - C)
            else:
                if open_rt & MASK_LT[op]:
                    out.append(op - 1)
                if open_rt & MASK_RT[op]:
                    out.append(op + 1)
        if open_dn & MASK_DN[me]:
            x = me + C
            if x != op:
                out.append(x)
            elif open_dn & MASK_DN[op]:
                out.append(op + C)
            else:
                if open_rt & MASK_LT[op]:
                    out.append(op - 1)
                if open_rt & MASK_RT[op]:
                    out.append(op + 1)
        if open_rt & MASK_LT[me]:
            x = me - 1
            if x != op:
                out.append(x)
            elif open_rt & MASK_LT[op]:
                out.append(op - 1)
            else:
                if open_dn & MASK_UP[op]:
                    out.append(op - C)
                if open_dn & MASK_DN[op]:
                    out.append(op + C)
        if open_rt & MASK_RT[me]:
            x = me + 1
            if x != op:
                out.append(x)
            elif open_rt & MASK_RT[op]:
                out.append(op + 1)
            else:
                if open_dn & MASK_UP[op]:
                    out.append(op - C)
                if open_dn & MASK_DN[op]:
                    out.append(op + C)
        return out

    def is_legal(self, mv: int) -> bool:
        s = self.turn
        if mv == PASS:
            return True
        if mv < N:
            return mv in self.pawn_moves(s)
        if self.wl[s] <= 0:
            return False
        w = mv - N
        if not (self.free_wmask & (1 << w)):
            return False
        ndn = self.open_dn ^ WALL_DN_MASK[w]
        nrt = self.open_rt ^ WALL_RT_MASK[w]
        return (
            fast_bitboard_dist(ndn, nrt, PBIT[self.pos[0]], ROW0_MASK) < INF and
            fast_bitboard_dist(ndn, nrt, PBIT[self.pos[1]], ROW8_MASK) < INF
        )

    def do(self, mv: int) -> Tuple[int, int, int]:
        s = self.turn
        if mv == PASS:
            self.turn ^= 1
            self.hash ^= self.zt
            return (2, 0, self.free_wmask)

        if mv < N:
            old_pos = self.pos[s]
            self.hash ^= self.zp[s][old_pos] ^ self.zp[s][mv]
            self.pos[s] = mv
            self.pos_hist[s].append((self.wmask, mv))
            info = (0, old_pos, self.free_wmask)
        else:
            w = mv - N
            old_fw = self.free_wmask
            self.open_dn ^= WALL_DN_MASK[w]
            self.open_rt ^= WALL_RT_MASK[w]
            self.wmask ^= 1 << w
            self.free_wmask = old_fw & ~CONFLICT_MASK[w]
            self.whash ^= self.zw[w]
            wleft = self.wl[s]
            self.hash ^= self.zl[s][wleft] ^ self.zl[s][wleft - 1]
            self.wl[s] = wleft - 1
            info = (1, w, old_fw)

        self.turn ^= 1
        self.hash ^= self.zt
        return info

    def undo(self, mv: int, info: Tuple[int, int, int]) -> None:
        self.turn ^= 1
        self.hash ^= self.zt
        if info[0] == 2:
            return

        s = self.turn
        if info[0] == 0:
            old_pos = info[1]
            self.pos_hist[s].pop()
            self.hash ^= self.zp[s][mv] ^ self.zp[s][old_pos]
            self.pos[s] = old_pos
        else:
            w, old_fw = info[1], info[2]
            self.open_dn ^= WALL_DN_MASK[w]
            self.open_rt ^= WALL_RT_MASK[w]
            self.wmask ^= 1 << w
            self.free_wmask = old_fw
            self.whash ^= self.zw[w]
            wleft = self.wl[s]
            self.hash ^= self.zl[s][wleft] ^ self.zl[s][wleft + 1]
            self.wl[s] = wleft + 1

    def tactical_walls(self, s: int, maxK: int, ply: int = 0, depth: int = 2,
                       dag_s: Optional[Tuple] = None, dag_o: Optional[Tuple] = None):
        o = 1 - s
        me, op = self.pos[s], self.pos[o]
        if dag_s is None:
            dag_s = self.get_dag(me, s)
        if dag_o is None:
            dag_o = self.get_dag(op, o)

        dMe, tunMe, degMe, _, candMe, directMe, cutMeIsect = dag_s
        dOp, tunOp, degOp, _, candOp, directOp, cutOpIsect = dag_o

        free = self.free_wmask
        open_dn = self.open_dn
        open_rt = self.open_rt
        wmask = self.wmask
        dcache = self.dcache
        pbit_me = PBIT[me]
        pbit_op = PBIT[op]
        gmask_s = GOAL_MASK[s]
        gmask_o = GOAL_MASK[o]

        # Prophylactic anti-trap defense: trigger when opponent threatens a >= 2 step trap
        proph_bonus = {}
        if self.wl[o] > 0 and (ply == 0 or (depth >= 3 and (tunMe >= 2 or dMe <= dOp))):
            threat_mask = cutMeIsect & free
            worst_gain = 1
            worst_w = -1
            while threat_mask:
                if time.perf_counter() >= self.deadline_hard:
                    raise HardTimeout()
                lb = threat_mask & -threat_mask
                threat_mask ^= lb
                w = WBIT_TO_IDX[lb]
                nwmask = wmask ^ lb
                ndn = open_dn ^ WALL_DN_MASK[w]
                nrt = open_rt ^ WALL_RT_MASK[w]

                k_me = (nwmask, me, s)
                nMe = dcache.get(k_me)
                if nMe is None:
                    nMe = fast_bitboard_dist(ndn, nrt, pbit_me, gmask_s)
                    dcache[k_me] = nMe
                if nMe >= INF or nMe - dMe < 2:
                    continue

                if not (cutOpIsect & lb):
                    nOp = dOp
                else:
                    k_op = (nwmask, op, o)
                    nOp = dcache.get(k_op)
                    if nOp is None:
                        nOp = fast_bitboard_dist(ndn, nrt, pbit_op, gmask_o)
                        dcache[k_op] = nOp
                    if nOp >= INF:
                        continue

                gain = (nMe - dMe) - (nOp - dOp)
                if gain > worst_gain:
                    worst_gain = gain
                    worst_w = w

            if worst_w >= 0 and worst_gain >= 2:
                bval = min(240, worst_gain * 50)
                cw = CROSS_WALL[worst_w]
                if free & (1 << cw):
                    proph_bonus[cw] = bval
                for ow in OVERLAP_WALLS[worst_w]:
                    if free & (1 << ow):
                        proph_bonus[ow] = (bval * 3) // 4

        cand_mask = (free if ply == 0 else candOp & free)
        for pw in proph_bonus:
            cand_mask |= 1 << pw
        k1 = self.killer1[ply] if ply < len(self.killer1) else -1
        k2 = self.killer2[ply] if ply < len(self.killer2) else -1
        if N <= k1 < TOTAL_MOVES and (free & (1 << (k1 - N))):
            cand_mask |= 1 << (k1 - N)
        if N <= k2 < TOTAL_MOVES and (free & (1 << (k2 - N))):
            cand_mask |= 1 << (k2 - N)

        # Feature 17: Category E full wall scan at root
        cat_e_mask = 0
        if ply == 0:
            touching_existing = 0
            wm = wmask
            while wm:
                lb = wm & -wm
                wm ^= lb
                touching_existing |= TOUCH_MASK[WBIT_TO_IDX[lb]]
            structural_mask = (BORDER_MASK | touching_existing) & free

            near_dag_mask = candOp
            co = candOp
            while co:
                lb = co & -co
                co ^= lb
                near_dag_mask |= TOUCH_MASK[WBIT_TO_IDX[lb]]

            cat_e_mask = structural_mask & near_dag_mask
            cand_mask |= cat_e_mask

        scored = []
        hist = self.history
        is_asym = (self.wl[o] == 0)

        while cand_mask:
            if time.perf_counter() >= self.deadline_hard:
                raise HardTimeout()
            lb = cand_mask & -cand_mask
            cand_mask ^= lb
            w = WBIT_TO_IDX[lb]
            nwmask = wmask ^ lb
            ndn = open_dn ^ WALL_DN_MASK[w]
            nrt = open_rt ^ WALL_RT_MASK[w]

            k_op = (nwmask, op, o)
            nOp = dcache.get(k_op)
            if nOp is None:
                nOp = fast_bitboard_dist(ndn, nrt, pbit_op, gmask_o)
                dcache[k_op] = nOp
            if nOp >= INF:
                continue

            d_op_inc = nOp - dOp
            pb = proph_bonus.get(w, 0)
            is_cat_e = bool(cat_e_mask & (1 << w))

            if d_op_inc < 1 and pb == 0 and not is_cat_e and not (ply == 0 and degOp >= 2):
                continue

            # Theorem 1 Invariance: if wall w does not cut me's dual shortest paths,
            # nMe is guaranteed to equal dMe (< INF) in O(1) without BFS!
            if not (cutMeIsect & lb):
                nMe = dMe
            else:
                k_me = (nwmask, me, s)
                nMe = dcache.get(k_me)
                if nMe is None:
                    nMe = fast_bitboard_dist(ndn, nrt, pbit_me, gmask_s)
                    dcache[k_me] = nMe
                if nMe >= INF:
                    continue

            d_me_inc = nMe - dMe
            net = d_op_inc - d_me_inc

            # Loosened filtering
            if d_op_inc >= 1 and (net >= 0 or (is_asym and net >= -1)):
                pass
            elif d_op_inc >= 2 and net >= -1:
                pass
            elif pb > 0 and d_me_inc <= 0:
                pass
            elif is_cat_e and d_me_inc <= 0 and d_op_inc >= 1:
                pb = max(pb, 30)
            elif ply == 0 and d_op_inc == 0 and d_me_inc <= 0 and degOp >= 2:
                ndag_o = compute_dag_node(ndn, nrt, op, o)
                if ndag_o[2] < degOp and ndag_o[1] > tunOp:
                    pb = 35
                else:
                    continue
            else:
                continue

            w_row, w_col = divmod(w % M, W1)
            dist_to_op = abs(w_row - ROW_OF[op]) + abs(w_col - COL_OF[op])
            sprint_mode = (dMe <= 5 and dMe < dOp)
            if sprint_mode and dist_to_op > 2 and pb == 0 and dOp > 2:
                continue
            prox_bonus = max(0, 30 - 10 * dist_to_op) if dist_to_op <= 2 else 0

            mv = N + w
            sc = net * 120 + d_op_inc * 28 + pb + prox_bonus + (hist[mv] >> 4)
            if mv == k1:
                sc += 95
            elif mv == k2:
                sc += 65
            scored.append((sc, net, mv))

        scored.sort(reverse=True)
        return [x[2] for x in scored[:maxK]], scored

    def search(self, depth: int, alpha: int, beta: int, ply: int, extensions: int = 0) -> int:
        self.nodes += 1
        if time.perf_counter() >= self.deadline_hard:
            raise HardTimeout()

        s = self.turn
        o = 1 - s
        key = self.hash ^ self.whash
        tt_key = (key, self.max_turns-self.ply_count-ply) if self.horizon_sensitive else key
        if ROW_OF[self.pos[o]] == self.goal[o]:
            return -WIN + ply
        if ROW_OF[self.pos[s]] == self.goal[s]:
            return WIN - ply

        if self.ply_count + ply >= self.max_turns:
            return self.draw_score()

        p_moves = self.pawn_moves(s)
        g_row = self.goal[s]
        for m in p_moves:
            if ROW_OF[m] == g_row:
                return WIN - ply

        # In-tree cycle detection: penalize cycles inside search branch
        if ply > 0 and key in self.path_keys:
            return self.draw_score()

        # Heuristic pawn-race leaf (exact table is built only at actual endgame roots)
        if ply > 0 and self.wl[0] == 0 and self.wl[1] == 0:
            return pawn_race_estimate(self, s)

        # Pre-Leaf Transposition Table lookup (reuses deeper evaluations even at depth <= 0)
        a0 = alpha
        ttmove = None
        tt_entry = self.tt.get(tt_key)
        if tt_entry is not None:
            tdepth, tval, tflag, ttmove = tt_entry
            if tval >= MATE_THRESHOLD:
                tval -= ply
            elif tval <= -MATE_THRESHOLD:
                tval += ply
            if tdepth >= depth and ply > 0:
                if tflag == 0:
                    return tval
                if tflag == 1 and tval >= beta:
                    return tval
                if tflag == 2 and tval <= alpha:
                    return tval
                if alpha >= beta:
                    return tval

        # Ultra-fast leaf evaluation at depth <= 0
        if depth <= 0:
            return self.leaf_eval(s, p_moves)

        dag_s = self.get_dag(self.pos[s], s)
        dag_o = self.get_dag(self.pos[o], o)

        first_mask = dag_s[3]
        dMe = dag_s[0]
        dOp = dag_o[0]
        fwd_moves = []
        other_moves = []
        goal_dist_s = DIST_TO_GOAL_ROW[s]
        for m in p_moves:
            if PBIT[m] & first_mask:
                fwd_moves.append((dMe - 1, goal_dist_s[m], -CEN_BONUS[m], m))
            else:
                dm = self.fast_d(m, s)
                if dm < dMe:
                    fwd_moves.append((dm, goal_dist_s[m], -CEN_BONUS[m], m))
                else:
                    other_moves.append((dm, goal_dist_s[m], -CEN_BONUS[m], m))
        fwd_moves.sort()
        other_moves.sort()

        fwd_list = [x[3] for x in fwd_moves]
        other_list = [x[3] for x in other_moves]
        wall_net = {}

        if self.wl[s] > 0:
            if ply == 0:
                k = self.k_root
            elif self.wl[o] == 0:
                k = self.k_inner_asym if depth >= 2 else self.k_leaf_asym
            elif depth >= 4:
                k = 12
            elif depth >= 2:
                k = self.k_inner
            else:
                k = self.k_leaf
            walls, scored = self.tactical_walls(s, k, ply=ply, depth=depth, dag_s=dag_s, dag_o=dag_o)
            for _, net, mv in scored:
                wall_net[mv] = net
            if fwd_list:
                order = fwd_list + walls + other_list
            else:
                order = other_list[:1] + walls + other_list[1:]
        else:
            order = fwd_list + other_list

        # Conditional PASS generation in candidate ordering
        mdist = abs(ROW_OF[self.pos[s]] - ROW_OF[self.pos[o]]) + abs(COL_OF[self.pos[s]] - COL_OF[self.pos[o]])
        pass_candidate = False
        if not fwd_list and not other_list:
            pass_candidate = True
        elif self.wl[s] == 0 and self.wl[o] == 0 and mdist <= 2 and not fwd_list:
            pass_candidate = True

        if pass_candidate:
            order.append(PASS)

        if not order:
            return self.leaf_eval(s, p_moves)

        if ply == 0 and self.root_allowed is not None:
            order = [m for m in order if m in self.root_allowed]
            order += [m for m in self.mandatory_root if m not in order]
            if not order:
                return self.leaf_eval(s, p_moves)

        # At root, order moves by exact minimax scores from previous completed iteration
        if ply == 0 and self.root_move_scores:
            order.sort(key=lambda mv: self.root_move_scores.get(mv, -WIN * 9), reverse=True)
        elif ttmove is not None and ttmove in order:
            order.remove(ttmove)
            order.insert(0, ttmove)

        best = -WIN * 10
        bestm = order[0]
        k1 = self.killer1[ply] if ply < len(self.killer1) else -1
        k2 = self.killer2[ply] if ply < len(self.killer2) else -1

        is_near_win = (dMe <= 1 or dOp <= 1)
        new_root_scores: Dict[int, int] = {}

        self.path_keys.add(key)
        try:
            for idx, m in enumerate(order):
                ext = 0
                if extensions < self.max_threat_extensions and idx < 2:
                    if is_near_win:
                        ext = 1
                    elif ply == 0 and m >= N and wall_net.get(m, 0) >= 3:
                        ext = 1

                next_depth = depth - 1 + ext
                next_ext = extensions + ext

                info = self.do(m)
                try:
                    if idx == 0:
                        v = -self.search(next_depth, -beta, -alpha, ply + 1, next_ext)
                    else:
                        v = -self.search(next_depth, -alpha - 1, -alpha, ply + 1, next_ext)
                        if alpha < v < beta:
                            v = -self.search(next_depth, -beta, -alpha, ply + 1, next_ext)
                    if ply == 0 and m < N and m in other_list and hasattr(_CTX, "pos_history"):
                        reps = _CTX.pos_history.count((self.wmask, m))
                        if reps >= 1:
                            v -= 180 * reps
                finally:
                    self.undo(m, info)

                if ply == 0:
                    new_root_scores[m] = v

                if v > best:
                    best, bestm = v, m
                    if ply == 0:
                        self.current_depth_best_move = m
                        self.current_depth_best_val = v
                if v > alpha:
                    alpha = v
                    if alpha >= beta:
                        if ply < len(self.killer1):
                            if self.killer1[ply] != m:
                                self.killer2[ply] = self.killer1[ply]
                                self.killer1[ply] = m
                            if m <= TOTAL_MOVES:
                                self.history[m] += depth * depth
                        break
        finally:
            self.path_keys.remove(key)

        if ply == 0 and new_root_scores:
            self.root_move_scores.update(new_root_scores)

        flag = 2 if best <= a0 else (1 if best >= beta else 0)
        tt_save_val = best + ply if best >= MATE_THRESHOLD else (best - ply if best <= -MATE_THRESHOLD else best)
        self.tt[tt_key] = (depth, tt_save_val, flag, bestm)
        return best

    def greedy_move(self) -> int:
        s = self.turn
        dag_s = self.get_dag(self.pos[s], s)
        first_mask = dag_s[3]
        ms = self.pawn_moves(s)
        best_key = (INF, INF, INF)
        best_m = ms[0] if ms else PASS
        goal_dist_s = DIST_TO_GOAL_ROW[s]
        for m in ms:
            dm = (dag_s[0] - 1) if (PBIT[m] & first_mask) else self.fast_d(m, s)
            cand_key = (dm, goal_dist_s[m], -CEN_BONUS[m])
            if cand_key < best_key:
                best_key = cand_key
                best_m = m
        return best_m

    def verify_safe_exploit(self, candidate_eval: int, v_star: int) -> bool:
        return candidate_eval >= v_star - self.exploit_margin

    def best_move(self, t_start: Optional[float] = None,
                  time_limit: float = T_SOFT,
                  hard_limit: float = T_HARD,
                  max_depth: int = 30) -> Tuple[int, int, int]:
        if t_start is None:
            t_start = time.perf_counter()

        self.deadline_soft = t_start + time_limit
        self.deadline_hard = t_start + hard_limit
        self.nodes = 0
        self.risk_aware = max(self.wall_danger(0), self.wall_danger(1)) >= 3
        self.tt_horizon = max_depth + self.max_threat_extensions + 2
        self.horizon_sensitive = self.max_turns-self.ply_count <= self.tt_horizon
        self.root_move_scores.clear()
        self.tt.clear()  # Scores depend on root history and remaining game horizon.
        _CTX.maintain()

        s = self.turn
        for m in self.pawn_moves(s):
            if ROW_OF[m] == self.goal[s]:
                return m, WIN, 1

        best = self.greedy_move()
        bestv = 0
        reached = 0

        for depth in range(1, max_depth + 1):
            self.current_depth_best_move = None
            self.current_depth_best_val = -WIN * 10

            try:
                if depth >= 4 and abs(bestv) < WIN - 1000:
                    delta = 85
                    a, b = bestv - delta, bestv + delta
                    v = self.search(depth, a, b, 0)
                    if v <= a or v >= b:
                        v = self.search(depth, -WIN * 10, WIN * 10, 0)
                else:
                    v = self.search(depth, -WIN * 10, WIN * 10, 0)
            except HardTimeout:
                self.search_timeout = True
                # Only a completed iteration is comparable across all root actions.
                break

            if self.current_depth_best_move is not None:
                best = self.current_depth_best_move
            else:
                e = self.tt.get((self.hash ^ self.whash, self.max_turns-self.ply_count) if self.horizon_sensitive else self.hash ^ self.whash)
                if e is not None and e[3] is not None:
                    best = e[3]

            bestv = v
            reached = depth
            self.root_eval = bestv

            if v >= WIN - 2:
                break

            elapsed = time.perf_counter() - t_start
            if elapsed > ABORT_RATIO * time_limit:
                break

        return best, bestv, reached


# Exact fixed-board retrograde: all pawn pairs, both turns, and legal PASS.
# status +1/-1/0 = forced win/loss/draw for the player to move.
# Terminal propagation uses minimum win length and maximum forced loss length.
_RACE_TABLE = None


def exact_race_action(st, remaining, deadline):
    global _RACE_TABLE
    from collections import deque
    topology = (st.open_dn, st.open_rt)
    if _RACE_TABLE is None or _RACE_TABLE[0] != topology:
        size = 2 * N * N
        reverse = [[] for _ in range(size)]
        degree = [0] * size
        status = [0] * size
        distance = [0] * size
        longest = [0] * size
        queue = deque()
        saved_pos, saved_turn = st.pos[:], st.turn
        try:
            for p0 in range(N):
                if time.perf_counter() >= deadline:
                    raise HardTimeout()
                for p1 in range(N):
                    if p0 == p1:
                        continue
                    st.pos[:] = (p0, p1)
                    for side in (0, 1):
                        key = (p0 * N + p1) * 2 + side
                        if ROW_OF[st.pos[side]] == st.goal[side]:
                            status[key] = 1
                            queue.append(key)
                            continue
                        if ROW_OF[st.pos[1-side]] == st.goal[1-side]:
                            status[key] = -1
                            queue.append(key)
                            continue
                        moves = st.pawn_moves(side) + [st.pos[side]]
                        degree[key] = len(moves)
                        for move in moves:
                            a, b = (move, p1) if side == 0 else (p0, move)
                            child = (a * N + b) * 2 + 1-side
                            reverse[child].append(key)
            count = 0
            while queue:
                child = queue.popleft()
                count += 1
                if count % 128 == 0 and time.perf_counter() >= deadline:
                    raise HardTimeout()
                for parent in reverse[child]:
                    if status[parent]:
                        continue
                    if status[child] == -1:
                        status[parent] = 1
                        distance[parent] = distance[child] + 1
                        queue.append(parent)
                    else:
                        degree[parent] -= 1
                        longest[parent] = max(longest[parent], distance[child] + 1)
                        if degree[parent] == 0:
                            status[parent] = -1
                            distance[parent] = longest[parent]
                            queue.append(parent)
        finally:
            st.pos[:] = saved_pos
            st.turn = saved_turn
        _RACE_TABLE = (topology, status, distance)
    _, status, distance = _RACE_TABLE
    side = st.turn
    options = []
    for move in st.pawn_moves(side) + [PASS]:
        p = st.pos[:]
        if move != PASS:
            p[side] = move
        child = (p[0] * N + p[1]) * 2 + 1-side
        result, length = -status[child], distance[child] + 1
        if move != PASS and ROW_OF[move] == st.goal[side]:
            result, length = 1, 1
        if length > remaining:
            result = 0
        # Win quickly, delay forced loss, prefer forward progress in draws.
        rank = (result, -length if result > 0 else length if result < 0 else 0,
                -st.fast_d(p[side], side), move != PASS)
        options.append((rank, move))
    return max(options)[1]


# ============================================================================
# 6. ACTION SERIALIZATION & MULTI-LAYER FALLBACKS
# ============================================================================
def action_to_dict(mv: int) -> Dict[str, Any]:
    if mv == PASS or mv is None:
        return {"type": "pass"}
    if mv < N:
        r, c = divmod(mv, C)
        return {"type": "move", "to": [int(r), int(c)]}
    w = mv - N
    v, i = divmod(w, M)
    r, c = divmod(i, W1)
    return {
        "type": "wall",
        "orientation": "H" if v == 0 else "V",
        "row": int(r),
        "col": int(c),
    }


_engine_move_to_action = action_to_dict
action_from_move_id = action_to_dict


def dict_to_action(act: Dict[str, Any]) -> Optional[int]:
    if not isinstance(act, dict):
        return None
    atype = act.get("type")
    if atype == "pass":
        return PASS
    if atype == "move":
        to = act.get("to")
        if isinstance(to, (list, tuple)) and len(to) == 2:
            return int(to[0]) * C + int(to[1])
        return None
    if atype == "wall":
        o = act.get("orientation")
        r, c = act.get("row"), act.get("col")
        if o in ("H", "V") and type(r) is int and type(c) is int:
            v_idx = 0 if o == "H" else 1
            return N + v_idx * M + r * W1 + c
        return None
    return None


def _is_legal_action(action: Any, obs: Dict[str, Any]) -> bool:
    if not isinstance(action, dict):
        return False
    atype = action.get("type")
    if atype == "pass":
        return True
    if atype == "move":
        to = action.get("to")
        if not isinstance(to, (list, tuple)) or len(to) != 2:
            return False
        r, c = to
        if type(r) is not int or type(c) is not int:
            return False
        legal_moves = obs.get("legal_moves", [])
        return [r, c] in legal_moves
    if atype == "wall":
        orient = action.get("orientation")
        r, c = action.get("row"), action.get("col")
        if orient not in ("H", "V") or type(r) is not int or type(c) is not int:
            return False
        for w in obs.get("legal_walls", []):
            if w.get("orientation") == orient and w.get("row") == r and w.get("col") == c:
                return True
        return False
    return False


def _sanitize_and_validate_action(action: Any, obs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(action, dict):
        return None
    atype = action.get("type")
    if atype == "pass":
        return {"type": "pass"}
    if atype == "move":
        to = action.get("to")
        if not isinstance(to, (list, tuple)) or len(to) != 2:
            return None
        try:
            r, c = int(to[0]), int(to[1])
        except (ValueError, TypeError):
            return None
        legal_moves = obs.get("legal_moves", [])
        for lm in legal_moves:
            if isinstance(lm, (list, tuple)) and len(lm) == 2 and int(lm[0]) == r and int(lm[1]) == c:
                return {"type": "move", "to": [int(r), int(c)]}
        return None
    if atype == "wall":
        orient = action.get("orientation")
        if orient not in ("H", "V"):
            return None
        try:
            r, c = int(action.get("row")), int(action.get("col"))
        except (ValueError, TypeError):
            return None
        legal_walls = obs.get("legal_walls", [])
        for lw in legal_walls:
            if (isinstance(lw, dict) and lw.get("orientation") == orient and
                    int(lw.get("row", -1)) == r and int(lw.get("col", -1)) == c):
                return {"type": "wall", "orientation": str(orient), "row": int(r), "col": int(c)}
        return None
    return None


def _get_distance(distances: Any, r: int, c: int) -> int:
    if isinstance(distances, (list, tuple)):
        if 0 <= r < len(distances):
            row_data = distances[r]
            if isinstance(row_data, (list, tuple)) and 0 <= c < len(row_data):
                d = row_data[c]
                if isinstance(d, int) and d >= 0:
                    return d
    elif isinstance(distances, dict):
        for key in ((r, c), f"[{r}, {c}]", f"[{r},{c}]", f"{r},{c}"):
            if key in distances:
                d = distances[key]
                if isinstance(d, int) and d >= 0:
                    return d
    return 9999


def _fallback(obs: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(obs, dict):
        return {"type": "pass"}

    legal_moves = obs.get("legal_moves")
    if isinstance(legal_moves, list) and len(legal_moves) > 0:
        distances = obs.get("distances")
        best_move = None
        best_dist = 9999
        for move in legal_moves:
            if isinstance(move, (list, tuple)) and len(move) == 2:
                try:
                    r, c = int(move[0]), int(move[1])
                    d = _get_distance(distances, r, c)
                    if d < best_dist:
                        best_dist = d
                        best_move = [r, c]
                except (ValueError, TypeError):
                    continue

        if best_move is not None:
            return {"type": "move", "to": [int(best_move[0]), int(best_move[1])]}

        first = legal_moves[0]
        if isinstance(first, (list, tuple)) and len(first) == 2:
            try:
                return {"type": "move", "to": [int(first[0]), int(first[1])]}
            except (ValueError, TypeError):
                pass

    return {"type": "pass"}


_fallback_action = _fallback


# ============================================================================
# 7. OPENING BOOK & MAIN ENTRY POINT: choose_action(observation)
# ============================================================================
OPENING_BOOK: Dict[Tuple[int, Tuple[int, int], Tuple[int, int], Tuple, Tuple], Dict[str, Any]] = {
    (0, (8, 4), (0, 4), (), ()): {"type": "move", "to": [7, 4]},
    (1, (0, 4), (8, 4), (), ()): {"type": "move", "to": [1, 4]},
    (1, (0, 4), (7, 4), (), ()): {"type": "move", "to": [1, 4]},
    (0, (8, 4), (1, 4), (), ()): {"type": "move", "to": [7, 4]},
    (1, (0, 4), (8, 3), (), ()): {"type": "move", "to": [1, 4]},
    (0, (8, 4), (0, 3), (), ()): {"type": "move", "to": [7, 4]},
    (0, (7, 4), (1, 4), (), ()): {"type": "move", "to": [6, 4]},
    (1, (1, 4), (7, 4), (), ()): {"type": "move", "to": [2, 4]},
    (0, (7, 4), (0, 3), (), ()): {"type": "move", "to": [6, 4]},
    (1, (1, 4), (8, 3), (), ()): {"type": "move", "to": [2, 4]},
    # Turn 5 (P0 at 5,4): P1 reaches central row 3
    (1, (2, 4), (5, 4), (), ()): {"type": "move", "to": [3, 4]},
    # Turn 7 Grandmaster P1 Counter-Wall V(4, 3):
    # When P0 drops barricade H(5, 4), P1 immediately drops vertical severing wall V(4, 3)
    (1, (3, 4), (5, 4), ((5, 4),), ()): {"type": "wall", "orientation": "V", "row": 4, "col": 3},
}


def lookup_opening_book(obs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Fast opening book lookup with left-right mirror symmetry (c <-> 8 - c)."""
    turn = obs.get("turn", 0)
    if not isinstance(turn, int) or turn > 8:
        return None
    me = obs.get("player_id")
    if me not in (0, 1):
        return None
    you = obs.get("you", {})
    opp = obs.get("opponent", {})
    y_pos = you.get("position")
    o_pos = opp.get("position")
    if not (isinstance(y_pos, (list, tuple)) and len(y_pos) == 2 and
            isinstance(o_pos, (list, tuple)) and len(o_pos) == 2):
        return None
    yr, yc = int(y_pos[0]), int(y_pos[1])
    opr, opc = int(o_pos[0]), int(o_pos[1])
    walls = obs.get("walls", {})
    wh = tuple(sorted((int(r), int(c)) for r, c in walls.get("horizontal", [])))
    wv = tuple(sorted((int(r), int(c)) for r, c in walls.get("vertical", [])))

    key = (int(me), (yr, yc), (opr, opc), wh, wv)
    act = OPENING_BOOK.get(key)
    if act is not None:
        clean = _sanitize_and_validate_action(act, obs)
        if clean is not None:
            return clean

    # Only apply mirror symmetry for initial development moves (turn <= 4).
    # Tactical reactive walls (turn > 4) must strictly match exact opponent barricade geometry.
    if turn > 4:
        return None

    wh_m = tuple(sorted((r, 7 - c) for r, c in wh))
    wv_m = tuple(sorted((r, 7 - c) for r, c in wv))
    key_m = (int(me), (yr, 8 - yc), (opr, 8 - opc), wh_m, wv_m)
    act_m = OPENING_BOOK.get(key_m)
    if act_m is not None:
        if act_m.get("type") == "move":
            mr, mc = act_m["to"]
            unmirrored = {"type": "move", "to": [int(mr), int(8 - mc)]}
        elif act_m.get("type") == "wall":
            unmirrored = {
                "type": "wall",
                "orientation": act_m["orientation"],
                "row": int(act_m["row"]),
                "col": int(7 - int(act_m["col"])),
            }
        else:
            unmirrored = {"type": "pass"}
        clean = _sanitize_and_validate_action(unmirrored, obs)
        if clean is not None:
            return clean

    return None



# Bot C — Robust Generalist, MSSV 25127055. Diagnostics are process-local only.
LAST_SEARCH = {}


def _root_safety(st, obs):
    side, other = st.turn, 1-st.turn
    moves = list(st.pawn_moves(side))
    walls = [dict_to_action(dict(type="wall", **w)) for w in obs["legal_walls"]]
    actions = moves + walls + [PASS]
    threat = any(ROW_OF[m] == st.goal[other] for m in st.pawn_moves(other))
    safe = []
    for mv in actions:
        info = st.do(mv)
        try:
            winning = mv < N and ROW_OF[mv] == st.goal[side]
            if winning or not any(ROW_OF[m] == st.goal[other] for m in st.pawn_moves(other)):
                safe.append(mv)
        finally:
            st.undo(mv, info)
    st.root_allowed = set(safe) if safe else set(actions)
    st.mandatory_root = safe if threat and safe else []
    return safe, threat


def choose_action(observation):
    global LAST_SEARCH
    started = time.perf_counter()
    LAST_SEARCH = {"nodes": 0, "depth": 0, "fallback": False, "search_timeout": False}
    fallback = _fallback(observation)
    if not isinstance(observation, dict) or observation.get("done"):
        LAST_SEARCH["reason"] = "terminal_or_invalid"
        return {"type": "pass"}
    try:
        goal = observation["you"]["goal_row"]
        for move in observation["legal_moves"]:
            if move[0] == goal:
                LAST_SEARCH["reason"] = "immediate_win"
                return {"type": "move", "to": list(move)}
        _CTX.update_turn(observation)
        _CTX.clean_cache()
        st = State.from_observation(observation)
        _CTX.record_game_key(st.hash ^ st.whash)
        _CTX.record_position(st.wmask, st.pos[st.turn])
        safe, threat = _root_safety(st, observation)
        LAST_SEARCH["immediate_threat"] = threat
        # Initialize a legal, non-losing-in-one-ply fallback before any search.
        safe_moves = [m for m in safe if m < N]
        if safe_moves:
            mv = min(safe_moves, key=lambda m: (st.fast_d(m, st.turn), -CEN_BONUS[m]))
            fallback = action_to_dict(mv)
        elif safe:
            fallback = action_to_dict(safe[0])
        if st.wl[0] == 0 and st.wl[1] == 0:
            try:
                mv = exact_race_action(st, st.max_turns-st.ply_count, started+0.58)
                act = _sanitize_and_validate_action(action_to_dict(mv), observation)
                if act is not None and mv in st.root_allowed:
                    LAST_SEARCH["reason"] = "exact_endgame"
                    return act
            except HardTimeout:
                LAST_SEARCH["search_timeout"] = True
        book = lookup_opening_book(observation)
        if book is not None and dict_to_action(book) in st.root_allowed:
            LAST_SEARCH["reason"] = "opening_book"
            return book
        soft, hard = get_time_budgets()
        if "WRONGWAY_TIMEOUT" not in os.environ:
            soft, hard = TOURNAMENT_T_SOFT, TOURNAMENT_T_HARD
        mv, value, depth = st.best_move(started, soft, hard)
        LAST_SEARCH.update(nodes=st.nodes, depth=depth, value=value,
                           search_timeout=st.search_timeout,
                           cache_entries=len(st.tt)+len(st.dcache)+len(st.dagcache), reason="search")
        act = _sanitize_and_validate_action(action_to_dict(mv), observation)
        if act is not None and mv in st.root_allowed:
            return act
        LAST_SEARCH["reason"] = "unsafe_or_invalid_search_result"
    except Exception as exc:
        LAST_SEARCH["reason"] = type(exc).__name__ + ": " + str(exc)
    finally:
        LAST_SEARCH["seconds"] = time.perf_counter()-started
    LAST_SEARCH["fallback"] = True
    return fallback
