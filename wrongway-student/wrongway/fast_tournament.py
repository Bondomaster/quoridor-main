#!/usr/bin/env python3
"""Grand Round-Robin Tournament for Wrongway Bots.
Đấu vòng tròn tính điểm giữa các ứng cử viên vô địch để tìm ra bot mạnh nhất.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT_DIR = Path(__file__).resolve().parent
SUBMISSIONS_DIR = ROOT_DIR / "submissions"
LOGS_DIR = ROOT_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# 10 Bot mạnh nhất trong toàn bộ hệ thống
CANDIDATE_BOTS = [
    "25.py",
    "t1.py",
    "alex.py",
    "hihi.py",
    "ki.py",
    "strong_agent.py",
    "optimized_agent.py",
    "test4.py",
    "test1.py",
    "minimax_alpha_beta_pruning.py"
]

MAX_TURNS = 100
MAX_WORKERS = 12


def run_match(p0_name: str, p1_name: str) -> dict:
    p0_path = SUBMISSIONS_DIR / p0_name
    p1_path = SUBMISSIONS_DIR / p1_name
    cmd = [
        sys.executable,
        str(ROOT_DIR / "arena.py"),
        "--player1", str(p0_path),
        "--player2", str(p1_path),
        "--max-turns", str(MAX_TURNS),
    ]
    t0 = time.time()
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=90)
        output = res.stdout.strip()
        first_line = output.splitlines()[0] if output else ""

        winner = None
        turns = MAX_TURNS

        if "Winner: P0" in first_line:
            winner = p0_name
        elif "Winner: P1" in first_line:
            winner = p1_name

        m = re.search(r"after (\d+) turns", first_line)
        if m:
            turns = int(m.group(1))

        return {
            "p0": p0_name,
            "p1": p1_name,
            "winner": winner,
            "turns": turns,
            "raw": first_line,
            "elapsed": time.time() - t0
        }
    except Exception as exc:
        return {
            "p0": p0_name,
            "p1": p1_name,
            "winner": None,
            "turns": MAX_TURNS,
            "raw": f"Error: {exc}",
            "elapsed": time.time() - t0
        }


def main():
    # Kiểm tra sự tồn tại của các bot
    active_bots = [b for b in CANDIDATE_BOTS if (SUBMISSIONS_DIR / b).exists()]

    matches = []
    for p0 in active_bots:
        for p1 in active_bots:
            if p0 != p1:
                matches.append((p0, p1))

    total_matches = len(matches)
    print("=" * 75)
    print("   GIẢI ĐẤU ĐỈNH CAO: TÌM KIẾM BOT MẠNH NHẤT WRONGWAY")
    print(f"   Số lượng bot tham gia: {len(active_bots)}")
    print(f"   Tổng số trận đấu vòng tròn: {total_matches} trận")
    print(f"   Số luồng CPU xử lý song song: {MAX_WORKERS}")
    print(f"   Giới hạn mỗi trận: {MAX_TURNS} lượt")
    print("=" * 75)

    # Khởi tạo bảng điểm
    table = {b: {"played": 0, "win": 0, "draw": 0, "loss": 0, "pts": 0, "win_turns": []} for b in active_bots}

    start_time = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_match, p0, p1): (p0, p1) for p0, p1 in matches}
        for future in as_completed(futures):
            res = future.result()
            completed += 1
            p0, p1, winner, turns = res["p0"], res["p1"], res["winner"], res["turns"]

            table[p0]["played"] += 1
            table[p1]["played"] += 1

            if winner == p0:
                table[p0]["win"] += 1
                table[p0]["pts"] += 3
                table[p0]["win_turns"].append(turns)
                table[p1]["loss"] += 1
                w_str = f"P0 {p0} THẮNG ({turns}t)"
            elif winner == p1:
                table[p1]["win"] += 1
                table[p1]["pts"] += 3
                table[p1]["win_turns"].append(turns)
                table[p0]["loss"] += 1
                w_str = f"P1 {p1} THẮNG ({turns}t)"
            else:
                table[p0]["draw"] += 1
                table[p0]["pts"] += 1
                table[p1]["draw"] += 1
                table[p1]["pts"] += 1
                w_str = f"HÒA ({turns}t)"

            pct = (completed / total_matches) * 100
            print(f"[{completed:2d}/{total_matches} - {pct:4.1f}%] {p0:15} vs {p1:15} -> {w_str}")

    elapsed_total = time.time() - start_time

    # Xếp hạng: sắp xếp theo Points giảm dần, tiếp theo là số trận thắng, sau đó là tốc độ thắng
    def sort_key(bot_name):
        data = table[bot_name]
        avg_win_turns = (sum(data["win_turns"]) / len(data["win_turns"])) if data["win_turns"] else 999
        return (data["pts"], data["win"], -avg_win_turns)

    ranked_bots = sorted(active_bots, key=sort_key, reverse=True)

    print("\n" + "=" * 85)
    print("                BẢNG XẾP HẠNG CHUNG CUỘC WRONGWAY ARENA")
    print("=" * 85)
    print(f"{'Hạng':<5} {'Tên Bot':<30} {'Trận':<6} {'Thắng':<7} {'Hòa':<6} {'Thua':<6} {'Điểm':<7} {'Tỉ lệ Thắng'}")
    print("-" * 85)

    for rank, b in enumerate(ranked_bots, start=1):
        d = table[b]
        win_rate = (d["win"] / d["played"]) * 100 if d["played"] else 0
        medal = "🥇 " if rank == 1 else ("🥈 " if rank == 2 else ("🥉 " if rank == 3 else f"{rank:2d}. "))
        print(f"{medal:<5} {b:<30} {d['played']:<6} {d['win']:<7} {d['draw']:<6} {d['loss']:<6} {d['pts']:<7} {win_rate:5.1f}%")

    print("=" * 85)
    champion = ranked_bots[0]
    champ_stats = table[champion]
    print(f"\n🏆 QUÁN QUÂN: >>> {champion} <<<")
    print(f"   Tổng điểm: {champ_stats['pts']} điểm | Thắng {champ_stats['win']}/{champ_stats['played']} trận ({champ_stats['draw']} hòa, {champ_stats['loss']} thua)")
    print(f"   Thời gian toàn bộ giải đấu: {elapsed_total:.1f} giây")
    print("=" * 85)


if __name__ == "__main__":
    main()

