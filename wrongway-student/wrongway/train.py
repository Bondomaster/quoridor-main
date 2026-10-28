#!/usr/bin/env python3
"""Trainer for 25127275.py using Genetic Algorithm (GA) & Parallel Self-Play/Tournament.
Tối ưu hóa các trọng số Heuristic của bot 25127275 để đánh bại các đối thủ mạnh.
Chạy: python wrongway/train.py --generations 5 --pop-size 8
"""

import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
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

TARGET_BOT_DEFAULT = "25.py"
CHECKPOINT_FILE = LOGS_DIR / "best_weights.json"

# Tham số mặc định và miền giá trị
PARAM_BOUNDS = {
    "W1_DIST":           (1.0, 5.0, 2.0),
    "W2_WALL":           (0.5, 3.5, 2.6),
    "W3_TEMPO":          (0.1, 1.5, 0.45),
    "W4_MOB":            (0.05, 0.8, 0.15),
    "CAND_EDGES":        (6, 20, 16),
    "OPENING_VARY_PROB": (0.0, 0.6, 0.35),
}


class Individual:
    def __init__(self, genes=None):
        if genes is None:
            self.genes = {}
            for param, (low, high, default) in PARAM_BOUNDS.items():
                if isinstance(default, int):
                    self.genes[param] = random.randint(low, high)
                else:
                    self.genes[param] = round(random.uniform(low, high), 2)
        else:
            self.genes = dict(genes)
        self.fitness = 0.0
        self.stats = {"win": 0, "draw": 0, "loss": 0, "turns": 0}

    def mutate(self, rate=0.35, sigma=0.25):
        for param, (low, high, default) in PARAM_BOUNDS.items():
            if random.random() < rate:
                val = self.genes[param]
                if isinstance(default, int):
                    val += random.choice([-2, -1, 1, 2])
                    val = max(low, min(high, int(val)))
                else:
                    val += random.gauss(0, (high - low) * sigma)
                    val = max(low, min(high, round(val, 2)))
                self.genes[param] = val


def make_candidate_file(cand_id: int, genes: dict, target_bot: Path) -> Path:
    """Tạo file bot tạm chứa bộ tham số của cá thể."""
    with open(target_bot, "r", encoding="utf-8") as f:
        code = f.read()

    # Thay thế các trọng số
    for param, val in genes.items():
        pattern = rf"^{param}\s*=\s*[\d\.]+"
        replacement = f"{param} = {val}"
        code = re.sub(pattern, replacement, code, flags=re.MULTILINE)

    cand_path = SUBMISSIONS_DIR / f"_cand_{cand_id}.py"
    with open(cand_path, "w", encoding="utf-8") as f:
        f.write(code)
    return cand_path


def run_single_match(p0_path: Path, p1_path: Path, max_turns: int = 150) -> dict:
    cmd = [
        sys.executable,
        str(ROOT_DIR / "arena.py"),
        "--player1", str(p0_path),
        "--player2", str(p1_path),
        "--max-turns", str(max_turns),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=60)
        output = res.stdout.strip()
        first_line = output.splitlines()[0] if output else ""

        winner = None
        turns = max_turns
        if "Winner: P0" in first_line:
            winner = 0
        elif "Winner: P1" in first_line:
            winner = 1

        m = re.search(r"after (\d+) turns", first_line)
        if m:
            turns = int(m.group(1))

        return {"winner": winner, "turns": turns, "raw": first_line}
    except Exception as exc:
        return {"winner": None, "turns": max_turns, "error": str(exc)}


def evaluate_individual(cand_id: int, ind: Individual, opponents: list, max_turns: int, target_bot: Path) -> Individual:
    cand_path = make_candidate_file(cand_id, ind.genes, target_bot)
    total_score = 0.0

    try:
        for opp_name in opponents:
            opp_path = SUBMISSIONS_DIR / opp_name
            if not opp_path.exists():
                continue

            # Đấu lượt đi trước (P0)
            res0 = run_single_match(cand_path, opp_path, max_turns=max_turns)
            if res0["winner"] == 0:
                ind.stats["win"] += 1
                total_score += 100.0 + max(0, (max_turns - res0["turns"]) * 0.2)
            elif res0["winner"] is None:
                ind.stats["draw"] += 1
                total_score += 30.0
            else:
                ind.stats["loss"] += 1
            ind.stats["turns"] += res0["turns"]

            # Đấu lượt đi sau (P1)
            res1 = run_single_match(opp_path, cand_path, max_turns=max_turns)
            if res1["winner"] == 1:
                ind.stats["win"] += 1
                total_score += 120.0 + max(0, (max_turns - res1["turns"]) * 0.2)  # thưởng thêm khi thắng lượt sau
            elif res1["winner"] is None:
                ind.stats["draw"] += 1
                total_score += 30.0
            else:
                ind.stats["loss"] += 1
            ind.stats["turns"] += res1["turns"]

    finally:
        if cand_path.exists():
            try:
                cand_path.unlink()
            except Exception:
                pass

    ind.fitness = round(total_score, 1)
    return ind


def apply_best_weights(best_genes: dict, target_bot: Path):
    """Ghi đè bộ tham số tốt nhất vào file bot mục tiêu."""
    backup_path = SUBMISSIONS_DIR / f"{target_bot.stem}_backup.py"
    shutil.copyfile(target_bot, backup_path)

    with open(target_bot, "r", encoding="utf-8") as f:
        code = f.read()

    for param, val in best_genes.items():
        pattern = rf"^{param}\s*=\s*[\d\.]+"
        replacement = f"{param} = {val}"
        code = re.sub(pattern, replacement, code, flags=re.MULTILINE)

    with open(target_bot, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"\n[OK] Đã cập nhật bộ trọng số tốt nhất vào {target_bot.name}!")
    print(f"[OK] File gốc đã được lưu dự phòng tại: {backup_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Train Heuristic Parameters for Bot")
    parser.add_argument("--bot", type=str, default=TARGET_BOT_DEFAULT, help="File bot cần train (mặc định 25.py)")
    parser.add_argument("--generations", type=int, default=5, help="Số thế hệ huấn luyện (mặc định 5)")
    parser.add_argument("--pop-size", type=int, default=8, help="Số cá thể mỗi thế hệ (mặc định 8)")
    parser.add_argument("--opponents", type=str, default="hihi.py,t1.py,minimax_alpha_beta_pruning.py",
                        help="Danh sách đối thủ phân tách bằng dấu phẩy")
    parser.add_argument("--max-turns", type=int, default=120, help="Số lượt tối đa mỗi trận (mặc định 120)")
    parser.add_argument("--workers", type=int, default=6, help="Số luồng song song (mặc định 6)")
    parser.add_argument("--apply", action="store_true", help="Tự động áp dụng bộ trọng số tốt nhất sau khi train")
    args = parser.parse_args()

    target_bot = SUBMISSIONS_DIR / args.bot
    if not target_bot.exists():
        print(f"Lỗi: Không tìm thấy file {target_bot}")
        sys.exit(1)

    if args.opponents.strip().lower() == "all":
        opponents = []
        for p in sorted(SUBMISSIONS_DIR.glob("*.py")):
            if p.name.startswith("_") or p.name in (target_bot.name, "game_state.py", "TEMPLATE.py", "__init__.py"):
                continue
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            if "choose_action" in content:
                opponents.append(p.name)
    else:
        opponents = [o.strip() for o in args.opponents.split(",") if o.strip()]

    print("=" * 65)
    print(f"  TIẾN TRÌNH HUẤN LUYỆN (GENETIC ALGORITHM) CHO {target_bot.name}")
    print(f"  Số thế hệ: {args.generations} | Quy mô quần thể: {args.pop_size}")
    print(f"  Đối thủ: {', '.join(opponents)}")
    print(f"  Số luồng CPU song song: {args.workers}")
    print("=" * 65)

    # Khởi tạo quần thể ban đầu: cá thể 0 luôn là bộ trọng số hiện tại của bot
    population = []
    base_genes = {k: v[2] for k, v in PARAM_BOUNDS.items()}
    population.append(Individual(base_genes))

    for _ in range(args.pop_size - 1):
        ind = Individual()
        population.append(ind)

    best_ever = None

    for gen in range(1, args.generations + 1):
        print(f"\n>>> THẾ HỆ {gen}/{args.generations} (Đang đánh giá {len(population)} cá thể)...")
        gen_start = time.time()

        # Đánh giá song song các cá thể trong quần thể
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(evaluate_individual, idx, ind, opponents, args.max_turns, target_bot)
                for idx, ind in enumerate(population)
            ]
            evaluated_pop = [f.result() for f in futures]

        # Sắp xếp theo Fitness giảm dần
        evaluated_pop.sort(key=lambda x: x.fitness, reverse=True)
        population = evaluated_pop

        best_cur = population[0]
        if best_ever is None or best_cur.fitness > best_ever.fitness:
            best_ever = Individual(best_cur.genes)
            best_ever.fitness = best_cur.fitness
            best_ever.stats = dict(best_cur.stats)

        elapsed = time.time() - gen_start
        print(f"  Top 1 Thế hệ {gen}: Điểm {best_cur.fitness:6.1f} | Thắng {best_cur.stats['win']}/{best_cur.stats['win']+best_cur.stats['loss']} ({best_cur.stats['draw']} Hòa) | Thời gian: {elapsed:.1f}s")
        print(f"  Trọng số tốt nhất: {best_cur.genes}")

        # Lưu checkpoint
        checkpoint_data = {
            "generation": gen,
            "best_fitness": best_ever.fitness,
            "best_genes": best_ever.genes,
            "stats": best_ever.stats,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2)

        if gen == args.generations:
            break

        # TẠO THẾ HỆ TIẾP THEO (Elitism + Crossover + Mutation)
        new_pop = []
        # Elitism: Giữ lại top 2 cá thể xuất sắc nhất
        new_pop.append(Individual(population[0].genes))
        new_pop.append(Individual(population[1].genes))

        while len(new_pop) < args.pop_size:
            # Tournament selection (chọn trong top 4)
            p1 = random.choice(population[:4])
            p2 = random.choice(population[:4])

            # Crossover: lai ghép
            child_genes = {}
            for param in PARAM_BOUNDS:
                child_genes[param] = p1.genes[param] if random.random() < 0.5 else p2.genes[param]

            child = Individual(child_genes)
            child.mutate(rate=0.35, sigma=0.25)
            new_pop.append(child)

        population = new_pop

    print("\n" + "=" * 65)
    print("  KẾT QUẢ HUẤN LUYỆN TỐI ƯU")
    print(f"  Điểm Fitness cao nhất: {best_ever.fitness}")
    print(f"  Tỉ lệ Thắng: {best_ever.stats['win']} | Hòa: {best_ever.stats['draw']} | Thua: {best_ever.stats['loss']}")
    print("  Bộ trọng số tối ưu tìm được:")
    for k, v in best_ever.genes.items():
        print(f"    * {k:12} = {v}")
    print("=" * 65)

    if args.apply:
        apply_best_weights(best_ever.genes, target_bot)
    else:
        print(f"\n[Gợi ý] Để áp dụng bộ trọng số này vào {target_bot.name}, chạy lệnh:")
        print(f"  python train.py --bot {target_bot.name} --apply")


if __name__ == "__main__":
    main()
