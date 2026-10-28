import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUBMISSIONS = ROOT / "wrongway" / "submissions"
ARENA = ROOT / "wrongway" / "arena.py"

opponents = [
    "alex.py",
    "hihi.py",
    "test4.py",
    "t1.py",
    "test1.py"
]

def run(p0, p1, max_turns=120):
    cmd = [
        sys.executable,
        str(ARENA),
        "--player1", str(SUBMISSIONS / p0),
        "--player2", str(SUBMISSIONS / p1),
        "--max-turns", str(max_turns)
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=150)
        out = proc.stdout.strip().splitlines()
        first = out[0] if out else "No output"
        return first, time.time() - t0
    except Exception as e:
        return f"Error: {e}", time.time() - t0

def main():
    print("=" * 60)
    print("BAT DAU KIEM TRA VB.PY VOI TOP 5 BOT")
    print("=" * 60)

    for opp in opponents:
        # Match 1: vb di truoc (P0)
        res1, dur1 = run("vb.py", opp)
        print(f"[P0 vb.py vs P1 {opp}] ({dur1:.1f}s): {res1}")

        # Match 2: vb di sau (P1)
        res2, dur2 = run(opp, "vb.py")
        print(f"[P0 {opp} vs P1 vb.py] ({dur2:.1f}s): {res2}")
        print("-" * 50)

if __name__ == "__main__":
    main()

