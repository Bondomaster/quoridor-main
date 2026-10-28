import subprocess, sys, re, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

N = 10

matchups = [
    ("test3", "hihi",     "wrongway/submissions/test3.py", "wrongway/submissions/hihi.py"),
    ("test3", "Champion", "wrongway/submissions/test3.py", "wrongway/submissions/Champion_v3_9f1ee8791ef3d8dc.py"),
]

def run(p0, p1):
    cmd = f"python wrongway/arena.py --player1 {p0} --player2 {p1}"
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding="utf-8")
    return (r.stdout.strip().splitlines() or ["?"])[0]

for name_a, name_b, path_a, path_b in matchups:
    wins_a = wins_b = draws = 0
    turns_list = []
    print(f"\n{'='*60}")
    print(f"  {name_a} vs {name_b}  ({N*2} games)")
    print(f"{'='*60}")

    for side in range(2):
        label = f"{name_a}=P0 {name_b}=P1" if side==0 else f"{name_b}=P0 {name_a}=P1"
        print(f"\n  --- Side {side+1}: {label} ---")
        for i in range(N):
            result = run(path_a, path_b) if side == 0 else run(path_b, path_a)
            m = re.search(r"Winner: P(\d) (\S+) after (\d+)", result)
            if m:
                winner_p = int(m.group(1))
                turns    = int(m.group(3))
                turns_list.append(turns)
                winner = name_a if (side==0 and winner_p==0) or (side==1 and winner_p==1) else name_b
                tag = "[WIN ]" if winner == name_a else "[LOSE]"
                if winner == name_a: wins_a += 1
                else:                wins_b += 1
                print(f"  Game {i+1:2d} {tag} {turns:3d}t  {result}")
            else:
                draws += 1
                m2 = re.search(r"after (\d+)", result)
                turns_list.append(int(m2.group(1)) if m2 else 500)
                print(f"  Game {i+1:2d} [DRAW]       {result}")

    total = wins_a + wins_b + draws
    avg   = sum(turns_list)/len(turns_list) if turns_list else 0
    pct   = 100*wins_a/total if total else 0
    print(f"\n  === SUMMARY: {name_a} vs {name_b} ===")
    print(f"  {name_a} W:{wins_a}  {name_b} W:{wins_b}  Draw:{draws}  ({total} games)")
    print(f"  {name_a} winrate: {wins_a}/{total} = {pct:.0f}%  avg turns: {avg:.1f}")
