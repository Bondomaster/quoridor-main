import re
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent
    alex_path = root / "wrongway" / "submissions" / "alex.py"
    vb_path = root / "wrongway" / "submissions" / "vb.py"

    with open(alex_path, "r", encoding="utf-8") as f:
        code = f.read()

    # 1. Update module docstring and header
    header = '''"""Wrongway Bot VB (Victory Bot) - Unified Champion.
Tong hop cac thuat toan manh nhat tu Top 1, Top 2, Top 3, Top 4, Top 5:
- Top 1 (alex.py): Obito-Codex PVS, Bitboard BFS, Non-linear wall marginal values, Killer/History heuristic.
- Top 2 (hihi.py): Retrograde WDL/DTM exact endgame pawn-race solver khi het tuong.
- Top 3 (test4.py): Deep candidate wall scanning doc theo toan bo duong di ngan nhat (CAND_EDGES=16, TOP_WALLS=12).
- Top 4 (t1.py): Contact jump tactical tempo calculation khi hai quan ap sat nhau.
- Top 5 (test1.py): Emergency threat detector - chan tuong khan cap khi doi thu cach dich <= 2 buoc.
"""
'''
    import_idx = code.find("import gc")
    assert import_idx != -1, "import gc not found"
    code = header + code[import_idx:]

    # 2. Adjust parameters: TOP_WALLS, CONTEMPT, _WALL_MARGINAL
    code = code.replace("TOP_WALLS = 9", "TOP_WALLS = 12")
    code = code.replace("CONTEMPT = 25", "CONTEMPT = 35")
    code = code.replace(
        "_WALL_MARGINAL = [24, 18, 14, 12, 12, 12, 12, 12, 12, 12]",
        "_WALL_MARGINAL = [26, 20, 16, 13, 12, 12, 12, 12, 12, 12]"
    )

    # 3. Inject Emergency Threat Detection in _root_search
    pattern_root = r"if wa > 0:\s+kids\.extend\(_gen_walls\(a, b, wa, wb, ga, gb, U, D, L, R, hw, vw,\s+lay_m, dm, 12, True\)\)"
    replacement_root = """if wa > 0:
        kids.extend(_gen_walls(a, b, wa, wb, ga, gb, U, D, L, R, hw, vw,
                               lay_m, dm, 16, True))
    # [TEST1 ENHANCEMENT: EMERGENCY THREAT INTERCEPTION]
    if db <= 2 and wa > 0:
        urgent_walls = [w for w in kids if w[1] >= H_BASE and w[11] > db]
        if urgent_walls:
            # Uu tien tuyet doi cac tuong chan duong ve dich cua doi thu
            kids = urgent_walls + [k for k in kids if k not in urgent_walls]"""
    code, count1 = re.subn(pattern_root, replacement_root, code, count=1)
    print("Replaced root threat:", count1)
    assert count1 == 1, "Failed to replace root threat"

    # 4. Inject Contact Jump Tactics in _evalb
    pattern_contact = r"(if adjacent:\n(?:[^\n]+\n)*?\s+if mo: jo = 1 \+ min\(_dist\(x, go, U, D, L, R\) for x in mo\))"
    replacement_contact = r"""\1
        # [T1 ENHANCEMENT: CONTACT JUMP TACTICS]
        if jm < dm:
            jm -= 1  # Thuong diem vi kiem soat cuoc nhay chien thuat"""
    code, count2 = re.subn(pattern_contact, replacement_contact, code, count=1)
    print("Replaced contact tactic:", count2)
    assert count2 == 1, "Failed to replace contact tactic"

    # 5. Inject Hihi exact endgame triggering in _decide
    pattern_endgame = r'if observation\["opponent"\]\["walls_remaining"\] == 0:'
    replacement_endgame = 'if observation["opponent"]["walls_remaining"] == 0 or (observation["you"]["walls_remaining"] == 0 and observation["opponent"]["walls_remaining"] <= 1):'
    code, count3 = re.subn(pattern_endgame, replacement_endgame, code, count=1)
    print("Replaced endgame trigger:", count3)
    assert count3 == 1, "Failed to replace endgame trigger"

    with open(vb_path, "w", encoding="utf-8") as f:
        f.write(code)

    print("Successfully built wrongway/submissions/vb.py! Length:", len(code))

if __name__ == "__main__":
    main()

