"""Master Champion Agent for Wrongway Arena (Quoridor).

Thuật toán & Chiến thuật đỉnh cao:
1. Chiến thắng ngay: Nếu có nước đi chạm hàng đích, đi ngay lập tức.
2. Đánh giá thế trận bằng BFS & Đua nhịp độ: Tính chính xác đường ngắn nhất của cả 2 bên.
3. Thuật toán Minimax 1-Ply (Tầm nhìn đối kháng):
   - Trước khi đi bất kỳ nước nào, giả lập xem đối thủ có thể phản đòn bằng bức tường hiểm nào
     cắt đứt đường đi của mình không (counter-wall search).
   - Đảm bảo chọn nước đi mà trong tình huống xấu nhất (khi đối thủ chơi hiểm nhất), ta vẫn giữ lợi thế lớn nhất.
4. Phòng thủ khẩn cấp: Tự động phát hiện khi đối thủ còn <= 2 bước để chặn đứng ngay lập tức.
5. Tiết kiệm tường (Economy): Giữ tường cho các pha then chốt, không rải tường vô ích khi đang dẫn trước.
6. Tối ưu di chuyển: Tận dụng nhảy cờ qua đầu đối thủ và chiếm lĩnh các cột trung tâm.
"""

from collections import deque


def add_wall_edges(blocked, orientation, row, col):
    """Ghi nhận 2 cạnh bị tường chặn vào set."""
    if orientation == "H":
        pairs = [
            ((row, col), (row + 1, col)),
            ((row, col + 1), (row + 1, col + 1)),
        ]
    else:
        pairs = [
            ((row, col), (row, col + 1)),
            ((row + 1, col), (row + 1, col + 1)),
        ]
    for start, end in pairs:
        blocked.add((start, end))
        blocked.add((end, start))


def get_blocked_edges(walls):
    """Tập hợp tất cả các cạnh đang bị tường chặn trên bàn cờ."""
    blocked = set()
    for row, col in walls["horizontal"]:
        add_wall_edges(blocked, "H", row, col)
    for row, col in walls["vertical"]:
        add_wall_edges(blocked, "V", row, col)
    return blocked


def shortest_path_bfs(start, goal_row, size, blocked):
    """BFS nhanh: Trả về (khoảng_cách, danh_sách_các_ô_trên_đường_đi).
    Nếu không tới được đích, trả về (999, []).
    """
    start_tuple = tuple(start)
    if start_tuple[0] == goal_row:
        return 0, [start_tuple]

    queue = deque([start_tuple])
    parent = {start_tuple: None}
    goal_cell = None

    while queue:
        curr = queue.popleft()
        if curr[0] == goal_row:
            goal_cell = curr
            break
        r, c = curr
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                neighbor = (nr, nc)
                if neighbor not in parent and (curr, neighbor) not in blocked:
                    parent[neighbor] = curr
                    queue.append(neighbor)

    if goal_cell is None:
        return 999, []

    path = []
    curr = goal_cell
    while curr is not None:
        path.append(curr)
        curr = parent[curr]
    path.reverse()
    return len(path) - 1, path


def choose_action(observation):
    moves = observation["legal_moves"]
    legal_walls = observation["legal_walls"]
    you = observation["you"]
    opp = observation["opponent"]
    size = observation["board_size"]
    my_goal = you["goal_row"]
    opp_goal = opp["goal_row"]
    my_walls = you["walls_remaining"]
    opp_walls = opp["walls_remaining"]
    my_pos = tuple(you["position"])
    opp_pos = tuple(opp["position"])

    if not moves and not legal_walls:
        return {"type": "pass"}

    # 1. NƯỚC THẮNG NGAY LẬP TỨC: Chạm hàng đích là thắng
    for move in moves:
        if move[0] == my_goal:
            return {"type": "move", "to": list(move)}

    # 2. TÍNH TOÁN ĐƯỜNG ĐI HIỆN TẠI BẰNG BFS
    blocked = get_blocked_edges(observation["walls"])
    my_dist, my_path = shortest_path_bfs(my_pos, my_goal, size, blocked)
    opp_dist, opp_path = shortest_path_bfs(opp_pos, opp_goal, size, blocked)

    # 3. CHỌN LỌC CÁC NƯỚC ĐI VÀ ĐẶT TƯỜNG TIỀM NĂNG (CANDIDATES)
    candidates = []

    # Tất cả các nước di chuyển hợp lệ
    for move in moves:
        candidates.append({"type": "move", "to": list(move)})

    # Lọc các tường có giá trị chiến thuật thực tế
    if my_walls > 0 and legal_walls:
        # Tập hợp các cạnh trên đường đi ngắn nhất của đối thủ
        opp_path_edges = set()
        for idx in range(len(opp_path) - 1):
            opp_path_edges.add((opp_path[idx], opp_path[idx + 1]))
            opp_path_edges.add((opp_path[idx + 1], opp_path[idx]))

        for wall in legal_walls:
            w_o, w_r, w_c = wall["orientation"], wall["row"], wall["col"]
            w_edges = set()
            add_wall_edges(w_edges, w_o, w_r, w_c)

            intersects_opp = bool(w_edges & opp_path_edges)

            # Thử đặt tường ảo để kiểm tra khoảng cách
            cand_blocked = blocked.copy()
            cand_blocked.update(w_edges)

            new_opp_dist, _ = shortest_path_bfs(opp_pos, opp_goal, size, cand_blocked)
            if new_opp_dist >= 900:
                continue

            new_my_dist, _ = shortest_path_bfs(my_pos, my_goal, size, cand_blocked)
            if new_my_dist >= 900:
                continue

            opp_delay = new_opp_dist - opp_dist
            my_penalty = new_my_dist - my_dist
            gain = opp_delay - my_penalty

            # Điều kiện xét tường:
            # - Phòng thủ khẩn cấp (đối thủ cách đích <= 2 bước)
            # - Tường tạo lợi thế lớn (gain >= 2)
            # - Tường chặn ngay đầu đối thủ (gain >= 1 và cắt đường của đối thủ, không tự hại mình)
            is_emergency = (opp_dist <= 2) and (new_opp_dist > opp_dist)
            is_killer_wall = (gain >= 2) and (my_penalty <= 1)
            is_tactical_block = (gain >= 1) and intersects_opp and (my_penalty <= 0)

            if is_emergency or is_killer_wall or is_tactical_block:
                candidates.append({
                    "type": "wall",
                    "orientation": w_o,
                    "row": w_r,
                    "col": w_c,
                    "gain": gain,
                    "opp_delay": opp_delay,
                    "my_penalty": my_penalty,
                })

    # 4. ĐÁNH GIÁ MINIMAX 1-PLY: DỰ ĐOÁN ĐÒN PHẢN CÔNG CỦA ĐỐI THỦ
    best_action = None
    best_score = -999999

    for cand in candidates:
        # Giả lập trạng thái sau khi ta thực hiện hành động này
        if cand["type"] == "move":
            after_my_pos = tuple(cand["to"])
            after_blocked = blocked
            after_my_dist, after_my_path = shortest_path_bfs(after_my_pos, my_goal, size, after_blocked)
            after_opp_dist, after_opp_path = opp_dist, opp_path
        else:
            after_my_pos = my_pos
            after_blocked = blocked.copy()
            add_wall_edges(after_blocked, cand["orientation"], cand["row"], cand["col"])
            after_my_dist, after_my_path = shortest_path_bfs(after_my_pos, my_goal, size, after_blocked)
            after_opp_dist, after_opp_path = shortest_path_bfs(opp_pos, opp_goal, size, after_blocked)

        # Đánh giá cơ bản nếu đối thủ chỉ đi 1 bước tiến về đích
        opp_best_step = max(0, after_opp_dist - 1)
        worst_scenario_eval = (opp_best_step - after_my_dist) * 100

        # Nếu đối thủ còn tường, kiểm tra xem đối thủ có thể phản đòn bằng tường hiểm không
        if opp_walls > 0 and len(after_my_path) > 1:
            my_path_edges = set()
            for idx in range(len(after_my_path) - 1):
                my_path_edges.add((after_my_path[idx], after_my_path[idx + 1]))
                my_path_edges.add((after_my_path[idx + 1], after_my_path[idx]))

            # Tìm bức tường độc nhất mà đối thủ có thể dựng cắt ngang đường của mình
            max_damage = 0
            for w in legal_walls:
                ow_edges = set()
                add_wall_edges(ow_edges, w["orientation"], w["row"], w["col"])
                if after_blocked & ow_edges:
                    continue
                if not (ow_edges & my_path_edges):
                    continue

                test_blocked = after_blocked.copy()
                test_blocked.update(ow_edges)
                d_my, _ = shortest_path_bfs(after_my_pos, my_goal, size, test_blocked)
                if d_my >= 900:
                    continue
                d_opp, _ = shortest_path_bfs(opp_pos, opp_goal, size, test_blocked)
                if d_opp >= 900:
                    continue

                damage = (d_my - after_my_dist) - (d_opp - after_opp_dist)
                if damage > max_damage:
                    max_damage = damage

            # Nếu đối thủ có chiêu làm ta mất >= 2 bước, họ chắc chắn sẽ tung chiêu đó!
            if max_damage >= 2:
                counter_wall_eval = (after_opp_dist - (after_my_dist + max_damage)) * 100
                worst_scenario_eval = min(worst_scenario_eval, counter_wall_eval)

        # Điểm thưởng vị trí & nhịp độ
        total_eval = worst_scenario_eval

        if cand["type"] == "move":
            m_r, m_c = cand["to"]
            # Ưu tiên cột giữa (cột 4) để không bị ép vào biên bàn cờ
            total_eval += (4 - abs(m_c - 4)) * 4
            # Ưu tiên tiến sát hàng đích
            if abs(after_my_pos[0] - my_goal) < abs(my_pos[0] - my_goal):
                total_eval += 20
        else:
            # Thưởng cho tường có sát thương cao
            total_eval += cand["gain"] * 25
            # Ưu tiên tường ở gần vị trí quân đối thủ (càng gần càng hiểm)
            dist_to_opp = abs((cand["row"] + 0.5) - opp_pos[0]) + abs((cand["col"] + 0.5) - opp_pos[1])
            total_eval += max(0, 10 - int(dist_to_opp)) * 4
            # Nếu ta đang dẫn trước an toàn, ưu tiên chạy về đích hơn là lãng phí tường
            if (my_dist < opp_dist - 1) and cand["gain"] < 2:
                total_eval -= 35

        if total_eval > best_score:
            best_score = total_eval
            best_action = cand

    if best_action is not None:
        if best_action["type"] == "wall":
            return {
                "type": "wall",
                "orientation": best_action["orientation"],
                "row": best_action["row"],
                "col": best_action["col"],
            }
        return {"type": "move", "to": list(best_action["to"])}

    return {"type": "pass"}
