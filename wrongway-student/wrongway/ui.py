#!/usr/bin/env python3
"""Wrongway Visual Arena - Giao diện trực quan mô phỏng trận đấu giữa các bot.
Không cần cài thêm thư viện (chỉ dùng Python Standard Library).
Chạy lệnh: python wrongway/ui.py
"""

import http.server
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.parse
import webbrowser

PORT = 8765
ROOT_DIR = Path(__file__).resolve().parent
SUBMISSIONS_DIR = ROOT_DIR / "submissions"
LOGS_DIR = ROOT_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

HTML_PAGE = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Wrongway Arena Visualizer</title>
  <style>
    :root {
      --bg: #0f172a;
      --panel: #1e293b;
      --panel-border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --p0-color: #38bdf8;
      --p1-color: #f43f5e;
      --wall-color: #f59e0b;
      --wall-highlight: #fbbf24;
      --board-bg: #0b1120;
      --tile-bg: #1e293b;
      --tile-hover: #334155;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    header {
      background: var(--panel);
      border-bottom: 1px solid var(--panel-border);
      padding: 12px 24px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 1.25rem;
      font-weight: 700;
      color: #38bdf8;
    }

    .selectors-bar {
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }

    .player-select-group {
      display: flex;
      align-items: center;
      gap: 8px;
      background: rgba(0,0,0,0.25);
      padding: 6px 12px;
      border-radius: 8px;
      border: 1px solid var(--panel-border);
    }

    .p-tag {
      font-weight: 700;
      font-size: 0.85rem;
      padding: 2px 8px;
      border-radius: 4px;
    }
    .p0-tag { background: rgba(56, 189, 248, 0.2); color: var(--p0-color); }
    .p1-tag { background: rgba(244, 63, 94, 0.2); color: var(--p1-color); }

    select, input, button {
      background: #0f172a;
      color: var(--text);
      border: 1px solid var(--panel-border);
      padding: 7px 12px;
      border-radius: 6px;
      font-size: 0.9rem;
      outline: none;
    }

    select:focus, input:focus { border-color: #38bdf8; }

    .btn-battle {
      background: linear-gradient(135deg, #0284c7, #2563eb);
      color: white;
      font-weight: 600;
      border: none;
      cursor: pointer;
      padding: 8px 18px;
      border-radius: 6px;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s;
    }
    .btn-battle:hover { filter: brightness(1.15); transform: translateY(-1px); }
    .btn-battle:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

    main {
      flex: 1;
      display: grid;
      grid-template-columns: auto 340px;
      gap: 20px;
      padding: 20px;
      max-width: 1400px;
      margin: 0 auto;
      width: 100%;
    }

    @media (max-width: 1000px) {
      main { grid-template-columns: 1fr; }
    }

    .board-container {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 16px;
    }

    .status-banner {
      width: 100%;
      max-width: 530px;
      background: var(--panel);
      border: 1px solid var(--panel-border);
      padding: 12px 16px;
      border-radius: 10px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .player-card {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .player-card.p1 { align-items: flex-end; text-align: right; }
    .player-name { font-weight: 700; font-size: 1rem; }
    .player-sub { font-size: 0.8rem; color: var(--text-muted); }

    /* BÀN CỜ QUORIDOR 9x9 */
    .board-wrapper {
      position: relative;
      background: var(--board-bg);
      padding: 12px;
      border-radius: 12px;
      box-shadow: 0 10px 30px -10px rgba(0,0,0,0.7);
      border: 1px solid var(--panel-border);
    }

    .board {
      position: relative;
      width: 510px;
      height: 510px;
    }

    .tile {
      position: absolute;
      width: 46px;
      height: 46px;
      background: var(--tile-bg);
      border-radius: 6px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.75rem;
      color: rgba(255,255,255,0.1);
      transition: background 0.15s;
    }
    .tile:hover { background: var(--tile-hover); }

    .tile.goal-p0 { border-top: 2px dashed rgba(56, 189, 248, 0.4); }
    .tile.goal-p1 { border-bottom: 2px dashed rgba(244, 63, 94, 0.4); }

    .pawn {
      position: absolute;
      width: 38px;
      height: 38px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 0.85rem;
      color: #fff;
      z-index: 10;
      transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1);
      box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }
    .pawn-0 { background: radial-gradient(circle at 30% 30%, #38bdf8, #0369a1); border: 2px solid #bae6fd; }
    .pawn-1 { background: radial-gradient(circle at 30% 30%, #fb7185, #be123c); border: 2px solid #fecdd3; }

    .wall {
      position: absolute;
      background: var(--wall-color);
      border-radius: 3px;
      z-index: 5;
      box-shadow: 0 2px 6px rgba(0,0,0,0.6);
      transition: transform 0.2s;
    }
    .wall.horizontal {
      height: 10px;
      width: 104px;
    }
    .wall.vertical {
      width: 10px;
      height: 104px;
    }
    .wall.newest {
      background: var(--wall-highlight);
      box-shadow: 0 0 10px rgba(251, 191, 36, 0.8);
      animation: pulse-wall 1s infinite alternate;
    }

    @keyframes pulse-wall {
      from { transform: scale(1); }
      to { transform: scale(1.08); }
    }

    /* CONTROLS BAR */
    .controls-panel {
      width: 100%;
      max-width: 530px;
      background: var(--panel);
      border: 1px solid var(--panel-border);
      padding: 14px;
      border-radius: 10px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .slider-row {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .turn-slider {
      flex: 1;
      accent-color: #38bdf8;
      cursor: pointer;
    }
    .turn-label {
      font-size: 0.85rem;
      font-weight: 600;
      min-width: 85px;
      text-align: right;
    }

    .btn-row {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
    }

    .ctrl-btn {
      background: #0f172a;
      border: 1px solid var(--panel-border);
      padding: 8px 14px;
      border-radius: 6px;
      cursor: pointer;
      font-weight: 600;
      font-size: 0.95rem;
      transition: background 0.15s;
    }
    .ctrl-btn:hover { background: #334155; }

    .speed-select {
      margin-left: auto;
      font-size: 0.85rem;
    }

    /* MOVE LOG PANEL */
    .side-panel {
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 10px;
      display: flex;
      flex-direction: column;
      height: 670px;
      overflow: hidden;
    }

    .side-panel-header {
      padding: 12px 16px;
      font-weight: 700;
      border-bottom: 1px solid var(--panel-border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .move-list {
      flex: 1;
      overflow-y: auto;
      padding: 8px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }

    .move-item {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px 10px;
      border-radius: 6px;
      font-size: 0.85rem;
      cursor: pointer;
      transition: background 0.1s;
    }
    .move-item:hover { background: rgba(255,255,255,0.05); }
    .move-item.active {
      background: rgba(56, 189, 248, 0.15);
      border: 1px solid rgba(56, 189, 248, 0.4);
    }

    .move-turn {
      color: var(--text-muted);
      font-weight: 600;
      min-width: 55px;
    }

    .winner-banner {
      padding: 10px;
      text-align: center;
      font-weight: 700;
      border-radius: 6px;
      margin: 8px;
      background: rgba(34, 197, 94, 0.2);
      border: 1px solid rgba(34, 197, 94, 0.5);
      color: #4ade80;
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <span>♟️ Wrongway Visual Arena</span>
    </div>
    <div class="selectors-bar">
      <div class="player-select-group">
        <span class="p-tag p0-tag">P0 (Đi trước)</span>
        <select id="p0-select"></select>
      </div>
      <span style="font-weight:800; color:var(--text-muted)">VS</span>
      <div class="player-select-group">
        <span class="p-tag p1-tag">P1 (Đi sau)</span>
        <select id="p1-select"></select>
      </div>
      <button class="btn-battle" id="start-btn" onclick="startMatch()">
        <span>⚔️ Bắt đầu đấu</span>
      </button>
    </div>
  </header>

  <main>
    <div class="board-container">
      <div class="status-banner">
        <div class="player-card">
          <div class="player-name" style="color:var(--p0-color)" id="p0-name">Player 0</div>
          <div class="player-sub" id="p0-info">Vị trí: [8, 4] | Tường: 10/10</div>
        </div>
        <div style="font-size:0.85rem; font-weight:700; color:var(--text-muted); text-align:center" id="match-summary">
          Chưa có trận đấu
        </div>
        <div class="player-card p1">
          <div class="player-name" style="color:var(--p1-color)" id="p1-name">Player 1</div>
          <div class="player-sub" id="p1-info">Vị trí: [0, 4] | Tường: 10/10</div>
        </div>
      </div>

      <div class="board-wrapper">
        <div class="board" id="board"></div>
      </div>

      <div class="controls-panel">
        <div class="slider-row">
          <input type="range" class="turn-slider" id="turn-slider" min="0" max="0" value="0" oninput="goToTurn(parseInt(this.value))">
          <span class="turn-label" id="turn-label">Lượt 0 / 0</span>
        </div>
        <div class="btn-row">
          <button class="ctrl-btn" onclick="goToTurn(0)" title="Về đầu trận">⏮️</button>
          <button class="ctrl-btn" onclick="stepTurn(-1)" title="Lùi 1 bước">◀</button>
          <button class="ctrl-btn" id="play-btn" onclick="togglePlay()" style="min-width:90px">▶ Phát</button>
          <button class="ctrl-btn" onclick="stepTurn(1)" title="Tiến 1 bước">▶</button>
          <button class="ctrl-btn" onclick="goToTurn(maxTurns)" title="Tới cuối trận">⏭️</button>
          <select class="speed-select" id="speed-select" onchange="updateSpeed()">
            <option value="1000">Tốc độ: 1.0s</option>
            <option value="400" selected>Tốc độ: 0.4s</option>
            <option value="180">Tốc độ: 0.2s</option>
            <option value="60">Tốc độ: Siêu tốc</option>
          </select>
        </div>
      </div>
    </div>

    <div class="side-panel">
      <div class="side-panel-header">
        <span>📜 Nhật ký nước đi</span>
        <span style="font-size:0.8rem; color:var(--text-muted)" id="total-turns-badge">0 nước</span>
      </div>
      <div id="winner-box"></div>
      <div class="move-list" id="move-list">
        <div style="color:var(--text-muted); text-align:center; padding:20px; font-size:0.9rem">
          Chọn hai bot và bấm "Bắt đầu đấu" để xem trận đấu.
        </div>
      </div>
    </div>
  </main>

  <script>
    const TILE_SIZE = 46;
    const GAP = 12;
    const STEP = TILE_SIZE + GAP; // 58px

    let matchData = null;
    let currentTurn = 0;
    let maxTurns = 0;
    let isPlaying = false;
    let playInterval = null;
    let playSpeed = 400;

    // Khởi tạo bàn cờ tĩnh 9x9
    function initBoardGrid() {
      const board = document.getElementById("board");
      board.innerHTML = "";
      for (let r = 0; r < 9; r++) {
        for (let c = 0; c < 9; c++) {
          const tile = document.createElement("div");
          tile.className = "tile" + (r === 0 ? " goal-p0" : "") + (r === 8 ? " goal-p1" : "");
          tile.style.top = (r * STEP) + "px";
          tile.style.left = (c * STEP) + "px";
          tile.title = `[${r}, ${c}]`;
          tile.innerText = `${r},${c}`;
          board.appendChild(tile);
        }
      }

      // Tạo 2 quân cờ
      const p0 = document.createElement("div");
      p0.id = "pawn-0";
      p0.className = "pawn pawn-0";
      p0.innerText = "P0";
      board.appendChild(p0);

      const p1 = document.createElement("div");
      p1.id = "pawn-1";
      p1.className = "pawn pawn-1";
      p1.innerText = "P1";
      board.appendChild(p1);

      updatePawnPos(p0, 8, 4);
      updatePawnPos(p1, 0, 4);
    }

    function updatePawnPos(elem, r, c) {
      const top = r * STEP + (TILE_SIZE - 38) / 2;
      const left = c * STEP + (TILE_SIZE - 38) / 2;
      elem.style.top = top + "px";
      elem.style.left = left + "px";
    }

    // Tải danh sách bots
    async function loadBots() {
      try {
        const res = await fetch("/api/bots");
        const bots = await res.json();
        const p0Sel = document.getElementById("p0-select");
        const p1Sel = document.getElementById("p1-select");
        p0Sel.innerHTML = "";
        p1Sel.innerHTML = "";

        bots.forEach(b => {
          const opt0 = document.createElement("option");
          opt0.value = b; opt0.innerText = b;
          p0Sel.appendChild(opt0);

          const opt1 = document.createElement("option");
          opt1.value = b; opt1.innerText = b;
          p1Sel.appendChild(opt1);
        });

        // Đặt mặc định
        if (bots.includes("25127275.py")) p0Sel.value = "25127275.py";
        if (bots.includes("Champion_v3_9f1ee8791ef3d8dc.py")) p1Sel.value = "Champion_v3_9f1ee8791ef3d8dc.py";
        else if (bots.includes("hihi.py")) p1Sel.value = "hihi.py";
        else if (bots.includes("minimax_alpha_beta_pruning.py")) p1Sel.value = "minimax_alpha_beta_pruning.py";
      } catch (err) {
        console.error("Lỗi tải danh sách bot:", err);
      }
    }

    // Chạy trận đấu
    async function startMatch() {
      const btn = document.getElementById("start-btn");
      const p0 = document.getElementById("p0-select").value;
      const p1 = document.getElementById("p1-select").value;

      btn.disabled = true;
      btn.innerHTML = "<span>⏳ Đang chạy ván đấu...</span>";
      pause();

      try {
        const res = await fetch(`/api/match?p1=${encodeURIComponent(p0)}&p2=${encodeURIComponent(p1)}`);
        matchData = await res.json();
        if (matchData.error) {
          alert("Lỗi trận đấu: " + matchData.error);
          return;
        }
        setupMatch();
      } catch (err) {
        alert("Lỗi kết nối máy chủ: " + err);
      } finally {
        btn.disabled = false;
        btn.innerHTML = "<span>⚔️ Bắt đầu đấu</span>";
      }
    }

    function setupMatch() {
      maxTurns = matchData.turns.length;
      document.getElementById("turn-slider").max = maxTurns;
      document.getElementById("p0-name").innerText = matchData.players[0].name;
      document.getElementById("p1-name").innerText = matchData.players[1].name;
      document.getElementById("total-turns-badge").innerText = maxTurns + " nước đi";

      // Winner box
      const winBox = document.getElementById("winner-box");
      if (matchData.winner !== null) {
        winBox.innerHTML = `<div class="winner-banner">🏆 Người thắng: Player ${matchData.winner} (${matchData.players[matchData.winner].name})</div>`;
      } else {
        winBox.innerHTML = `<div class="winner-banner" style="color:#e2e8f0; background:rgba(255,255,255,0.1); border-color:#64748b">🤝 Hòa cờ sau ${maxTurns} lượt</div>`;
      }

      // Render move list
      const list = document.getElementById("move-list");
      list.innerHTML = "";

      // Nước 0 (Initial)
      const initItem = document.createElement("div");
      initItem.className = "move-item active";
      initItem.id = `move-item-0`;
      initItem.innerHTML = `<span class="move-turn">Lượt 0</span><span>Khởi đầu ván đấu</span>`;
      initItem.onclick = () => goToTurn(0);
      list.appendChild(initItem);

      matchData.turns.forEach((t, idx) => {
        const item = document.createElement("div");
        item.className = "move-item";
        item.id = `move-item-${t.turn}`;
        const pTag = `<span class="p-tag ${t.player === 0 ? 'p0-tag' : 'p1-tag'}">P${t.player}</span>`;
        let actDesc = "";
        const a = t.action;
        if (!a || a.type === "pass") {
          actDesc = "Bỏ lượt (Pass)";
        } else if (a.type === "move") {
          actDesc = `Đi tới [${a.to[0]}, ${a.to[1]}]`;
        } else if (a.type === "wall") {
          actDesc = `Đặt tường ${a.orientation} tại [${a.row}, ${a.col}]`;
        }
        item.innerHTML = `<span class="move-turn">#${t.turn}</span>${pTag}<span>${actDesc}</span>`;
        item.onclick = () => goToTurn(t.turn);
        list.appendChild(item);
      });

      goToTurn(0);
      togglePlay(); // Tự động phát
    }

    function renderState(state, actionJustTaken = null) {
      const p0Pos = state.positions[0];
      const p1Pos = state.positions[1];
      const p0 = document.getElementById("pawn-0");
      const p1 = document.getElementById("pawn-1");

      updatePawnPos(p0, p0Pos[0], p0Pos[1]);
      updatePawnPos(p1, p1Pos[0], p1Pos[1]);

      document.getElementById("p0-info").innerText = `Vị trí: [${p0Pos[0]}, ${p0Pos[1]}] | Tường: ${state.walls_remaining[0]}/10`;
      document.getElementById("p1-info").innerText = `Vị trí: [${p1Pos[0]}, ${p1Pos[1]}] | Tường: ${state.walls_remaining[1]}/10`;

      // Xóa tất cả tường cũ
      document.querySelectorAll(".wall").forEach(w => w.remove());
      const board = document.getElementById("board");

      // Tường ngang: H (r, c)
      state.walls.horizontal.forEach(w => {
        const [r, c] = w;
        const wallElem = document.createElement("div");
        wallElem.className = "wall horizontal";
        wallElem.style.top = (r * STEP + TILE_SIZE + 1) + "px";
        wallElem.style.left = (c * STEP) + "px";
        if (actionJustTaken && actionJustTaken.type === "wall" && actionJustTaken.orientation === "H" && actionJustTaken.row === r && actionJustTaken.col === c) {
          wallElem.classList.add("newest");
        }
        board.appendChild(wallElem);
      });

      // Tường dọc: V (r, c)
      state.walls.vertical.forEach(w => {
        const [r, c] = w;
        const wallElem = document.createElement("div");
        wallElem.className = "wall vertical";
        wallElem.style.top = (r * STEP) + "px";
        wallElem.style.left = (c * STEP + TILE_SIZE + 1) + "px";
        if (actionJustTaken && actionJustTaken.type === "wall" && actionJustTaken.orientation === "V" && actionJustTaken.row === r && actionJustTaken.col === c) {
          wallElem.classList.add("newest");
        }
        board.appendChild(wallElem);
      });
    }

    function goToTurn(t) {
      if (!matchData) return;
      currentTurn = Math.max(0, Math.min(t, maxTurns));
      document.getElementById("turn-slider").value = currentTurn;
      document.getElementById("turn-label").innerText = `Lượt ${currentTurn} / ${maxTurns}`;

      let state, act = null;
      if (currentTurn === 0) {
        state = matchData.initial_state;
      } else {
        const turnObj = matchData.turns[currentTurn - 1];
        state = turnObj.state;
        act = turnObj.action;
      }

      renderState(state, act);

      // Highlight move log item
      document.querySelectorAll(".move-item").forEach(it => it.classList.remove("active"));
      const activeItem = document.getElementById(`move-item-${currentTurn}`);
      if (activeItem) {
        activeItem.classList.add("active");
        activeItem.scrollIntoView({ block: "nearest", behavior: "smooth" });
      }
    }

    function stepTurn(delta) {
      pause();
      goToTurn(currentTurn + delta);
    }

    function togglePlay() {
      if (isPlaying) {
        pause();
      } else {
        play();
      }
    }

    function play() {
      if (!matchData) return;
      if (currentTurn >= maxTurns) goToTurn(0);
      isPlaying = true;
      document.getElementById("play-btn").innerText = "⏸ Tạm dừng";
      playInterval = setInterval(() => {
        if (currentTurn < maxTurns) {
          goToTurn(currentTurn + 1);
        } else {
          pause();
        }
      }, playSpeed);
    }

    function pause() {
      isPlaying = false;
      document.getElementById("play-btn").innerText = "▶ Phát";
      if (playInterval) clearInterval(playInterval);
      playInterval = null;
    }

    function updateSpeed() {
      playSpeed = parseInt(document.getElementById("speed-select").value);
      if (isPlaying) {
        pause();
        play();
      }
    }

    // Chạy khi nạp trang
    window.onload = () => {
      initBoardGrid();
      loadBots();
    };
  </script>
</body>
</html>
"""


class ArenaHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            return

        if path == "/api/bots":
            bots = []
            if SUBMISSIONS_DIR.exists():
                for p in sorted(SUBMISSIONS_DIR.glob("*.py")):
                    if p.name not in ("__init__.py", "TEMPLATE.py"):
                        bots.append(p.name)
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(bots).encode("utf-8"))
            return

        if path == "/api/match":
            p1 = query.get("p1", [None])[0]
            p2 = query.get("p2", [None])[0]
            max_turns = query.get("max_turns", ["500"])[0]

            if not p1 or not p2:
                self.send_error(400, "Missing p1 or p2 parameter")
                return

            p1_path = SUBMISSIONS_DIR / p1
            p2_path = SUBMISSIONS_DIR / p2

            if not p1_path.exists() or not p2_path.exists():
                self.send_error(404, "Bot file not found")
                return

            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                temp_log_path = Path(tmp.name)

            try:
                cmd = [
                    sys.executable,
                    str(ROOT_DIR / "arena.py"),
                    "--player1", str(p1_path),
                    "--player2", str(p2_path),
                    "--max-turns", str(max_turns),
                    "--log", str(temp_log_path),
                ]
                subprocess.run(cmd, capture_output=True, text=True, timeout=120)

                if temp_log_path.exists() and temp_log_path.stat().st_size > 0:
                    with open(temp_log_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(json.dumps(data).encode("utf-8"))
                else:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Failed to run match"}).encode("utf-8"))
            except Exception as exc:
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))
            finally:
                if temp_log_path.exists():
                    try:
                        temp_log_path.unlink()
                    except Exception:
                        pass
            return

        self.send_error(404, "Not Found")

    def log_message(self, format, *args):
        # Ẩn bớt log request để terminal sạch sẽ
        pass


def main():
    server = http.server.HTTPServer(("127.0.0.1", PORT), ArenaHandler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"\n=======================================================")
    print(f"  Wrongway Visual Arena đang chạy!")
    print(f"  Truy cập tại: {url}")
    print(f"  (Nhấn Ctrl+C để dừng máy chủ)")
    print(f"=======================================================\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã tắt máy chủ.")
        server.server_close()


if __name__ == "__main__":
    main()

