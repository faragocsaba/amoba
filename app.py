import math
import random
import time
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()

templates = Jinja2Templates(directory="templates")

BOARD_SIZE = 10
WIN_LENGTH = 5
EMPTY = '.'
PLAYER_X = 'X'
PLAYER_O = 'O'

class GameState(BaseModel):
    board: list[list[str]]
    ai_type: str

def get_neighboring_moves(board):
    candidates = set()
    has_stones = False
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != EMPTY:
                has_stones = True
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr][nc] == EMPTY:
                            candidates.add((nr, nc))
    if not has_stones:
        return [(BOARD_SIZE//2, BOARD_SIZE//2)]
    return list(candidates)

class GomokuAI:
    def __init__(self, board):
        self.board = board

    def check_winner(self):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] == EMPTY: continue
                player = self.board[r][c]
                for dr, dc in directions:
                    if self.check_line(r, c, dr, dc, player):
                        return player
        if not any(EMPTY in row for row in self.board): return "DRAW"
        return None

    def check_line(self, r, c, dr, dc, player):
        for i in range(WIN_LENGTH):
            nr, nc = r + dr * i, c + dc * i
            if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and self.board[nr][nc] == player):
                return False
        return True

    def get_opening_move(self):
        stones_cnt = sum(row.count(PLAYER_X) + row.count(PLAYER_O) for row in self.board)
        center = BOARD_SIZE // 2
        if stones_cnt == 0: return center, center
        if stones_cnt == 1:
            # Egyszerűsített nyitás válasz
            if self.board[center][center] == EMPTY: return center, center
            return center - 1, center - 1
        return None

    def get_immediate_move(self, candidates):
        # Trivial win
        for (r, c) in candidates:
            self.board[r][c] = PLAYER_O
            if self.check_winner() == PLAYER_O:
                self.board[r][c] = EMPTY
                return (r, c)
            self.board[r][c] = EMPTY
        # Trivial defense
        for (r, c) in candidates:
            self.board[r][c] = PLAYER_X
            if self.check_winner() == PLAYER_X:
                self.board[r][c] = EMPTY
                return (r, c)
            self.board[r][c] = EMPTY
        return None


# --- MINIMAX AI ---
class GomokuMinimaxAI(GomokuAI):
    def __init__(self, board):
        super().__init__(board)
        self.search_depth = 3

    def evaluate(self, player):
        total_score = 0
        scores = {0: 0, 1: 1, 2: 10, 3: 100, 4: 1000, 5: 100000}

        def evaluate_section(window):
            c_p = window.count(player)
            c_o = window.count(PLAYER_X if player == PLAYER_O else PLAYER_O)
            if c_p > 0 and c_o > 0: return 0
            if c_p == 0 and c_o == 0: return 0
            if c_o == 0: return scores.get(c_p, 0)
            if c_p == 0: return -scores.get(c_o, 0) * 1.2
            return 0

        # 1. horizontal
        for r in range(BOARD_SIZE):
            row = self.board[r]
            for c in range(BOARD_SIZE - WIN_LENGTH + 1):
                total_score += evaluate_section(row[c:c+WIN_LENGTH])
        # 2. vertical
        for c in range(BOARD_SIZE):
            col = [self.board[r][c] for r in range(BOARD_SIZE)]
            for r in range(BOARD_SIZE - WIN_LENGTH + 1):
                total_score += evaluate_section(col[r:r+WIN_LENGTH])
        # 3. diagonal
        for r in range(BOARD_SIZE - WIN_LENGTH + 1):
            for c in range(BOARD_SIZE - WIN_LENGTH + 1):
                win = [self.board[r+i][c+i] for i in range(WIN_LENGTH)]
                total_score += evaluate_section(win)
        # 4. opposite diagonal
        for r in range(BOARD_SIZE - WIN_LENGTH + 1):
            for c in range(BOARD_SIZE - 1, WIN_LENGTH - 2, -1):
                win = [self.board[r+i][c-i] for i in range(WIN_LENGTH)]
                total_score += evaluate_section(win)

        return total_score

    def minimax(self, depth, alpha, beta, is_maximizing):
        winner = self.check_winner()
        if winner or depth == 0:
            if winner == PLAYER_O: return 1000000
            if winner == PLAYER_X: return -1000000
            if winner == "DRAW": return 0
            return self.evaluate(PLAYER_O)

        candidates = get_neighboring_moves(self.board)

        if is_maximizing:
            max_eval = -math.inf
            for (r, c) in candidates:
                self.board[r][c] = PLAYER_O
                eval = self.minimax(depth - 1, alpha, beta, False)
                self.board[r][c] = EMPTY
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha: break
            return max_eval
        else:
            min_eval = math.inf
            for (r, c) in candidates:
                self.board[r][c] = PLAYER_X
                eval = self.minimax(depth - 1, alpha, beta, True)
                self.board[r][c] = EMPTY
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha: break
            return min_eval

    def get_best_move(self):
        # 1. Opening
        opening = self.get_opening_move()
        if opening: return opening

        candidates = get_neighboring_moves(self.board)

        # 2. Immediate win / defense (special logic)
        immediate = self.get_immediate_move(candidates)
        if immediate: return immediate

        # 3. Minimax search
        stones = sum(row.count(PLAYER_X) + row.count(PLAYER_O) for row in self.board)
        self.search_depth = 2 if stones > 15 else 3

        print(f"Minimax gondolkodik (mélység: {self.search_depth})...")
        best_score = -math.inf
        best_move = None
        alpha = -math.inf
        beta = math.inf

        for (r, c) in candidates:
            self.board[r][c] = PLAYER_O
            score = self.minimax(self.search_depth, alpha, beta, False)
            self.board[r][c] = EMPTY

            if score > best_score:
                best_score = score
                best_move = (r, c)
            alpha = max(alpha, score)

        return best_move


# --- MONTE CARLO AI ---
THINK_TIME = 3.0

class MCTSNode:
    def __init__(self, board, parent=None, move=None, player=PLAYER_O):
        self.board = board
        self.parent = parent
        self.move = move
        self.player = player
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = get_neighboring_moves(board)

    def uct_select_child(self):
        s = sorted(self.children, key=lambda c: c.wins / c.visits + math.sqrt(2 * math.log(self.visits) / c.visits))
        return s[-1]

    def add_child(self, move, board_state, player):
        child = MCTSNode(board_state, parent=self, move=move, player=player)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def update(self, result):
        self.visits += 1
        self.wins += result

class GomokuMonteCarloAI(GomokuAI):
    def check_max_line_fast(self, board, r, c, player):
        max_len = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for i in range(1, WIN_LENGTH):
                nr, nc = r + dr*i, c + dc*i
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr][nc] == player: count += 1
                else: break
            for i in range(1, WIN_LENGTH):
                nr, nc = r - dr*i, c - dc*i
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr][nc] == player: count += 1
                else: break
            if count > max_len: max_len = count
        return max_len

    def get_best_move(self):
        # 1. Opening
        opening = self.get_opening_move()
        if opening: return opening

        candidates = get_neighboring_moves(self.board)

        # 2. Immediate win / defense (special logic)
        immediate = self.get_immediate_move(candidates)
        if immediate: return immediate

        # 3. Monte-Carlo search
        root_node = MCTSNode(board=self.board, player=PLAYER_X)
        end_time = time.time() + THINK_TIME
        simulations = 0

        while time.time() < end_time:
            node = root_node
            temp_board = [row[:] for row in node.board]

            # Selection
            while node.untried_moves == [] and node.children != []:
                node = node.uct_select_child()
                temp_board[node.move[0]][node.move[1]] = node.player

            # Expansion
            if node.untried_moves != []:
                m = random.choice(node.untried_moves)
                player_to_move = PLAYER_X if node.player == PLAYER_O else PLAYER_O
                temp_board[m[0]][m[1]] = player_to_move
                node = node.add_child(m, temp_board, player_to_move)

            # Simulation (quick playout)
            current_player = node.player
            winner = None
            sim_moves = 0

            sim_candidates = get_neighboring_moves(temp_board)
            random.shuffle(sim_candidates)

            while sim_moves < len(sim_candidates):
                move = sim_candidates[sim_moves]
                current_player = PLAYER_X if current_player == PLAYER_O else PLAYER_O
                temp_board[move[0]][move[1]] = current_player

                if self.check_max_line_fast(temp_board, move[0], move[1], current_player) >= 5:
                    winner = current_player
                    break
                sim_moves += 1

            if winner is None: winner = "DRAW"

            # Backpropagation
            while node is not None:
                score = 0
                if winner == "DRAW": score = 0.5
                elif winner == node.player: score = 1
                node.update(score)
                node = node.parent
            simulations += 1

        if not root_node.children: return None
        return sorted(root_node.children, key=lambda c: c.visits)[-1].move


# --- ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/move")
async def calculate_move(state: GameState):
    board = state.board
    ai_type = state.ai_type
    
    if ai_type == "minimax":
        ai = GomokuMinimaxAI(board)
    elif ai_type == "montecarlo":
        ai = GomokuMonteCarloAI(board)
    else:
        return {"row": -1, "col": -1, "winner": "ERROR"}
        
    move = ai.get_best_move()
    
    if move:
        ai.board[move[0]][move[1]] = PLAYER_O
        winner = ai.check_winner()
        return {"row": move[0], "col": move[1], "winner": winner}
    
    return {"row": -1, "col": -1, "winner": "DRAW"}
