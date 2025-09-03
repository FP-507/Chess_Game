import json
import os
import collections
import random
import pygame
import sys
import chess
import chess.pgn

# --- Q-learning IA ---
Q_FILE = "ia_qtable.json"
ALPHA = 0.5
GAMMA = 0.9

def load_qtable():
    if os.path.exists(Q_FILE):
        with open(Q_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_qtable(qtable):
    with open(Q_FILE, 'w') as f:
        json.dump(qtable, f)

qtable = load_qtable()

# --- Configuración ---
BOARD_SIZE = 700
SIDEBAR_WIDTH = 420
WIDTH, HEIGHT = BOARD_SIZE + SIDEBAR_WIDTH, BOARD_SIZE
DIMENSION = 8
SQ_SIZE = BOARD_SIZE // DIMENSION
FPS = 60
IMAGES = {}

LIGHT_SQUARE = (230, 230, 230)
DARK_SQUARE = (80, 80, 80)
SIDEBAR_COLOR = (36, 37, 46)
TEXT_COLOR = (230, 230, 230)
ACCENT_COLOR = (255, 204, 0)
HIST_BG = (50, 50, 65)
BUTTON_COLOR = (60, 60, 80)
BUTTON_HOVER = (90, 90, 120)
BUTTON_TEXT = (255, 255, 255)
LAST_MOVE_COLOR = (255, 204, 0, 80)

# --- Dificultad ---
difficulty = "hard"  # "easy", "normal", "hard"
epsilon_map = {"easy": 0.2, "normal": 0.01, "hard": 0.0}
search_depth_map = {"easy": 1, "normal": 1, "hard": 2}
epsilon = epsilon_map[difficulty]
search_depth = search_depth_map[difficulty]

def load_images():
    pieces = ['wp', 'wR', 'wN', 'wB', 'wQ', 'wK', 'bp', 'bR', 'bN', 'bB', 'bQ', 'bK']
    for piece in pieces:
        path = os.path.join('assets', f'{piece}.png')
        if os.path.exists(path):
            img = pygame.image.load(path).convert_alpha()
            IMAGES[piece] = pygame.transform.smoothscale(img, (SQ_SIZE, SQ_SIZE))

def draw_board(screen):
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = LIGHT_SQUARE if (r + c) % 2 == 0 else DARK_SQUARE
            pygame.draw.rect(screen, color, (c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))

def draw_last_move(screen, move):
    if move is None:
        return
    from_sq = move.from_square
    to_sq = move.to_square
    for sq in [from_sq, to_sq]:
        row = 7 - chess.square_rank(sq)
        col = chess.square_file(sq)
        s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
        s.fill(LAST_MOVE_COLOR)
        screen.blit(s, (col*SQ_SIZE, row*SQ_SIZE))

def draw_pieces(screen, board):
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            color = 'w' if piece.color == chess.WHITE else 'b'
            if piece.piece_type == chess.PAWN:
                piece_str = color + 'p'
            else:
                piece_str = color + piece.symbol().upper()
            if piece_str in IMAGES:
                screen.blit(IMAGES[piece_str], (col*SQ_SIZE, row*SQ_SIZE))

def draw_sidebar(screen, board, move_log, mouse_pos, reset_hover, ia_color, player_color):
    pygame.draw.rect(screen, SIDEBAR_COLOR, (BOARD_SIZE, 0, SIDEBAR_WIDTH, HEIGHT))
    font_title = pygame.font.SysFont('Arial', 30, bold=True)
    title_text = font_title.render("♟ AJEDREZ PREMIUM", True, ACCENT_COLOR)
    screen.blit(title_text, (BOARD_SIZE + 30, 20))
    pygame.draw.line(screen, ACCENT_COLOR, (BOARD_SIZE + 20, 60), (WIDTH - 30, 60), 3)
    font = pygame.font.SysFont('Arial', 22, bold=True)
    turn = 'Blancas' if board.turn == chess.WHITE else 'Negras'
    turn_text = font.render(f'Turno: {turn}', True, TEXT_COLOR)
    screen.blit(turn_text, (BOARD_SIZE + 30, 70))
    starter = 'IA' if (ia_color == chess.WHITE and board.fullmove_number == 1 and board.turn == chess.WHITE) or (ia_color == chess.BLACK and board.fullmove_number == 1 and board.turn == chess.BLACK) else 'Jugador'
    font_starter = pygame.font.SysFont('Arial', 18, bold=True)
    starter_color = ACCENT_COLOR if starter == 'IA' else (180,255,180)
    screen.blit(font_starter.render(f'Inicia: {starter}', True, starter_color), (BOARD_SIZE + 30, 100))
    font_role = pygame.font.SysFont('Arial', 18)
    ia_str = 'Blancas' if ia_color == chess.WHITE else 'Negras'
    player_str = 'Blancas' if player_color == chess.WHITE else 'Negras'
    screen.blit(font_role.render(f'IA: {ia_str}', True, ACCENT_COLOR), (BOARD_SIZE + 30, 130))
    screen.blit(font_role.render(f'Jugador: {player_str}', True, (180,255,180)), (BOARD_SIZE + 140, 130))
    button_rect = pygame.Rect(BOARD_SIZE + 30, 160, 200, 40)
    color = BUTTON_HOVER if reset_hover else BUTTON_COLOR
    pygame.draw.rect(screen, color, button_rect, border_radius=10)
    font_btn = pygame.font.SysFont('Arial', 20, bold=True)
    btn_text = font_btn.render('Rendirse', True, BUTTON_TEXT)
    screen.blit(btn_text, (button_rect.x + 60, button_rect.y + 8))
    font2 = pygame.font.SysFont('Consolas', 17)
    pygame.draw.rect(screen, HIST_BG, (BOARD_SIZE + 30, 220, SIDEBAR_WIDTH - 60, HEIGHT - 240), border_radius=10)
    hist_title = font2.render('Historial:', True, ACCENT_COLOR)
    screen.blit(hist_title, (BOARD_SIZE + 40, 230))
    for i, move in enumerate(move_log[-18:]):
        move_text = font2.render(move, True, TEXT_COLOR)
        screen.blit(move_text, (BOARD_SIZE + 45, 260 + i*22))
    font_stats = pygame.font.SysFont('Arial', 16)
    total = stats['games']
    ia_wins = stats['ia_wins']
    player_wins = stats['player_wins']
    draws = stats['draws']
    if total > 0:
        win_pct = int(100 * ia_wins / total)
        lose_pct = int(100 * player_wins / total)
        draw_pct = int(100 * draws / total)
        stat_text = f"Partidas: {total}  IA: {win_pct}%  Jugador: {lose_pct}%  Tablas: {draw_pct}%"
    else:
        stat_text = "Partidas: 0  IA: 0%  Jugador: 0%  Tablas: 0%"
    screen.blit(font_stats.render(stat_text, True, ACCENT_COLOR), (BOARD_SIZE + 35, HEIGHT - 35))
    return button_rect

def draw_promotion_choices(screen, color, pos):
    pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
    labels = ['Dama', 'Torre', 'Alfil', 'Caballo']
    icons = ['Q', 'R', 'B', 'N']
    buttons = []
    x, y = pos
    for i, piece in enumerate(pieces):
        rect = pygame.Rect(x + i*80, y, 70, 70)
        pygame.draw.rect(screen, BUTTON_COLOR, rect, border_radius=8)
        font = pygame.font.SysFont('Arial', 28, bold=True)
        text = font.render(icons[i], True, ACCENT_COLOR)
        screen.blit(text, (rect.x + 18, rect.y + 15))
        font2 = pygame.font.SysFont('Arial', 14)
        label = font2.render(labels[i], True, BUTTON_TEXT)
        screen.blit(label, (rect.x + 5, rect.y + 50))
        buttons.append((rect, piece))
    return buttons

def evaluate_board(board):
    values = {chess.PAWN: 1, chess.KNIGHT: 3.2, chess.BISHOP: 3.3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    eval = 0

    # Material
    for piece_type in values:
        eval += len(board.pieces(piece_type, chess.WHITE)) * values[piece_type]
        eval -= len(board.pieces(piece_type, chess.BLACK)) * values[piece_type]

    # Penalización por perder dama o torre
    if len(board.pieces(chess.QUEEN, chess.WHITE)) < 1:
        eval -= 4
    if len(board.pieces(chess.QUEEN, chess.BLACK)) < 1:
        eval += 4
    if len(board.pieces(chess.ROOK, chess.WHITE)) < 2:
        eval -= 2
    if len(board.pieces(chess.ROOK, chess.BLACK)) < 2:
        eval += 2

    # Bonificación por pareja de alfiles
    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        eval += 0.5
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        eval -= 0.5

    # Penalización por piezas indefensas (no defendidas por otra pieza propia)
    for color in [chess.WHITE, chess.BLACK]:
        for sq in board.pieces(chess.QUEEN, color) | board.pieces(chess.ROOK, color) | board.pieces(chess.BISHOP, color) | board.pieces(chess.KNIGHT, color):
            defenders = [att for att in board.attackers(color, sq) if board.piece_at(att) and board.piece_at(att).color == color]
            if not defenders:
                if color == chess.WHITE:
                    eval -= 0.7
                else:
                    eval += 0.7

    # Bonificación por movilidad (más jugadas legales es mejor)
    eval += 0.12 * len(list(board.legal_moves))

    # Control del centro (premia piezas en las casillas centrales)
    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    for sq in center_squares:
        piece = board.piece_at(sq)
        if piece:
            if piece.color == chess.WHITE:
                eval += 0.4
            else:
                eval -= 0.4

    # Estructura de peones
    for color in [chess.WHITE, chess.BLACK]:
        pawns = board.pieces(chess.PAWN, color)
        files = [chess.square_file(sq) for sq in pawns]
        # Peones doblados
        for f in set(files):
            if files.count(f) > 1:
                if color == chess.WHITE:
                    eval -= 0.25 * (files.count(f) - 1)
                else:
                    eval += 0.25 * (files.count(f) - 1)
        # Peones aislados
        for sq in pawns:
            file = chess.square_file(sq)
            neighbors = [file - 1, file + 1]
            isolated = True
            for n in neighbors:
                if 0 <= n < 8:
                    if any(chess.square_file(p) == n for p in pawns):
                        isolated = False
            if isolated:
                if color == chess.WHITE:
                    eval -= 0.3
                else:
                    eval += 0.3
        # Peones pasados
        for sq in pawns:
            rank = chess.square_rank(sq)
            file = chess.square_file(sq)
            passed = True
            for r in range(rank+1, 8) if color == chess.WHITE else range(rank-1, -1, -1):
                for f in [file-1, file, file+1]:
                    if 0 <= f < 8:
                        opp_sq = chess.square(f, r)
                        if board.piece_at(opp_sq) and board.piece_at(opp_sq).piece_type == chess.PAWN and board.piece_at(opp_sq).color != color:
                            passed = False
            if passed:
                if color == chess.WHITE:
                    eval += 0.4
                else:
                    eval -= 0.4

    # Penalización por rey expuesto (sin peones cerca)
    for color in [chess.WHITE, chess.BLACK]:
        king_sq = list(board.pieces(chess.KING, color))
        if king_sq:
            king_sq = king_sq[0]
            rank = chess.square_rank(king_sq)
            file = chess.square_file(king_sq)
            protection = 0
            for dr in [-1, 0, 1]:
                for df in [-1, 0, 1]:
                    if dr == 0 and df == 0:
                        continue
                    sq = chess.square(file + df, rank + dr)
                    if 0 <= file + df < 8 and 0 <= rank + dr < 8:
                        if board.piece_at(sq) and board.piece_at(sq).piece_type == chess.PAWN and board.piece_at(sq).color == color:
                            protection += 0.3
            if protection < 0.3:
                if color == chess.WHITE:
                    eval -= 0.7
                else:
                    eval += 0.7

    # Penalización por piezas atrapadas (sin movimientos)
    for color in [chess.WHITE, chess.BLACK]:
        for piece_type in [chess.BISHOP, chess.KNIGHT, chess.ROOK, chess.QUEEN]:
            for sq in board.pieces(piece_type, color):
                moves = [move for move in board.legal_moves if move.from_square == sq]
                if not moves:
                    if color == chess.WHITE:
                        eval -= 0.4
                    else:
                        eval += 0.4

    # Penalización por repetición, tablas, material insuficiente
    if board.is_repetition(2):
        eval -= 1
    if board.can_claim_fifty_moves():
        eval -= 1
    if board.is_insufficient_material():
        eval -= 1

    return eval

def minimax(board, depth, alpha, beta, maximizing, color):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board), None
    legal_moves = list(board.legal_moves)
    best_move = None
    if maximizing:
        max_eval = -float('inf')
        for move in legal_moves:
            board.push(move)
            eval, _ = minimax(board, depth-1, alpha, beta, False, color)
            board.pop()
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in legal_moves:
            board.push(move)
            eval, _ = minimax(board, depth-1, alpha, beta, True, color)
            board.pop()
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move

def q_choose_move(board, color, epsilon=epsilon):
    fen = board.fen()
    legal_moves = list(board.legal_moves)
    # Si dificultad alta, usa minimax con alpha-beta
    if search_depth > 1:
        _, move = minimax(board, search_depth, -float('inf'), float('inf'), color == chess.WHITE, color)
        if move:
            return move
    if random.random() < epsilon:
        return random.choice(legal_moves)
    move_scores = []
    for move in legal_moves:
        q_value = qtable.get(fen + str(move), None)
        if q_value is not None:
            score = q_value
        else:
            board.push(move)
            score = evaluate_board(board)
            board.pop()
        move_scores.append((move, score))
    if color == chess.WHITE:
        best_score = max(score for _, score in move_scores)
        best_moves = [move for move, score in move_scores if score == best_score]
    else:
        best_score = min(score for _, score in move_scores)
        best_moves = [move for move, score in move_scores if score == best_score]
    return random.choice(best_moves)

game_memory = []
stats = collections.defaultdict(int)

def export_pgn(move_log, ia_color, player_color, result):
    game = chess.pgn.Game()
    game.headers["Event"] = "Chess Game"
    game.headers["White"] = "IA" if ia_color == chess.WHITE else "Jugador"
    game.headers["Black"] = "IA" if ia_color == chess.BLACK else "Jugador"
    game.headers["Result"] = result
    node = game
    for san in move_log:
        move = node.board().parse_san(san)
        node = node.add_main_variation(move)
    with open("last_game.pgn", "w") as f:
        print(game, file=f)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Ajedrez (pygame + python-chess)')
    clock = pygame.time.Clock()
    load_images()

    board = chess.Board()
    move_log = []
    dragging = False
    drag_start = None
    drag_piece = None
    drag_pos = (0, 0)
    last_move = None

    ia_color = random.choice([chess.WHITE, chess.BLACK])
    player_color = chess.BLACK if ia_color == chess.WHITE else chess.WHITE

    game_memory = []
    running = True

    promotion_pending = None
    promotion_buttons = []

    ia_move_pending = False
    ia_move_timer = 0
    IA_DELAY = 400

    def reset_game():
        nonlocal board, move_log, dragging, drag_start, drag_piece, drag_pos, ia_color, player_color, promotion_pending, promotion_buttons, ia_move_pending, ia_move_timer, last_move
        board = chess.Board()
        move_log.clear()
        dragging = False
        drag_start = None
        drag_piece = None
        drag_pos = (0, 0)
        ia_color = random.choice([chess.WHITE, chess.BLACK])
        player_color = chess.BLACK if ia_color == chess.WHITE else chess.WHITE
        game_memory.clear()
        promotion_pending = None
        promotion_buttons = []
        ia_move_pending = False
        ia_move_timer = 0
        last_move = None

    def guardar_experiencia(resultado):
        # Guarda la experiencia de la partida actual
        stats['games'] += 1
        if resultado == "1-0":
            stats['player_wins'] += 1
        elif resultado == "0-1":
            stats['ia_wins'] += 1
        else:
            stats['draws'] += 1
        for fen, move, reward in game_memory:
            key = fen + str(move)
            old_q = qtable.get(key, 0)
            qtable[key] = old_q + ALPHA * (reward - old_q)
        save_qtable(qtable)
        game_memory.clear()

    while running:
        mouse_pos = pygame.mouse.get_pos()
        reset_hover = False

        draw_board(screen)
        draw_last_move(screen, last_move)
        if dragging and drag_piece:
            for square in chess.SQUARES:
                if square == drag_start:
                    continue
                piece = board.piece_at(square)
                if piece:
                    row = 7 - chess.square_rank(square)
                    col = chess.square_file(square)
                    color = 'w' if piece.color == chess.WHITE else 'b'
                    if piece.piece_type == chess.PAWN:
                        piece_str = color + 'p'
                    else:
                        piece_str = color + piece.symbol().upper()
                    if piece_str in IMAGES:
                        screen.blit(IMAGES[piece_str], (col*SQ_SIZE, row*SQ_SIZE))
            color = 'w' if drag_piece.color == chess.WHITE else 'b'
            if drag_piece.piece_type == chess.PAWN:
                piece_str = color + 'p'
            else:
                piece_str = color + drag_piece.symbol().upper()
            if piece_str in IMAGES:
                screen.blit(IMAGES[piece_str], (drag_pos[0] - SQ_SIZE//2, drag_pos[1] - SQ_SIZE//2))
        else:
            draw_pieces(screen, board)

        button_rect = draw_sidebar(screen, board, move_log, mouse_pos, reset_hover, ia_color, player_color)

        if promotion_pending:
            promotion_buttons = draw_promotion_choices(screen, promotion_pending[2], (BOARD_SIZE//2 - 160, BOARD_SIZE//2 - 35))
        else:
            promotion_buttons = []

        # Turno de la IA (no bloquea la interfaz)
        if board.turn == ia_color and not promotion_pending and not dragging and not board.is_game_over():
            if not ia_move_pending:
                ia_move_pending = True
                ia_move_timer = pygame.time.get_ticks()
            elif pygame.time.get_ticks() - ia_move_timer > IA_DELAY:
                move = q_choose_move(board, ia_color)
                if move:
                    san = board.san(move)
                    board.push(move)
                    move_log.append(san)
                    last_move = move
                    # Experiencia: mate, tablas, normal
                    if board.is_checkmate():
                        guardar_experiencia("0-1")
                        export_pgn(move_log, ia_color, player_color, "0-1")
                        reset_game()
                    elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_fifty_moves() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
                        guardar_experiencia("1/2-1/2")
                        export_pgn(move_log, ia_color, player_color, "1/2-1/2")
                        reset_game()
                ia_move_pending = False
        else:
            ia_move_pending = False

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    x, y = event.pos
                    if button_rect.collidepoint(x, y):
                        reset_hover = True
                        guardar_experiencia("0-1")
                        export_pgn(move_log, ia_color, player_color, "0-1")
                        reset_game()
                    elif x < BOARD_SIZE and y < BOARD_SIZE and not promotion_pending:
                        col = x // SQ_SIZE
                        row = y // SQ_SIZE
                        square = chess.square(col, 7 - row)
                        piece = board.piece_at(square)
                        if piece and piece.color == board.turn:
                            dragging = True
                            drag_start = square
                            drag_piece = piece
                            drag_pos = event.pos
                    elif promotion_pending and promotion_buttons:
                        for rect, piece_type in promotion_buttons:
                            if rect.collidepoint(x, y):
                                move = chess.Move(promotion_pending[0], promotion_pending[1], promotion=piece_type)
                                if move in board.legal_moves:
                                    san = board.san(move)
                                    board.push(move)
                                    move_log.append(san)
                                    last_move = move
                                    if board.is_checkmate():
                                        guardar_experiencia("1-0")
                                        export_pgn(move_log, ia_color, player_color, "1-0")
                                        reset_game()
                                    elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_fifty_moves() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
                                        guardar_experiencia("1/2-1/2")
                                        export_pgn(move_log, ia_color, player_color, "1/2-1/2")
                                        reset_game()
                                promotion_pending = None
                                break
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and dragging and not promotion_pending:
                    x, y = event.pos
                    if x < BOARD_SIZE and y < BOARD_SIZE:
                        col = x // SQ_SIZE
                        row = y // SQ_SIZE
                        target = chess.square(col, 7 - row)
                        move = chess.Move(drag_start, target)
                        if drag_piece and drag_piece.piece_type == chess.PAWN and (chess.square_rank(target) == 0 or chess.square_rank(target) == 7):
                            promotion_pending = (drag_start, target, drag_piece.color)
                        elif move in board.legal_moves:
                            san = board.san(move)
                            board.push(move)
                            move_log.append(san)
                            last_move = move
                            if board.is_checkmate():
                                guardar_experiencia("1-0")
                                export_pgn(move_log, ia_color, player_color, "1-0")
                                reset_game()
                            elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_fifty_moves() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
                                guardar_experiencia("1/2-1/2")
                                export_pgn(move_log, ia_color, player_color, "1/2-1/2")
                                reset_game()
                    dragging = False
                    drag_start = None
                    drag_piece = None
            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    drag_pos = event.pos

        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
