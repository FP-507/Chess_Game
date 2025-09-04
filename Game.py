import json
import os
import random
import pygame
import sys
import chess
import chess.pgn
import math
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any

# --- Configuración ---
Q_FILE = "ia_qtable.json"
ALPHA = 0.5
GAMMA = 0.9
BOARD_SIZE = 700
SIDEBAR_WIDTH = 420
WIDTH, HEIGHT = BOARD_SIZE + SIDEBAR_WIDTH, BOARD_SIZE
DIMENSION = 8
SQ_SIZE = BOARD_SIZE // DIMENSION
FPS = 60
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
CHECK_COLOR = (255, 0, 0, 100)
HIGHLIGHT_COLOR = (0, 255, 0, 100)
DIFFICULTIES = ["easy", "normal", "hard"]

# Configuración de dificultad
DIFFICULTY_CONFIG = {
    "easy": {"epsilon": 0.2, "search_depth": 1},
    "normal": {"epsilon": 0.01, "search_depth": 1},
    "hard": {"epsilon": 0.0, "search_depth": 2}
}

# Valores de piezas para evaluación
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3.2,
    chess.BISHOP: 3.3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}

# Casillas centrales para evaluación
CENTER_SQUARES = {chess.D4, chess.D5, chess.E4, chess.E5}

class SoundGenerator:
    """Generador de sonidos para el juego de ajedrez"""
    
    @staticmethod
    def generate_sine_wave(frequency, duration, volume=0.5, sample_rate=44100):
        """Generar una onda sinusoidal"""
        samples = np.zeros(int(duration * sample_rate))
        for i in range(len(samples)):
            samples[i] = volume * np.sin(2 * np.pi * frequency * i / sample_rate)
        return np.clip(samples * 32767, -32767, 32767).astype(np.int16)
    
    @staticmethod
    def generate_square_wave(frequency, duration, volume=0.5, sample_rate=44100):
        """Generar una onda cuadrada"""
        samples = np.zeros(int(duration * sample_rate))
        for i in range(len(samples)):
            samples[i] = volume * (1.0 if np.sin(2 * np.pi * frequency * i / sample_rate) > 0 else -1.0)
        return np.clip(samples * 32767, -32767, 32767).astype(np.int16)
    
    @staticmethod
    def generate_move_sound():
        """Generar sonido de movimiento de pieza"""
        return SoundGenerator.generate_sine_wave(300, 0.1, 0.3)
    
    @staticmethod
    def generate_capture_sound():
        """Generar sonido de captura de pieza"""
        # Dos tonos rápidos
        samples1 = SoundGenerator.generate_sine_wave(400, 0.05, 0.4)
        samples2 = SoundGenerator.generate_sine_wave(300, 0.05, 0.4)
        return np.concatenate((samples1, samples2))
    
    @staticmethod
    def generate_check_sound():
        """Generar sonido de jaque"""
        # Tonos ascendentes
        samples1 = SoundGenerator.generate_sine_wave(400, 0.05, 0.4)
        samples2 = SoundGenerator.generate_sine_wave(500, 0.05, 0.4)
        samples3 = SoundGenerator.generate_sine_wave(600, 0.05, 0.4)
        return np.concatenate((samples1, samples2, samples3))
    
    @staticmethod
    def generate_checkmate_sound():
        """Generar sonido de jaque mate"""
        # Tres tonos descendentes
        samples1 = SoundGenerator.generate_sine_wave(600, 0.1, 0.5)
        samples2 = SoundGenerator.generate_sine_wave(500, 0.1, 0.5)
        samples3 = SoundGenerator.generate_sine_wave(400, 0.1, 0.5)
        return np.concatenate((samples1, samples2, samples3))
    
    @staticmethod
    def generate_promote_sound():
        """Generar sonido de promoción"""
        # Tres tonos ascendentes
        samples1 = SoundGenerator.generate_sine_wave(400, 0.05, 0.4)
        samples2 = SoundGenerator.generate_sine_wave(500, 0.05, 0.4)
        samples3 = SoundGenerator.generate_sine_wave(600, 0.05, 0.4)
        samples4 = SoundGenerator.generate_sine_wave(700, 0.1, 0.5)
        return np.concatenate((samples1, samples2, samples3, samples4))
    
    @staticmethod
    def generate_castle_sound():
        """Generar sonido de enroque"""
        # Dos tonos simultáneos
        duration = 0.2
        sample_rate = 44100
        samples = np.zeros(int(duration * sample_rate))
        for i in range(len(samples)):
            samples[i] = 0.3 * (np.sin(2 * np.pi * 300 * i / sample_rate) + 
                               np.sin(2 * np.pi * 400 * i / sample_rate))
        return np.clip(samples * 32767, -32767, 32767).astype(np.int16)
    
    @staticmethod
    def generate_click_sound():
        """Generar sonido de clic"""
        return SoundGenerator.generate_square_wave(200, 0.05, 0.3)

class ChessGame:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()  # Inicializar mixer para sonidos
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Chess Game Premium')
        self.clock = pygame.time.Clock()
        self.images = {}
        self.sounds = {}
        self.qtable = self.load_qtable()
        self.stats = {"games": 0, "ia_wins": 0, "player_wins": 0, "draws": 0}
        self.transposition_table = {}
        self.difficulty = "hard"
        self.epsilon = DIFFICULTY_CONFIG[self.difficulty]["epsilon"]
        self.search_depth = DIFFICULTY_CONFIG[self.difficulty]["search_depth"]
        self.animation_queue = []
        self.current_animation = None
        self.animation_time = 0
        self.game_state = "menu"  # menu, playing, game_over
        
        # Precompute attacked squares for both colors
        self._attacked_by_white = set()
        self._attacked_by_black = set()
        
        self.load_images()
        self.generate_sounds()  # Generar sonidos programáticamente
        
    def generate_sounds(self):
        """Generar todos los sonidos del juego"""
        sound_generator = SoundGenerator()
        
        # Generar sonidos
        move_sound = pygame.mixer.Sound(buffer=sound_generator.generate_move_sound())
        capture_sound = pygame.mixer.Sound(buffer=sound_generator.generate_capture_sound())
        check_sound = pygame.mixer.Sound(buffer=sound_generator.generate_check_sound())
        checkmate_sound = pygame.mixer.Sound(buffer=sound_generator.generate_checkmate_sound())
        promote_sound = pygame.mixer.Sound(buffer=sound_generator.generate_promote_sound())
        castle_sound = pygame.mixer.Sound(buffer=sound_generator.generate_castle_sound())
        click_sound = pygame.mixer.Sound(buffer=sound_generator.generate_click_sound())
        
        # Almacenar sonidos
        self.sounds = {
            "move": move_sound,
            "capture": capture_sound,
            "check": check_sound,
            "checkmate": checkmate_sound,
            "promote": promote_sound,
            "castle": castle_sound,
            "click": click_sound
        }
    
    def play_sound(self, name):
        """Reproducir efecto de sonido"""
        if name in self.sounds:
            self.sounds[name].play()
    
    def load_qtable(self):
        if os.path.exists(Q_FILE):
            try:
                with open(Q_FILE, 'r') as f:
                    return json.load(f)
            except (IOError, json.JSONDecodeError) as e:
                print(f"Error loading Q-table: {e}")
                return {}
        return {}
    
    def save_qtable(self):
        try:
            with open(Q_FILE, 'w') as f:
                json.dump(self.qtable, f)
        except IOError as e:
            print(f"Error saving Q-table: {e}")
    
    def load_images(self):
        pieces = ['wp', 'wR', 'wN', 'wB', 'wQ', 'wK', 'bp', 'bR', 'bN', 'bB', 'bQ', 'bK']
        for piece in pieces:
            path = os.path.join('assets', f'{piece}.png')
            try:
                if os.path.exists(path):
                    img = pygame.image.load(path).convert_alpha()
                    self.images[piece] = pygame.transform.smoothscale(img, (SQ_SIZE, SQ_SIZE))
                else:
                    print(f"Advertencia: Falta la imagen {path}")
                    # Crear una imagen de placeholder
                    surf = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
                    color = (255, 255, 255) if piece.startswith('w') else (0, 0, 0)
                    pygame.draw.circle(surf, color, (SQ_SIZE//2, SQ_SIZE//2), SQ_SIZE//3)
                    font = pygame.font.SysFont('Arial', 30)
                    symbol = piece[1].upper() if piece[1] != 'p' else 'P'
                    text = font.render(symbol, True, (255, 0, 0) if piece.startswith('w') else (0, 0, 255))
                    surf.blit(text, (SQ_SIZE//2 - text.get_width()//2, SQ_SIZE//2 - text.get_height()//2))
                    self.images[piece] = surf
            except Exception as e:
                print(f"Error cargando imagen {path}: {e}")
                # Crear una imagen de fallback
                surf = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
                color = (255, 0, 0) if piece.startswith('w') else (0, 0, 255)
                pygame.draw.rect(surf, color, (0, 0, SQ_SIZE, SQ_SIZE))
                self.images[piece] = surf
    
    def set_difficulty(self, difficulty):
        if difficulty in DIFFICULTY_CONFIG:
            self.difficulty = difficulty
            self.epsilon = DIFFICULTY_CONFIG[difficulty]["epsilon"]
            self.search_depth = DIFFICULTY_CONFIG[difficulty]["search_depth"]
            self.play_sound("click")
    
    def draw_board(self, screen):
        # Dibujar el tablero de ajedrez
        for r in range(DIMENSION):
            for c in range(DIMENSION):
                color = LIGHT_SQUARE if (r + c) % 2 == 0 else DARK_SQUARE
                pygame.draw.rect(screen, color, (c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))
    
    def draw_last_move(self, screen, board, move):
        if move is None:
            return
        
        for sq in [move.from_square, move.to_square]:
            row = 7 - chess.square_rank(sq)
            col = chess.square_file(sq)
            s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
            s.fill(LAST_MOVE_COLOR)
            screen.blit(s, (col * SQ_SIZE, row * SQ_SIZE))
            
        # Resaltar jaque
        if board.is_check():
            king_sq = board.king(board.turn)
            if king_sq is not None:
                row = 7 - chess.square_rank(king_sq)
                col = chess.square_file(king_sq)
                s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
                s.fill(CHECK_COLOR)
                screen.blit(s, (col * SQ_SIZE, row * SQ_SIZE))
    
    def draw_highlights(self, screen, board, selected_square, valid_moves):
        """Resaltar casillas seleccionadas y movimientos válidos"""
        if selected_square is not None:
            # Resaltar casilla seleccionada
            row = 7 - chess.square_rank(selected_square)
            col = chess.square_file(selected_square)
            s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
            s.fill(HIGHLIGHT_COLOR)
            screen.blit(s, (col * SQ_SIZE, row * SQ_SIZE))
            
            # Resaltar movimientos válidos
            for move in valid_moves:
                if move.from_square == selected_square:
                    to_sq = move.to_square
                    row = 7 - chess.square_rank(to_sq)
                    col = chess.square_file(to_sq)
                    s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
                    
                    # Diferente color para capturas
                    if board.piece_at(to_sq):
                        s.fill((255, 0, 0, 100))  # Rojo para capturas
                    else:
                        s.fill((0, 255, 0, 100))  # Verde para movimientos
                    
                    screen.blit(s, (col * SQ_SIZE, row * SQ_SIZE))
    
    def draw_pieces(self, screen, board):
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row = 7 - chess.square_rank(square)
                col = chess.square_file(square)
                color = 'w' if piece.color == chess.WHITE else 'b'
                piece_type = piece.symbol().upper() if piece.piece_type != chess.PAWN else 'p'
                piece_str = color + piece_type
                
                if piece_str in self.images:
                    screen.blit(self.images[piece_str], (col * SQ_SIZE, row * SQ_SIZE))
    
    def draw_sidebar(self, screen, board, move_log, mouse_pos, reset_hover, ia_color, player_color):
        # Dibujar fondo de la barra lateral
        pygame.draw.rect(screen, SIDEBAR_COLOR, (BOARD_SIZE, 0, SIDEBAR_WIDTH, HEIGHT))
        
        # Título
        font_title = pygame.font.SysFont('Arial', 30, bold=True)
        title_text = font_title.render("♟ AJEDREZ PREMIUM", True, ACCENT_COLOR)
        screen.blit(title_text, (BOARD_SIZE + 30, 20))
        pygame.draw.line(screen, ACCENT_COLOR, (BOARD_SIZE + 20, 60), (WIDTH - 30, 60), 3)
        
        # Información del juego
        font = pygame.font.SysFont('Arial', 22, bold=True)
        turn = 'Blancas' if board.turn == chess.WHITE else 'Negras'
        turn_text = font.render(f'Turno: {turn}', True, TEXT_COLOR)
        screen.blit(turn_text, (BOARD_SIZE + 30, 70))
        
        starter = 'IA' if (ia_color == chess.WHITE and board.fullmove_number == 1 and board.turn == chess.WHITE) or \
                         (ia_color == chess.BLACK and board.fullmove_number == 1 and board.turn == chess.BLACK) else 'Jugador'
        font_starter = pygame.font.SysFont('Arial', 18, bold=True)
        starter_color = ACCENT_COLOR if starter == 'IA' else (180, 255, 180)
        screen.blit(font_starter.render(f'Inicia: {starter}', True, starter_color), (BOARD_SIZE + 30, 100))
        
        font_role = pygame.font.SysFont('Arial', 18)
        ia_str = 'Blancas' if ia_color == chess.WHITE else 'Negras'
        player_str = 'Blancas' if player_color == chess.WHITE else 'Negras'
        screen.blit(font_role.render(f'IA: {ia_str}', True, ACCENT_COLOR), (BOARD_SIZE + 30, 130))
        screen.blit(font_role.render(f'Jugador: {player_str}', True, (180, 255, 180)), (BOARD_SIZE + 140, 130))
        
        # Botón de rendición
        button_rect = pygame.Rect(BOARD_SIZE + 30, 160, 200, 40)
        color = BUTTON_HOVER if reset_hover else BUTTON_COLOR
        pygame.draw.rect(screen, color, button_rect, border_radius=10)
        font_btn = pygame.font.SysFont('Arial', 20, bold=True)
        btn_text = font_btn.render('Rendirse', True, BUTTON_TEXT)
        screen.blit(btn_text, (button_rect.x + 60, button_rect.y + 8))
        
        # Selector de dificultad
        font_diff = pygame.font.SysFont('Arial', 16)
        screen.blit(font_diff.render("Dificultad:", True, TEXT_COLOR), (BOARD_SIZE + 30, 210))
        
        diff_buttons = []
        for i, diff in enumerate(DIFFICULTIES):
            diff_rect = pygame.Rect(BOARD_SIZE + 30 + i*120, 235, 110, 30)
            is_selected = (diff == self.difficulty)
            diff_color = ACCENT_COLOR if is_selected else BUTTON_COLOR
            pygame.draw.rect(screen, diff_color, diff_rect, border_radius=5)
            diff_text = font_diff.render(diff.capitalize(), True, BUTTON_TEXT)
            screen.blit(diff_text, (diff_rect.x + 10, diff_rect.y + 5))
            diff_buttons.append((diff_rect, diff))
        
        # Historial de movimientos
        font2 = pygame.font.SysFont('Consolas', 17)
        pygame.draw.rect(screen, HIST_BG, (BOARD_SIZE + 30, 275, SIDEBAR_WIDTH - 60, HEIGHT - 295), border_radius=10)
        hist_title = font2.render('Historial:', True, ACCENT_COLOR)
        screen.blit(hist_title, (BOARD_SIZE + 40, 285))
        
        for i, move in enumerate(move_log[-18:]):
            move_text = font2.render(move, True, TEXT_COLOR)
            screen.blit(move_text, (BOARD_SIZE + 45, 315 + i * 22))
        
        # Estadísticas
        font_stats = pygame.font.SysFont('Arial', 16)
        total = self.stats['games']
        ia_wins = self.stats['ia_wins']
        player_wins = self.stats['player_wins']
        draws = self.stats['draws']
        
        if total > 0:
            win_pct = int(100 * ia_wins / total)
            lose_pct = int(100 * player_wins / total)
            draw_pct = int(100 * draws / total)
            stat_text = f"Partidas: {total}  IA: {win_pct}%  Jugador: {lose_pct}%  Tablas: {draw_pct}%"
        else:
            stat_text = "Partidas: 0  IA: 0%  Jugador: 0%  Tablas: 0%"
        
        screen.blit(font_stats.render(stat_text, True, ACCENT_COLOR), (BOARD_SIZE + 35, HEIGHT - 35))
        
        return button_rect, diff_buttons
    
    def draw_promotion_choices(self, screen, color, pos):
        pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        labels = ['Dama', 'Torre', 'Alfil', 'Caballo']
        icons = ['Q', 'R', 'B', 'N']
        buttons = []
        x, y = pos
        
        # Fondo semitransparente
        s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 150))
        screen.blit(s, (0, 0))
        
        for i, piece in enumerate(pieces):
            rect = pygame.Rect(x + i * 80, y, 70, 70)
            pygame.draw.rect(screen, BUTTON_COLOR, rect, border_radius=8)
            pygame.draw.rect(screen, ACCENT_COLOR, rect, 2, border_radius=8)
            font = pygame.font.SysFont('Arial', 28, bold=True)
            text = font.render(icons[i], True, ACCENT_COLOR)
            screen.blit(text, (rect.x + 18, rect.y + 15))
            font2 = pygame.font.SysFont('Arial', 14)
            label = font2.render(labels[i], True, BUTTON_TEXT)
            screen.blit(label, (rect.x + 5, rect.y + 50))
            buttons.append((rect, piece))
        
        return buttons
    
    def update_attacked_squares(self, board):
        """Precompute attacked squares for both colors"""
        self._attacked_by_white.clear()
        self._attacked_by_black.clear()
        
        for square in chess.SQUARES:
            if board.is_attacked_by(chess.WHITE, square):
                self._attacked_by_white.add(square)
            if board.is_attacked_by(chess.BLACK, square):
                self._attacked_by_black.add(square)
    
    def evaluate_board(self, board):
        """Evaluate the board position with optimized calculations"""
        # Material balance
        eval = 0
        for piece_type, value in PIECE_VALUES.items():
            eval += len(board.pieces(piece_type, chess.WHITE)) * value
            eval -= len(board.pieces(piece_type, chess.BLACK)) * value
        
        # Update attacked squares
        self.update_attacked_squares(board)
        
        # Piece-specific evaluations
        for color in [chess.WHITE, chess.BLACK]:
            sign = 1 if color == chess.WHITE else -1
            
            # Queen presence
            if len(board.pieces(chess.QUEEN, color)) < 1:
                eval -= sign * 4
            
            # Rook count
            if len(board.pieces(chess.ROOK, color)) < 2:
                eval -= sign * 2
            
            # Bishop pair bonus
            if len(board.pieces(chess.BISHOP, color)) >= 2:
                eval += sign * 0.5
            
            # Piece mobility and safety
            for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                for sq in board.pieces(piece_type, color):
                    # Check if piece is undefended
                    defenders = self._attacked_by_white if color == chess.WHITE else self._attacked_by_black
                    if sq not in defenders:
                        eval -= sign * 0.7
            
            # Center control
            for sq in CENTER_SQUARES:
                piece = board.piece_at(sq)
                if piece and piece.color == color:
                    eval += sign * 0.4
            
            # Pawn structure evaluation
            pawns = board.pieces(chess.PAWN, color)
            files = [chess.square_file(sq) for sq in pawns]
            
            # Doubled pawns penalty
            for f in set(files):
                count = files.count(f)
                if count > 1:
                    eval -= sign * 0.25 * (count - 1)
            
            # Isolated pawns penalty
            for sq in pawns:
                file = chess.square_file(sq)
                neighbors = [file - 1, file + 1]
                isolated = True
                
                for n in neighbors:
                    if 0 <= n < 8 and any(chess.square_file(p) == n for p in pawns):
                        isolated = False
                        break
                
                if isolated:
                    eval -= sign * 0.3
            
            # Passed pawns bonus
            for sq in pawns:
                rank = chess.square_rank(sq)
                file = chess.square_file(sq)
                passed = True
                
                direction = 1 if color == chess.WHITE else -1
                for r in range(rank + direction, 8 if color == chess.WHITE else -1, direction):
                    for f in [file - 1, file, file + 1]:
                        if 0 <= f < 8:
                            opp_sq = chess.square(f, r)
                            opp_pawn = board.piece_at(opp_sq)
                            if opp_pawn and opp_pawn.piece_type == chess.PAWN and opp_pawn.color != color:
                                passed = False
                                break
                    if not passed:
                        break
                
                if passed:
                    eval += sign * 0.4
            
            # King safety
            king_sq = board.king(color)
            if king_sq is not None:
                rank = chess.square_rank(king_sq)
                file = chess.square_file(king_sq)
                protection = 0
                
                for dr in [-1, 0, 1]:
                    for df in [-1, 0, 1]:
                        if dr == 0 and df == 0:
                            continue
                        
                        r, f = rank + dr, file + df
                        if 0 <= r < 8 and 0 <= f < 8:
                            sq = chess.square(f, r)
                            piece = board.piece_at(sq)
                            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                                protection += 0.3
                
                if protection < 0.3:
                    eval -= sign * 0.7
            
            # Piece mobility
            for piece_type in [chess.BISHOP, chess.KNIGHT, chess.ROOK, chess.QUEEN]:
                for sq in board.pieces(piece_type, color):
                    if not any(move for move in board.legal_moves if move.from_square == sq):
                        eval -= sign * 0.4
        
        # Game state evaluations
        if board.is_repetition(2):
            eval -= 1
        
        if board.can_claim_fifty_moves():
            eval -= 1
        
        if board.is_insufficient_material():
            eval -= 1
        
        # Mobility bonus
        eval += 0.12 * board.legal_moves.count()
        
        return eval
    
    def minimax(self, board, depth, alpha, beta, maximizing, color, depth_initial):
        """Optimized minimax with alpha-beta pruning and transposition table"""
        key = board.fen()
        
        # Check transposition table
        if key in self.transposition_table:
            entry_depth, entry_eval, entry_move = self.transposition_table[key]
            if entry_depth >= depth:
                return entry_eval, entry_move
        
        # Terminal conditions
        if depth == 0 or board.is_game_over():
            eval = self.evaluate_board(board)
            self.transposition_table[key] = (depth_initial - depth, eval, None)
            return eval, None
        
        legal_moves = list(board.legal_moves)
        best_move = None
        
        # Sort moves for better pruning (captures first)
        legal_moves.sort(key=lambda move: (
            board.is_capture(move),
            board.gives_check(move),
            move.promotion is not None
        ), reverse=True)
        
        if maximizing:
            max_eval = -float('inf')
            for move in legal_moves:
                board.push(move)
                eval, _ = self.minimax(board, depth - 1, alpha, beta, False, color, depth_initial)
                board.pop()
                
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            
            self.transposition_table[key] = (depth_initial - depth, max_eval, best_move)
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in legal_moves:
                board.push(move)
                eval, _ = self.minimax(board, depth - 1, alpha, beta, True, color, depth_initial)
                board.pop()
                
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            
            self.transposition_table[key] = (depth_initial - depth, min_eval, best_move)
            return min_eval, best_move
    
    def q_choose_move(self, board, color):
        """Choose a move using Q-learning and minimax"""
        fen = board.fen()
        legal_moves = list(board.legal_moves)
        
        # Opening book from Q-table
        if board.fullmove_number == 1 and board.turn == color:
            opening_keys = [key for key in self.qtable.keys() if key.startswith(fen)]
            opening_moves = []
            
            for key in opening_keys:
                move_str = key[len(fen):].strip()
                if 4 <= len(move_str) <= 5:
                    try:
                        move = chess.Move.from_uci(move_str)
                        if move in legal_moves:
                            opening_moves.append(move)
                    except Exception:
                        continue
            
            if opening_moves:
                return random.choice(opening_moves)
        
        # Use minimax for deeper search
        if self.search_depth > 1:
            # Clear transposition table for new search
            self.transposition_table.clear()
            _, move = self.minimax(board, self.search_depth, -float('inf'), float('inf'), 
                                  color == chess.WHITE, color, self.search_depth)
            if move:
                return move
        
        # Exploration with epsilon probability
        if random.random() < self.epsilon:
            return random.choice(legal_moves)
        
        # Exploitation: choose best move from Q-table or evaluation
        move_scores = []
        for move in legal_moves:
            q_value = self.qtable.get(fen + str(move), None)
            if q_value is not None:
                score = q_value
            else:
                board.push(move)
                score = self.evaluate_board(board)
                board.pop()
            move_scores.append((move, score))
        
        if color == chess.WHITE:
            best_score = max(score for _, score in move_scores)
            best_moves = [move for move, score in move_scores if score == best_score]
        else:
            best_score = min(score for _, score in move_scores)
            best_moves = [move for move, score in move_scores if score == best_score]
        
        return random.choice(best_moves) if best_moves else random.choice(legal_moves)
    
    def export_pgn(self, move_log, ia_color, player_color, result):
        """Export game to PGN file"""
        game = chess.pgn.Game()
        game.headers["Event"] = "Chess Game"
        game.headers["White"] = "IA" if ia_color == chess.WHITE else "Jugador"
        game.headers["Black"] = "IA" if ia_color == chess.BLACK else "Jugador"
        game.headers["Result"] = result
        
        node = game
        for san in move_log:
            try:
                move = node.board().parse_san(san)
                node = node.add_main_variation(move)
            except Exception as e:
                print(f"Error parsing SAN move {san}: {e}")
                continue
        
        try:
            with open("last_game.pgn", "w") as f:
                print(game, file=f)
        except IOError as e:
            print(f"Error saving PGN: {e}")
    
    def add_animation(self, move, board, piece_str):
        """Add an animation to the queue"""
        start_row = 7 - chess.square_rank(move.from_square)
        start_col = chess.square_file(move.from_square)
        end_row = 7 - chess.square_rank(move.to_square)
        end_col = chess.square_file(move.to_square)
        
        start_pos = (start_col * SQ_SIZE, start_row * SQ_SIZE)
        end_pos = (end_col * SQ_SIZE, end_row * SQ_SIZE)
        
        # Determine animation type
        is_capture = board.is_capture(move)
        is_castle = board.is_castling(move)
        is_promotion = move.promotion is not None
        
        self.animation_queue.append({
            'type': 'move',
            'piece_str': piece_str,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'duration': 0.3 if is_capture or is_castle or is_promotion else 0.2,
            'is_capture': is_capture,
            'is_castle': is_castle,
            'is_promotion': is_promotion,
            'captured_piece': board.piece_at(move.to_square) if is_capture else None
        })
    
    def update_animations(self, dt):
        """Update all active animations"""
        if self.current_animation is None and self.animation_queue:
            self.current_animation = self.animation_queue.pop(0)
            self.animation_time = 0
            
            # Play appropriate sound
            if self.current_animation['is_capture']:
                self.play_sound("capture")
            elif self.current_animation['is_castle']:
                self.play_sound("castle")
            elif self.current_animation['is_promotion']:
                self.play_sound("promote")
            else:
                self.play_sound("move")
        
        if self.current_animation:
            self.animation_time += dt
            progress = min(1.0, self.animation_time / self.current_animation['duration'])
            
            # Easing function for smoother animation
            progress = 1 - (1 - progress) * (1 - progress)  # Ease out quadratic
            
            if progress >= 1.0:
                self.current_animation = None
                return None
            
            return progress
        
        return None
    
    def draw_animation(self, screen, board, progress):
        """Draw the current animation"""
        if self.current_animation is None:
            return
        
        anim = self.current_animation
        x = anim['start_pos'][0] + (anim['end_pos'][0] - anim['start_pos'][0]) * progress
        y = anim['start_pos'][1] + (anim['end_pos'][1] - anim['start_pos'][1]) * progress
        
        # Draw the board and pieces (excluding the moving piece)
        self.draw_board(screen)
        
        # Draw all pieces except the one being animated
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row = 7 - chess.square_rank(square)
                col = chess.square_file(square)
                color = 'w' if piece.color == chess.WHITE else 'b'
                piece_type = piece.symbol().upper() if piece.piece_type != chess.PAWN else 'p'
                piece_str = color + piece_type
                
                # Don't draw the piece that's being animated
                if piece_str == anim['piece_str'] and square == chess.square(
                    chess.square_file(anim['start_pos'][0] // SQ_SIZE),
                    7 - (anim['start_pos'][1] // SQ_SIZE)
                ):
                    continue
                
                if piece_str in self.images:
                    screen.blit(self.images[piece_str], (col * SQ_SIZE, row * SQ_SIZE))
        
        # Draw the moving piece
        if anim['piece_str'] in self.images:
            # Add a slight bounce effect at the end
            if progress > 0.8 and anim['is_capture']:
                bounce = math.sin((progress - 0.8) * 10 * math.pi) * 5
                screen.blit(self.images[anim['piece_str']], (x, y - bounce))
            else:
                screen.blit(self.images[anim['piece_str']], (x, y))
    
    def guardar_experiencia_jugada(self, board, move, is_ia):
        """Save move experience to Q-table"""
        fen = board.fen()
        reward = 0
        
        # Only calculate rewards for pseudo-legal moves
        if move in board.pseudo_legal_moves:
            if board.is_capture(move):
                reward += 2 if is_ia else 1.5
            if board.gives_check(move):
                reward += 1 if is_ia else 0.7
            if move.promotion:
                reward += 2 if is_ia else 1.5
            
            board.push(move)
            if board.is_check():
                reward -= 2 if is_ia else 1.5
            if board.is_checkmate():
                reward += 10 if is_ia else 8
            if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_fifty_moves():
                reward -= 2
            board.pop()
        
        key = fen + str(move)
        old_q = self.qtable.get(key, 0)
        if reward != 0:
            self.qtable[key] = old_q + ALPHA * (reward - old_q)
    
    def guardar_experiencia(self, resultado):
        """Update statistics and save Q-table"""
        self.stats['games'] += 1
        if resultado == "1-0":
            self.stats['player_wins'] += 1
        elif resultado == "0-1":
            self.stats['ia_wins'] += 1
        else:
            self.stats['draws'] += 1
        
        self.save_qtable()
    
    def entrenamiento(self, partidas=50, visual=False):
        """Train AI with optional visualization"""
        if visual:
            screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption('Entrenamiento IA vs IA')
            clock = pygame.time.Clock()
        else:
            print(f"Entrenando IA vs IA ({partidas} partidas)...")
        
        for i in range(partidas):
            board = chess.Board()
            move_log = []
            last_move = None
            
            while not board.is_game_over():
                if visual:
                    # Draw the board
                    mouse_pos = pygame.mouse.get_pos()
                    self.draw_board(screen)
                    self.draw_last_move(screen, board, last_move)
                    self.draw_pieces(screen, board)
                    self.draw_sidebar(screen, board, move_log, mouse_pos, False, chess.WHITE, chess.BLACK)
                    pygame.display.flip()
                    
                    # Handle events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return
                
                # AI makes a move
                move = self.q_choose_move(board, board.turn)
                if move:
                    san = board.san(move)
                    board.push(move)
                    self.guardar_experiencia_jugada(board, move, is_ia=True)
                    move_log.append(san)
                    last_move = move
                    
                    if visual:
                        piece = board.piece_at(move.to_square)
                        color = 'w' if piece.color == chess.WHITE else 'b'
                        piece_type = piece.symbol().upper() if piece.piece_type != chess.PAWN else 'p'
                        piece_str = color + piece_type
                        
                        self.add_animation(move, board, piece_str)
                        
                        # Animate
                        animating = True
                        start_time = time.time()
                        while animating:
                            dt = clock.tick(FPS) / 1000.0
                            progress = self.update_animations(dt)
                            
                            if progress is not None:
                                self.draw_animation(screen, board, progress)
                                pygame.display.flip()
                            else:
                                animating = False
                            
                            for event in pygame.event.get():
                                if event.type == pygame.QUIT:
                                    pygame.quit()
                                    return
                        
                        pygame.time.delay(200)  # Delay between moves
            
            # Game over
            result = board.result()
            self.guardar_experiencia(result)
            
            if visual:
                self.export_pgn(move_log, chess.WHITE, chess.BLACK, result)
            
            if not visual and (i + 1) % 10 == 0:
                print(f"Partidas entrenadas: {i + 1}/{partidas}")
        
        if visual:
            pygame.time.wait(1200)
        else:
            print("Entrenamiento terminado.")
    
    def draw_main_menu(self, screen, selected_idx, options):
        """Draw the main menu"""
        # Fondo con efecto de degradado
        for y in range(HEIGHT):
            color_val = 30 + (y / HEIGHT) * 10
            pygame.draw.line(screen, (color_val, color_val + 2, color_val + 10), (0, y), (WIDTH, y))
        
        # Título con efecto de sombra
        font_title = pygame.font.SysFont('Arial', 48, bold=True)
        font_btn = pygame.font.SysFont('Arial', 32, bold=True)
        
        # Sombra del título
        title = font_title.render("♟ Chess Game Premium", True, (20, 20, 30))
        screen.blit(title, (WIDTH//2 - title.get_width()//2 + 3, 83))
        
        # Título principal
        title = font_title.render("♟ Chess Game Premium", True, ACCENT_COLOR)
        screen.blit(title, (WIDTH//2 - title.get_width()//2, 80))
        
        # Opciones del menú
        for idx, text in enumerate(options):
            btn_rect = pygame.Rect(WIDTH//2 - 180, 200 + idx*90, 360, 70)
            
            # Efecto de hover
            mouse_pos = pygame.mouse.get_pos()
            is_hover = btn_rect.collidepoint(mouse_pos)
            
            color = ACCENT_COLOR if idx == selected_idx or is_hover else BUTTON_COLOR
            pygame.draw.rect(screen, color, btn_rect, border_radius=18)
            
            # Efecto de resaltado para la opción seleccionada
            if idx == selected_idx:
                pygame.draw.rect(screen, (255, 255, 255, 50), btn_rect, 3, border_radius=18)
            
            btn_text = font_btn.render(text, True, BUTTON_TEXT if idx != selected_idx else (30, 32, 40))
            screen.blit(btn_text, (btn_rect.x + (btn_rect.width - btn_text.get_width()) // 2, btn_rect.y + 18))
        
        # Pie de página
        font_footer = pygame.font.SysFont('Arial', 18)
        footer_text = font_footer.render("by GitHub Copilot", True, (120, 120, 120))
        screen.blit(footer_text, (WIDTH - footer_text.get_width() - 20, HEIGHT - 40))
        
        # Efecto de partículas en el fondo (opcional)
        for i in range(20):
            x = random.randint(0, WIDTH)
            y = random.randint(0, HEIGHT)
            size = random.randint(1, 3)
            pygame.draw.circle(screen, (200, 200, 255, 100), (x, y), size)
    
    def show_graphics_placeholder(self, screen):
        """Placeholder for graphics screen"""
        # Fondo con efecto de fade
        for alpha in range(0, 255, 5):
            s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            s.fill((30, 32, 40, alpha))
            screen.blit(s, (0, 0))
            pygame.display.flip()
            pygame.time.delay(10)
        
        font = pygame.font.SysFont('Arial', 36, bold=True)
        text = font.render("Gráficas - En Desarrollo", True, ACCENT_COLOR)
        
        # Efecto de aparición del texto
        for alpha in range(0, 255, 15):
            s = pygame.Surface((text.get_width() + 20, text.get_height() + 20), pygame.SRCALPHA)
            s.fill((30, 32, 40, alpha))
            screen.blit(s, (WIDTH//2 - s.get_width()//2, HEIGHT//2 - s.get_height()//2))
            
            text_surf = font.render("Gráficas - En Desarrollo", True, ACCENT_COLOR)
            text_surf.set_alpha(alpha)
            screen.blit(text_surf, (WIDTH//2 - text_surf.get_width()//2, HEIGHT//2 - text_surf.get_height()//2))
            
            pygame.display.flip()
            pygame.time.delay(30)
        
        pygame.time.delay(1500)
        
        # Efecto de desaparición
        for alpha in range(255, 0, -15):
            s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            s.fill((30, 32, 40, 255 - alpha))
            screen.blit(s, (0, 0))
            
            text_surf = font.render("Gráficas - En Desarrollo", True, ACCENT_COLOR)
            text_surf.set_alpha(alpha)
            screen.blit(text_surf, (WIDTH//2 - text_surf.get_width()//2, HEIGHT//2 - text_surf.get_height()//2))
            
            pygame.display.flip()
            pygame.time.delay(30)
    
    def draw_game_over(self, screen, result):
        """Draw game over screen"""
        # Fondo semitransparente
        s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        screen.blit(s, (0, 0))
        
        # Caja de resultado
        result_rect = pygame.Rect(WIDTH//2 - 200, HEIGHT//2 - 100, 400, 200)
        pygame.draw.rect(screen, SIDEBAR_COLOR, result_rect, border_radius=20)
        pygame.draw.rect(screen, ACCENT_COLOR, result_rect, 3, border_radius=20)
        
        # Texto de resultado
        font = pygame.font.SysFont('Arial', 36, bold=True)
        if result == "1-0":
            text = font.render("¡Has ganado!", True, (180, 255, 180))
        elif result == "0-1":
            text = font.render("La IA ha ganado", True, ACCENT_COLOR)
        else:
            text = font.render("Empate", True, (200, 200, 200))
        
        screen.blit(text, (WIDTH//2 - text.get_width()//2, HEIGHT//2 - 70))
        
        # Botón de continuar
        font_btn = pygame.font.SysFont('Arial', 24)
        btn_rect = pygame.Rect(WIDTH//2 - 100, HEIGHT//2 + 20, 200, 50)
        pygame.draw.rect(screen, BUTTON_COLOR, btn_rect, border_radius=10)
        pygame.draw.rect(screen, ACCENT_COLOR, btn_rect, 2, border_radius=10)
        
        btn_text = font_btn.render("Continuar", True, BUTTON_TEXT)
        screen.blit(btn_text, (btn_rect.x + (btn_rect.width - btn_text.get_width()) // 2, 
                              btn_rect.y + (btn_rect.height - btn_text.get_height()) // 2))
        
        return btn_rect
    
    def main_menu(self):
        """Main menu loop"""
        options = ["Jugar contra la IA", "Entrenar IA", "Ver gráficas", "Salir"]
        selected_idx = 0
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_DOWN, pygame.K_s]:
                        selected_idx = (selected_idx + 1) % len(options)
                        self.play_sound("click")
                    elif event.key in [pygame.K_UP, pygame.K_w]:
                        selected_idx = (selected_idx - 1) % len(options)
                        self.play_sound("click")
                    elif event.key in [pygame.K_RETURN, pygame.K_SPACE]:
                        self.play_sound("click")
                        if options[selected_idx] == "Jugar contra la IA":
                            self.game_state = "playing"
                            self.main()
                            self.game_state = "menu"
                        elif options[selected_idx] == "Entrenar IA":
                            self.entrenamiento(partidas=1, visual=True)
                        elif options[selected_idx] == "Ver gráficas":
                            self.show_graphics_placeholder(self.screen)
                        elif options[selected_idx] == "Salir":
                            running = False
                            break
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.play_sound("click")
                        mx, my = event.pos
                        for idx, text in enumerate(options):
                            btn_rect = pygame.Rect(WIDTH//2 - 180, 200 + idx*90, 360, 70)
                            if btn_rect.collidepoint(mx, my):
                                selected_idx = idx
                                if text == "Jugar contra la IA":
                                    self.game_state = "playing"
                                    self.main()
                                    self.game_state = "menu"
                                elif text == "Entrenar IA":
                                    self.entrenamiento(partidas=1, visual=True)
                                elif text == "Ver gráficas":
                                    self.show_graphics_placeholder(self.screen)
                                elif text == "Salir":
                                    running = False
                                    break
            
            if not running:
                break
            
            # Solo dibujar si el display está inicializado
            if pygame.display.get_init():
                try:
                    self.draw_main_menu(self.screen, selected_idx, options)
                    pygame.display.update()
                except pygame.error:
                    running = False
                    break
            
            self.clock.tick(FPS)
        
        pygame.quit()
        sys.exit()
    
    def main(self):
        """Main game loop"""
        board = chess.Board()
        move_log = []
        dragging = False
        drag_start = None
        drag_piece = None
        drag_pos = (0, 0)
        selected_square = None
        valid_moves = []
        last_move = None
        ia_color = random.choice([chess.WHITE, chess.BLACK])
        player_color = chess.BLACK if ia_color == chess.WHITE else chess.WHITE
        promotion_pending = None
        promotion_buttons = []
        ia_move_pending = False
        ia_move_timer = 0
        IA_DELAY = 400
        game_result = None
        reset_hover = False
        
        def reset_game():
            nonlocal board, move_log, dragging, drag_start, drag_piece, drag_pos
            nonlocal ia_color, player_color, promotion_pending, promotion_buttons
            nonlocal ia_move_pending, ia_move_timer, last_move, selected_square, valid_moves, game_result
            
            board = chess.Board()
            move_log.clear()
            dragging = False
            drag_start = None
            drag_piece = None
            drag_pos = (0, 0)
            selected_square = None
            valid_moves = []
            ia_color = random.choice([chess.WHITE, chess.BLACK])
            player_color = chess.BLACK if ia_color == chess.WHITE else chess.WHITE
            promotion_pending = None
            promotion_buttons = []
            ia_move_pending = False
            ia_move_timer = 0
            last_move = None
            game_result = None
        
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0  # Delta time in seconds
            
            # Actualizar animaciones
            animation_progress = self.update_animations(dt)
            
            # Obtener posición del ratón
            mouse_pos = pygame.mouse.get_pos()
            reset_hover = False
            
            # Dibujar el tablero y piezas
            if animation_progress is None:
                self.draw_board(self.screen)
                self.draw_last_move(self.screen, board, last_move)
                self.draw_highlights(self.screen, board, selected_square, valid_moves)
                self.draw_pieces(self.screen, board)
            else:
                self.draw_animation(self.screen, board, animation_progress)
            
            # Dibujar barra lateral
            button_rect, diff_buttons = self.draw_sidebar(self.screen, board, move_log, mouse_pos, reset_hover, ia_color, player_color)
            
            # Dibujar opciones de promoción si es necesario
            if promotion_pending:
                promotion_buttons = self.draw_promotion_choices(
                    self.screen, promotion_pending[2], (BOARD_SIZE//2 - 160, BOARD_SIZE//2 - 35))
            else:
                promotion_buttons = []
            
            # Dibujar pantalla de juego terminado
            if game_result:
                continue_btn = self.draw_game_over(self.screen, game_result)
            
            # Lógica de IA
            if not game_result and board.turn == ia_color and not promotion_pending and not dragging and animation_progress is None:
                if not ia_move_pending:
                    ia_move_pending = True
                    ia_move_timer = pygame.time.get_ticks()
                elif pygame.time.get_ticks() - ia_move_timer > IA_DELAY:
                    move = self.q_choose_move(board, ia_color)
                    if move:
                        san = board.san(move)
                        piece = board.piece_at(move.from_square)
                        color = 'w' if piece.color == chess.WHITE else 'b'
                        piece_type = piece.symbol().upper() if piece.piece_type != chess.PAWN else 'p'
                        piece_str = color + piece_type
                        
                        self.add_animation(move, board, piece_str)
                        
                        board.push(move)
                        self.guardar_experiencia_jugada(board, move, is_ia=True)
                        move_log.append(san)
                        last_move = move
                        
                        # Reproducir sonido de jaque si es necesario
                        if board.is_check():
                            self.play_sound("check")
                        
                        # Verificar estado del juego
                        if board.is_checkmate():
                            game_result = "0-1"
                            self.play_sound("checkmate")
                            self.guardar_experiencia("0-1")
                            self.export_pgn(move_log, ia_color, player_color, "0-1")
                            self.save_qtable()
                        elif board.is_stalemate() or board.is_insufficient_material() or \
                             board.can_claim_fifty_moves() or board.is_seventyfive_moves() or \
                             board.is_fivefold_repetition():
                            game_result = "1/2-1/2"
                            self.guardar_experiencia("1/2-1/2")
                            self.export_pgn(move_log, ia_color, player_color, "1/2-1/2")
                            self.save_qtable()
                    
                    ia_move_pending = False
            else:
                ia_move_pending = False
            
            pygame.display.flip()
            
            # Manejo de eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1 and not game_result:
                        x, y = event.pos
                        
                        # Botón de rendición
                        if button_rect.collidepoint(x, y):
                            reset_hover = True
                            game_result = "0-1"
                            self.guardar_experiencia("0-1")
                            self.export_pgn(move_log, ia_color, player_color, "0-1")
                            self.save_qtable()
                        
                        # Selector de dificultad
                        for rect, diff in diff_buttons:
                            if rect.collidepoint(x, y):
                                self.set_difficulty(diff)
                        
                        # Interacción con el tablero
                        if x < BOARD_SIZE and y < BOARD_SIZE and not promotion_pending and animation_progress is None:
                            col = x // SQ_SIZE
                            row = y // SQ_SIZE
                            square = chess.square(col, 7 - row)
                            piece = board.piece_at(square)
                            
                            if piece and piece.color == player_color:
                                dragging = True
                                drag_start = square
                                selected_square = square
                                drag_piece = piece
                                drag_pos = event.pos
                                # Obtener movimientos válidos para esta pieza
                                valid_moves = [move for move in board.legal_moves if move.from_square == square]
                                self.play_sound("click")
                            elif selected_square is not None:
                                # Intentar mover a la casilla seleccionada
                                move = chess.Move(selected_square, square)
                                if move in valid_moves:
                                    # Verificar promoción
                                    if drag_piece and drag_piece.piece_type == chess.PAWN and \
                                       (chess.square_rank(square) == 0 or chess.square_rank(square) == 7):
                                        promotion_pending = (selected_square, square, drag_piece.color)
                                    else:
                                        # Mover la pieza
                                        san = board.san(move)
                                        piece = board.piece_at(selected_square)
                                        color = 'w' if piece.color == chess.WHITE else 'b'
                                        piece_type = piece.symbol().upper() if piece.piece_type != chess.PAWN else 'p'
                                        piece_str = color + piece_type
                                        
                                        self.add_animation(move, board, piece_str)
                                        
                                        board.push(move)
                                        self.guardar_experiencia_jugada(board, move, is_ia=False)
                                        move_log.append(san)
                                        last_move = move
                                        
                                        # Reproducir sonido de jaque si es necesario
                                        if board.is_check():
                                            self.play_sound("check")
                                        
                                        # Verificar estado del juego
                                        if board.is_checkmate():
                                            game_result = "1-0"
                                            self.play_sound("checkmate")
                                            self.guardar_experiencia("1-0")
                                            self.export_pgn(move_log, ia_color, player_color, "1-0")
                                            self.save_qtable()
                                        elif board.is_stalemate() or board.is_insufficient_material() or \
                                             board.can_claim_fifty_moves() or board.is_seventyfive_moves() or \
                                             board.is_fivefold_repetition():
                                            game_result = "1/2-1/2"
                                            self.guardar_experiencia("1/2-1/2")
                                            self.export_pgn(move_log, ia_color, player_color, "1/2-1/2")
                                            self.save_qtable()
                                        
                                        selected_square = None
                                        valid_moves = []
                                else:
                                    # Seleccionar otra pieza
                                    if piece and piece.color == player_color:
                                        selected_square = square
                                        valid_moves = [move for move in board.legal_moves if move.from_square == square]
                                        self.play_sound("click")
                                    else:
                                        selected_square = None
                                        valid_moves = []
                        
                        # Opciones de promoción
                        elif promotion_pending and promotion_buttons:
                            clicked = False
                            for rect, piece_type in promotion_buttons:
                                if rect.collidepoint(x, y):
                                    move = chess.Move(promotion_pending[0], promotion_pending[1], promotion=piece_type)
                                    if move in board.legal_moves:
                                        san = board.san(move)
                                        piece = board.piece_at(promotion_pending[0])
                                        color = 'w' if piece.color == chess.WHITE else 'b'
                                        piece_type_str = piece.symbol().upper() if piece.piece_type != chess.PAWN else 'p'
                                        piece_str = color + piece_type_str
                                        
                                        self.add_animation(move, board, piece_str)
                                        self.play_sound("promote")
                                        
                                        board.push(move)
                                        self.guardar_experiencia_jugada(board, move, is_ia=False)
                                        move_log.append(san)
                                        last_move = move
                                        
                                        # Verificar estado del juego
                                        if board.is_checkmate():
                                            game_result = "1-0"
                                            self.play_sound("checkmate")
                                            self.guardar_experiencia("1-0")
                                            self.export_pgn(move_log, ia_color, player_color, "1-0")
                                            self.save_qtable()
                                        elif board.is_stalemate() or board.is_insufficient_material() or \
                                             board.can_claim_fifty_moves() or board.is_seventyfive_moves() or \
                                             board.is_fivefold_repetition():
                                            game_result = "1/2-1/2"
                                            self.guardar_experiencia("1/2-1/2")
                                            self.export_pgn(move_log, ia_color, player_color, "1/2-1/2")
                                            self.save_qtable()
                                    
                                    promotion_pending = None
                                    clicked = True
                                    selected_square = None
                                    valid_moves = []
                                    break
                            
                            if not clicked:
                                promotion_pending = None
                                selected_square = None
                                valid_moves = []
                    
                    # Botón de continuar en pantalla de juego terminado
                    elif game_result and event.button == 1:
                        x, y = event.pos
                        if continue_btn.collidepoint(x, y):
                            self.play_sound("click")
                            reset_game()
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1 and dragging and not promotion_pending and not game_result and animation_progress is None:
                        x, y = event.pos
                        dragging = False
                        
                        if x < BOARD_SIZE and y < BOARD_SIZE:
                            col = x // SQ_SIZE
                            row = y // SQ_SIZE
                            target = chess.square(col, 7 - row)
                            
                            if drag_start != target:
                                move = chess.Move(drag_start, target)
                                
                                # Verificar promoción
                                if drag_piece and drag_piece.piece_type == chess.PAWN and \
                                   (chess.square_rank(target) == 0 or chess.square_rank(target) == 7):
                                    promotion_pending = (drag_start, target, drag_piece.color)
                                elif move in valid_moves:
                                    # Mover la pieza
                                    san = board.san(move)
                                    piece = board.piece_at(drag_start)
                                    color = 'w' if piece.color == chess.WHITE else 'b'
                                    piece_type = piece.symbol().upper() if piece.piece_type != chess.PAWN else 'p'
                                    piece_str = color + piece_type
                                    
                                    self.add_animation(move, board, piece_str)
                                    
                                    board.push(move)
                                    self.guardar_experiencia_jugada(board, move, is_ia=False)
                                    move_log.append(san)
                                    last_move = move
                                    
                                    # Reproducir sonido de jaque si es necesario
                                    if board.is_check():
                                        self.play_sound("check")
                                    
                                    # Verificar estado del juego
                                    if board.is_checkmate():
                                        game_result = "1-0"
                                        self.play_sound("checkmate")
                                        self.guardar_experiencia("1-0")
                                        self.export_pgn(move_log, ia_color, player_color, "1-0")
                                        self.save_qtable()
                                    elif board.is_stalemate() or board.is_insufficient_material() or \
                                         board.can_claim_fifty_moves() or board.is_seventyfive_moves() or \
                                         board.is_fivefold_repetition():
                                        game_result = "1/2-1/2"
                                        self.guardar_experiencia("1/2-1/2")
                                        self.export_pgn(move_log, ia_color, player_color, "1/2-1/2")
                                        self.save_qtable()
                            
                            selected_square = None
                            valid_moves = []
                        
                        drag_start = None
                        drag_piece = None
                
                elif event.type == pygame.MOUSEMOTION:
                    if dragging:
                        drag_pos = event.pos
        
        # No llamar a pygame.quit() aquí, solo salir del bucle
        return

if __name__ == "__main__":
    game = ChessGame()
    game.main_menu()