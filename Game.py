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
from typing import Dict, List, Tuple, Optional, Deque
from collections import deque
import threading

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
    "easy": {"epsilon": 0.2, "search_depth": 1, "time_limit": 1.0},
    "normal": {"epsilon": 0.01, "search_depth": 2, "time_limit": 2.0},
    "hard": {"epsilon": 0.0, "search_depth": 3, "time_limit": 5.0}
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

# Tablas de piezas posicionales (Piece-Square Tables)
PAWN_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5, -10, 0,  0, -10, -5,  5,
    5, 10, 10, -20, -20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
]

KNIGHT_TABLE = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,  0,  0,  0,  0, -20, -40,
    -30,  0, 10, 15, 15, 10,  0, -30,
    -30,  5, 15, 20, 20, 15,  5, -30,
    -30,  0, 15, 20, 20, 15,  0, -30,
    -30,  5, 10, 15, 15, 10,  5, -30,
    -40, -20,  0,  5,  5,  0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50
]

BISHOP_TABLE = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,  0,  0,  0,  0,  0,  0, -10,
    -10,  0, 10, 10, 10, 10,  0, -10,
    -10,  5,  5, 10, 10,  5,  5, -10,
    -10,  0,  5, 10, 10,  5,  0, -10,
    -10,  5,  5,  5,  5,  5,  5, -10,
    -10,  0,  5,  0,  0,  5,  0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20
]

ROOK_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0,  5,  5,  0,  0,  0
]

QUEEN_TABLE = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10,  0,  0,  0,  0,  0,  0, -10,
    -10,  0,  5,  5,  5,  5,  0, -10,
    -5,  0,  5,  5,  5,  5,  0, -5,
    0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0, -10,
    -10,  0,  5,  0,  0,  0,  0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20
]

KING_MIDDLEGAME_TABLE = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20
]

KING_ENDGAME_TABLE = [
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10,  0,  0, -10, -20, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -30,  0,  0,  0,  0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50
]

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
        samples1 = SoundGenerator.generate_sine_wave(400, 0.05, 0.4)
        samples2 = SoundGenerator.generate_sine_wave(300, 0.05, 0.4)
        return np.concatenate((samples1, samples2))
    
    @staticmethod
    def generate_check_sound():
        """Generar sonido de jaque"""
        samples1 = SoundGenerator.generate_sine_wave(400, 0.05, 0.4)
        samples2 = SoundGenerator.generate_sine_wave(500, 0.05, 0.4)
        samples3 = SoundGenerator.generate_sine_wave(600, 0.05, 0.4)
        return np.concatenate((samples1, samples2, samples3))
    
    @staticmethod
    def generate_checkmate_sound():
        """Generar sonido de jaque mate"""
        samples1 = SoundGenerator.generate_sine_wave(600, 0.1, 0.5)
        samples2 = SoundGenerator.generate_sine_wave(500, 0.1, 0.5)
        samples3 = SoundGenerator.generate_sine_wave(400, 0.1, 0.5)
        return np.concatenate((samples1, samples2, samples3))
    
    @staticmethod
    def generate_promote_sound():
        """Generar sonido de promoción"""
        samples1 = SoundGenerator.generate_sine_wave(400, 0.05, 0.4)
        samples2 = SoundGenerator.generate_sine_wave(500, 0.05, 0.4)
        samples3 = SoundGenerator.generate_sine_wave(600, 0.05, 0.4)
        samples4 = SoundGenerator.generate_sine_wave(700, 0.1, 0.5)
        return np.concatenate((samples1, samples2, samples3, samples4))
    
    @staticmethod
    def generate_castle_sound():
        """Generar sonido de enroque"""
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

class ReplayBuffer:
    """Buffer de experiencia para aprendizaje por refuerzo"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        self.lock = threading.Lock()
    
    def add(self, state, action, reward, next_state, done):
        with self.lock:
            self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        with self.lock:
            if len(self.buffer) < batch_size:
                return random.sample(self.buffer, len(self.buffer))
            return random.sample(self.buffer, batch_size)
    
    def size(self):
        with self.lock:
            return len(self.buffer)

class ChessGame:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
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
        self.time_limit = DIFFICULTY_CONFIG[self.difficulty]["time_limit"]
        self.animation_queue = []
        self.current_animation = None
        self.animation_time = 0
        self.game_state = "menu"
        self.replay_buffer = ReplayBuffer(5000)
        self.killer_moves = {}
        self.history_table = {}
        
        self._attacked_by_white = set()
        self._attacked_by_black = set()
        
        self.load_images()
        self.generate_sounds()
        
        self.training_thread = None
        self.stop_training = False
        
    def generate_sounds(self):
        """Generar todos los sonidos del juego"""
        sound_generator = SoundGenerator()
        
        move_sound = pygame.mixer.Sound(buffer=sound_generator.generate_move_sound())
        capture_sound = pygame.mixer.Sound(buffer=sound_generator.generate_capture_sound())
        check_sound = pygame.mixer.Sound(buffer=sound_generator.generate_check_sound())
        checkmate_sound = pygame.mixer.Sound(buffer=sound_generator.generate_checkmate_sound())
        promote_sound = pygame.mixer.Sound(buffer=sound_generator.generate_promote_sound())
        castle_sound = pygame.mixer.Sound(buffer=sound_generator.generate_castle_sound())
        click_sound = pygame.mixer.Sound(buffer=sound_generator.generate_click_sound())
        
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
                surf = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
                color = (255, 0, 0) if piece.startswith('w') else (0, 0, 255)
                pygame.draw.rect(surf, color, (0, 0, SQ_SIZE, SQ_SIZE))
                self.images[piece] = surf
    
    def set_difficulty(self, difficulty):
        if difficulty in DIFFICULTY_CONFIG:
            self.difficulty = difficulty
            self.epsilon = DIFFICULTY_CONFIG[difficulty]["epsilon"]
            self.search_depth = DIFFICULTY_CONFIG[difficulty]["search_depth"]
            self.time_limit = DIFFICULTY_CONFIG[difficulty]["time_limit"]
            self.play_sound("click")
    
    def is_endgame(self, board):
        """Determinar si estamos en un final de juego"""
        queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
        rooks = len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK))
        bishops = len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.BLACK))
        knights = len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.BLACK))
        
        total_pieces = sum(len(board.pieces(pt, color)) for pt in PIECE_VALUES for color in [chess.WHITE, chess.BLACK])
        
        return (queens == 0 or 
                (queens <= 2 and rooks + bishops + knights <= 4) or
                total_pieces <= 10)
    
    def get_piece_square_value(self, piece, square, endgame):
        """Obtener el valor posicional de una pieza en una casilla"""
        if piece.piece_type == chess.PAWN:
            table = PAWN_TABLE
        elif piece.piece_type == chess.KNIGHT:
            table = KNIGHT_TABLE
        elif piece.piece_type == chess.BISHOP:
            table = BISHOP_TABLE
        elif piece.piece_type == chess.ROOK:
            table = ROOK_TABLE
        elif piece.piece_type == chess.QUEEN:
            table = QUEEN_TABLE
        elif piece.piece_type == chess.KING:
            table = KING_ENDGAME_TABLE if endgame else KING_MIDDLEGAME_TABLE
        else:
            return 0
        
        if piece.color == chess.WHITE:
            return table[square]
        else:
            return table[chess.square_mirror(square)]
    
    def order_moves(self, board, moves, tt_move=None):
        """Ordenar movimientos para mejorar la poda alfa-beta"""
        scored_moves = []
        
        for move in moves:
            score = 0
            
            if tt_move and move == tt_move:
                score += 10000
            
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                if captured_piece:
                    score += 10 * PIECE_VALUES[captured_piece.piece_type]
                    attacker = board.piece_at(move.from_square)
                    if attacker:
                        score += PIECE_VALUES[captured_piece.piece_type] * 10 - PIECE_VALUES[attacker.piece_type]
            
            if move.promotion:
                score += PIECE_VALUES[move.promotion] * 9
            
            if board.gives_check(move):
                score += 50
            
            if move in self.killer_moves.get(board.fen(), []):
                score += 900
            
            score += self.history_table.get((move.from_square, move.to_square), 0)
            
            scored_moves.append((score, move))
        
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return [move for _, move in scored_moves]
    
    def update_history(self, move, depth):
        """Actualizar la tabla de historia"""
        key = (move.from_square, move.to_square)
        self.history_table[key] = self.history_table.get(key, 0) + depth * depth
    
    def update_killer(self, move, board_fen):
        """Actualizar los killer moves"""
        if board_fen not in self.killer_moves:
            self.killer_moves[board_fen] = []
        
        if move not in self.killer_moves[board_fen]:
            self.killer_moves[board_fen].insert(0, move)
            if len(self.killer_moves[board_fen]) > 2:
                self.killer_moves[board_fen].pop()
    
    def quiescence_search(self, board, alpha, beta, color):
        """Búsqueda de quietud para evitar el horizonte effect"""
        stand_pat = self.evaluate_board(board, color)
        
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        
        captures = [move for move in board.legal_moves if board.is_capture(move)]
        captures = self.order_moves(board, captures)
        
        for move in captures:
            board.push(move)
            score = -self.quiescence_search(board, -beta, -alpha, -color)
            board.pop()
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        
        return alpha
    
    def evaluate_board(self, board, color=chess.WHITE):
        """Evaluar la posición del tablero con tablas posicionales"""
        endgame = self.is_endgame(board)
        eval = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                material = PIECE_VALUES[piece.piece_type]
                positional = self.get_piece_square_value(piece, square, endgame) / 100.0
                
                if piece.color == chess.WHITE:
                    eval += material + positional
                else:
                    eval -= material + positional
        
        mobility = len(list(board.legal_moves)) * 0.1
        if board.turn == chess.WHITE:
            eval += mobility
        else:
            eval -= mobility
        
        center_control = 0
        for square in CENTER_SQUARES:
            if board.is_attacked_by(chess.WHITE, square):
                center_control += 0.1
            if board.is_attacked_by(chess.BLACK, square):
                center_control -= 0.1
        eval += center_control
        
        pawn_structure = self.evaluate_pawn_structure(board)
        eval += pawn_structure
        
        if color == chess.BLACK:
            eval = -eval
        
        return eval
    
    def evaluate_pawn_structure(self, board):
        """Evaluar la estructura de peones"""
        score = 0
        
        for color in [chess.WHITE, chess.BLACK]:
            pawns = board.pieces(chess.PAWN, color)
            pawn_files = [chess.square_file(sq) for sq in pawns]
            
            for file in set(pawn_files):
                count = pawn_files.count(file)
                if count > 1:
                    if color == chess.WHITE:
                        score -= 0.2 * (count - 1)
                    else:
                        score += 0.2 * (count - 1)
            
            for sq in pawns:
                file = chess.square_file(sq)
                isolated = True
                for adj_file in [file - 1, file + 1]:
                    if 0 <= adj_file < 8 and any(f == adj_file for f in pawn_files):
                        isolated = False
                        break
                
                if isolated:
                    if color == chess.WHITE:
                        score -= 0.3
                    else:
                        score += 0.3
            
            for sq in pawns:
                file = chess.square_file(sq)
                rank = chess.square_rank(sq)
                passed = True
                
                for opp_sq in board.pieces(chess.PAWN, not color):
                    opp_file = chess.square_file(opp_sq)
                    opp_rank = chess.square_rank(opp_sq)
                    
                    if abs(opp_file - file) <= 1 and (
                        (color == chess.WHITE and opp_rank > rank) or
                        (color == chess.BLACK and opp_rank < rank)
                    ):
                        passed = False
                        break
                
                if passed:
                    advance_bonus = 0.1 * (rank if color == chess.WHITE else 7 - rank)
                    if color == chess.WHITE:
                        score += 0.5 + advance_bonus
                    else:
                        score -= 0.5 + advance_bonus
        
        return score
    
    def iterative_deepening(self, board, color, max_depth, time_limit):
        """Búsqueda con iterative deepening y límite de tiempo"""
        best_move = None
        start_time = time.time()
        
        for depth in range(1, max_depth + 1):
            if time.time() - start_time > time_limit:
                break
                
            score, move = self.minimax(
                board, depth, -float('inf'), float('inf'), 
                color == chess.WHITE, color, depth, start_time, time_limit
            )
            
            if move is not None:
                best_move = move
                
            if abs(score) > 10000:
                break
        
        return best_move
    
    def minimax(self, board, depth, alpha, beta, maximizing, color, depth_initial, start_time, time_limit):
        """Algoritmo minimax con poda alfa-beta y límite de tiempo"""
        if time.time() - start_time > time_limit:
            return 0, None
        
        key = board.fen()
        if key in self.transposition_table:
            entry_depth, entry_eval, entry_move, entry_flag = self.transposition_table[key]
            if entry_depth >= depth:
                if entry_flag == "exact":
                    return entry_eval, entry_move
                elif entry_flag == "lowerbound":
                    alpha = max(alpha, entry_eval)
                elif entry_flag == "upperbound":
                    beta = min(beta, entry_eval)
                
                if alpha >= beta:
                    return entry_eval, entry_move
        
        if depth == 0 or board.is_game_over():
            eval = self.quiescence_search(board, alpha, beta, 1 if maximizing else -1)
            return eval, None
        
        legal_moves = list(board.legal_moves)
        
        tt_move = self.transposition_table.get(key, (0, 0, None, ""))[2] if key in self.transposition_table else None
        legal_moves = self.order_moves(board, legal_moves, tt_move)
        
        best_move = None
        best_score = -float('inf') if maximizing else float('inf')
        
        for move in legal_moves:
            if time.time() - start_time > time_limit:
                break
                
            board.push(move)
            score, _ = self.minimax(
                board, depth - 1, alpha, beta, not maximizing, color, 
                depth_initial, start_time, time_limit
            )
            board.pop()
            
            if maximizing:
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, best_score)
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, best_score)
            
            if beta <= alpha:
                self.update_killer(move, key)
                self.update_history(move, depth)
                break
        
        flag = "exact"
        if best_score <= alpha:
            flag = "upperbound"
        elif best_score >= beta:
            flag = "lowerbound"
            
        self.transposition_table[key] = (depth, best_score, best_move, flag)
        
        return best_score, best_move
    
    def q_choose_move(self, board, color):
        """Elegir un movimiento usando Q-learning y minimax"""
        fen = board.fen()
        legal_moves = list(board.legal_moves)
        
        if self.search_depth > 0:
            move = self.iterative_deepening(board, color, self.search_depth + 2, self.time_limit)
            if move:
                return move
        
        if random.random() < self.epsilon:
            return random.choice(legal_moves)
        
        move_scores = []
        for move in legal_moves:
            q_value = self.qtable.get(fen + str(move), None)
            if q_value is not None:
                score = q_value
            else:
                board.push(move)
                score = self.evaluate_board(board, color)
                board.pop()
            move_scores.append((move, score))
        
        if color == chess.WHITE:
            best_score = max(score for _, score in move_scores)
            best_moves = [move for move, score in move_scores if score == best_score]
        else:
            best_score = min(score for _, score in move_scores)
            best_moves = [move for move, score in move_scores if score == best_score]
        
        return random.choice(best_moves) if best_moves else random.choice(legal_moves)
    
    def draw_board(self, screen):
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
            row = 7 - chess.square_rank(selected_square)
            col = chess.square_file(selected_square)
            s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
            s.fill(HIGHLIGHT_COLOR)
            screen.blit(s, (col * SQ_SIZE, row * SQ_SIZE))
            
            for move in valid_moves:
                if move.from_square == selected_square:
                    to_sq = move.to_square
                    row = 7 - chess.square_rank(to_sq)
                    col = chess.square_file(to_sq)
                    s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
                    
                    if board.piece_at(to_sq):
                        s.fill((255, 0, 0, 100))
                    else:
                        s.fill((0, 255, 0, 100))
                    
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
        pygame.draw.rect(screen, SIDEBAR_COLOR, (BOARD_SIZE, 0, SIDEBAR_WIDTH, HEIGHT))
        
        font_title = pygame.font.SysFont('Arial', 30, bold=True)
        title_text = font_title.render("♟ AJEDREZ PREMIUM", True, ACCENT_COLOR)
        screen.blit(title_text, (BOARD_SIZE + 30, 20))
        pygame.draw.line(screen, ACCENT_COLOR, (BOARD_SIZE + 20, 60), (WIDTH - 30, 60), 3)
        
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
        
        button_rect = pygame.Rect(BOARD_SIZE + 30, 160, 200, 40)
        color = BUTTON_HOVER if reset_hover else BUTTON_COLOR
        pygame.draw.rect(screen, color, button_rect, border_radius=10)
        font_btn = pygame.font.SysFont('Arial', 20, bold=True)
        btn_text = font_btn.render('Rendirse', True, BUTTON_TEXT)
        screen.blit(btn_text, (button_rect.x + 60, button_rect.y + 8))
        
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
        
        font2 = pygame.font.SysFont('Consolas', 17)
        pygame.draw.rect(screen, HIST_BG, (BOARD_SIZE + 30, 275, SIDEBAR_WIDTH - 60, HEIGHT - 295), border_radius=10)
        hist_title = font2.render('Historial:', True, ACCENT_COLOR)
        screen.blit(hist_title, (BOARD_SIZE + 40, 285))
        
        for i, move in enumerate(move_log[-18:]):
            move_text = font2.render(move, True, TEXT_COLOR)
            screen.blit(move_text, (BOARD_SIZE + 45, 315 + i * 22))
        
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
    
    def add_animation(self, move, board, piece_str):
        """Add an animation to the queue"""
        start_row = 7 - chess.square_rank(move.from_square)
        start_col = chess.square_file(move.from_square)
        end_row = 7 - chess.square_rank(move.to_square)
        end_col = chess.square_file(move.to_square)
        
        start_pos = (start_col * SQ_SIZE, start_row * SQ_SIZE)
        end_pos = (end_col * SQ_SIZE, end_row * SQ_SIZE)
        
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
            
            progress = 1 - (1 - progress) * (1 - progress)
            
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

        self.draw_board(screen)

        # Calcular casillas de origen y destino
        from_sq = chess.square(
            int(anim['start_pos'][0] // SQ_SIZE),
            7 - int(anim['start_pos'][1] // SQ_SIZE)
        )
        to_sq = chess.square(
            int(anim['end_pos'][0] // SQ_SIZE),
            7 - int(anim['end_pos'][1] // SQ_SIZE)
        )

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row = 7 - chess.square_rank(square)
                col = chess.square_file(square)
                color = 'w' if piece.color == chess.WHITE else 'b'
                piece_type = piece.symbol().upper() if piece.piece_type != chess.PAWN else 'p'
                piece_str = color + piece_type

                # No dibujar la pieza animada ni en origen ni en destino
                if piece_str == anim['piece_str'] and (square == from_sq or square == to_sq):
                    continue

                if piece_str in self.images:
                    screen.blit(self.images[piece_str], (col * SQ_SIZE, row * SQ_SIZE))

        # Dibuja la pieza animada en su posición interpolada
        if anim['piece_str'] in self.images:
            if progress > 0.8 and anim['is_capture']:
                bounce = math.sin((progress - 0.8) * 10 * math.pi) * 5
                screen.blit(self.images[anim['piece_str']], (x, y - bounce))
            else:
                screen.blit(self.images[anim['piece_str']], (x, y))
    
    def guardar_experiencia_jugada(self, board, move, is_ia):
        """Save move experience to Q-table"""
        fen = board.fen()
        reward = 0
        
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
    
    def train_from_buffer(self, batch_size=32):
        """Entrenar la Q-table desde el buffer de experiencia"""
        if self.replay_buffer.size() < batch_size:
            return
        
        batch = self.replay_buffer.sample(batch_size)
        for state, action, reward, next_state, done in batch:
            old_q = self.qtable.get(state + action, 0)
            
            if done:
                new_q = reward
            else:
                next_actions = [str(move) for move in chess.Board(next_state).legal_moves]
                next_max = max([self.qtable.get(next_state + a, 0) for a in next_actions], default=0)
                new_q = reward + GAMMA * next_max
            
            self.qtable[state + action] = old_q + ALPHA * (new_q - old_q)
    
    def background_training(self, num_games=100):
        """Entrenamiento en segundo plano"""
        self.stop_training = False
        for i in range(num_games):
            if self.stop_training:
                break
                
            board = chess.Board()
            while not board.is_game_over() and not self.stop_training:
                move = self.q_choose_move(board, board.turn)
                if move:
                    old_state = board.fen()
                    board.push(move)
                    new_state = board.fen()
                    
                    reward = 0
                    if board.is_checkmate():
                        reward = 10 if board.turn != chess.WHITE else -10
                    elif board.is_stalemate():
                        reward = 0
                    elif board.is_check():
                        reward = 1 if board.turn != chess.WHITE else -1
                    
                    self.replay_buffer.add(old_state, str(move), reward, new_state, board.is_game_over())
                    
                    if self.replay_buffer.size() % 100 == 0:
                        self.train_from_buffer()
            
            if i % 10 == 0:
                self.save_qtable()
        
        self.save_qtable()

    def start_training(self, num_games=100):
        """Iniciar entrenamiento en segundo plano"""
        if self.training_thread and self.training_thread.is_alive():
            return
            
        self.training_thread = threading.Thread(target=self.background_training, args=(num_games,))
        self.training_thread.daemon = True
        self.training_thread.start()

    def stop_background_training(self):
        """Detener el entrenamiento en segundo plano"""
        self.stop_training = True
        if self.training_thread:
            self.training_thread.join(timeout=1.0)
    
    def draw_main_menu(self, screen, selected_idx, options):
        """Draw the main menu"""
        for y in range(HEIGHT):
            color_val = 30 + (y / HEIGHT) * 10
            pygame.draw.line(screen, (color_val, color_val + 2, color_val + 10), (0, y), (WIDTH, y))

        font_title = pygame.font.SysFont('Arial', 48, bold=True)
        font_btn = pygame.font.SysFont('Arial', 32, bold=True)
        
        title = font_title.render("♟ Chess Game Premium", True, (20, 20, 30))
        screen.blit(title, (WIDTH//2 - title.get_width()//2 + 3, 83))
        
        title = font_title.render("♟ Chess Game Premium", True, ACCENT_COLOR)
        screen.blit(title, (WIDTH//2 - title.get_width()//2, 80))
        
        for idx, text in enumerate(options):
            btn_rect = pygame.Rect(WIDTH//2 - 180, 200 + idx*90, 360, 70)
            
            mouse_pos = pygame.mouse.get_pos()
            is_hover = btn_rect.collidepoint(mouse_pos)
            
            color = ACCENT_COLOR if idx == selected_idx or is_hover else BUTTON_COLOR
            pygame.draw.rect(screen, color, btn_rect, border_radius=18)

            if idx == selected_idx:
                pygame.draw.rect(screen, (255, 255, 255, 50), btn_rect, 3, border_radius=18)
            
            btn_text = font_btn.render(text, True, BUTTON_TEXT if idx != selected_idx else (30, 32, 40))
            screen.blit(btn_text, (btn_rect.x + (btn_rect.width - btn_text.get_width()) // 2, btn_rect.y + 18))
        
        font_footer = pygame.font.SysFont('Arial', 18)
        footer_text = font_footer.render("by GitHub Copilot", True, (120, 120, 120))
        screen.blit(footer_text, (WIDTH - footer_text.get_width() - 20, HEIGHT - 40))
        
        for i in range(20):
            x = random.randint(0, WIDTH)
            y = random.randint(0, HEIGHT)
            size = random.randint(1, 3)
            pygame.draw.circle(screen, (200, 200, 255, 100), (x, y), size)
    
            
            pygame.display.flip()
            pygame.time.delay(30)
    
    def draw_game_over(self, screen, result):
        """Draw game over screen"""
        s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        screen.blit(s, (0, 0))
        
        result_rect = pygame.Rect(WIDTH//2 - 200, HEIGHT//2 - 100, 400, 200)
        pygame.draw.rect(screen, SIDEBAR_COLOR, result_rect, border_radius=20)
        pygame.draw.rect(screen, ACCENT_COLOR, result_rect, 3, border_radius=20)
        
        font = pygame.font.SysFont('Arial', 36, bold=True)
        if result == "1-0":
            text = font.render("¡Has ganado!", True, (180, 255, 180))
        elif result == "0-1":
            text = font.render("La IA ha ganado", True, ACCENT_COLOR)
        else:
            text = font.render("Empate", True, (200, 200, 200))
        
        screen.blit(text, (WIDTH//2 - text.get_width()//2, HEIGHT//2 - 70))
        
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
        options = ["Jugar contra la IA", "Entrenar IA", "Salir"]
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
                                elif text == "Salir":
                                    running = False
                                    break

            if not running:
                break

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
        IA_DELAY = 100
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
            dt = self.clock.tick(FPS) / 1000.0
            
            animation_progress = self.update_animations(dt)
            
            mouse_pos = pygame.mouse.get_pos()
            reset_hover = False
            
            if animation_progress is None:
                self.draw_board(self.screen)
                self.draw_last_move(self.screen, board, last_move)
                self.draw_highlights(self.screen, board, selected_square, valid_moves)
                self.draw_pieces(self.screen, board)
            else:
                self.draw_animation(self.screen, board, animation_progress)
            
            button_rect, diff_buttons = self.draw_sidebar(self.screen, board, move_log, mouse_pos, reset_hover, ia_color, player_color)
            
            if promotion_pending:
                promotion_buttons = self.draw_promotion_choices(
                    self.screen, promotion_pending[2], (BOARD_SIZE//2 - 160, BOARD_SIZE//2 - 35))
            else:
                promotion_buttons = []
            
            if game_result:
                continue_btn = self.draw_game_over(self.screen, game_result)
            
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
                        
                        if board.is_check():
                            self.play_sound("check")
                        
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
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1 and not game_result:
                        x, y = event.pos
                        
                        if button_rect.collidepoint(x, y):
                            reset_hover = True
                            game_result = "0-1"
                            self.guardar_experiencia("0-1")
                            self.export_pgn(move_log, ia_color, player_color, "0-1")
                            self.save_qtable()
                        
                        for rect, diff in diff_buttons:
                            if rect.collidepoint(x, y):
                                self.set_difficulty(diff)
                        
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
                                valid_moves = [move for move in board.legal_moves if move.from_square == square]
                                self.play_sound("click")
                            elif selected_square is not None:
                                move = chess.Move(selected_square, square)
                                if move in valid_moves:
                                    if drag_piece and drag_piece.piece_type == chess.PAWN and \
                                       (chess.square_rank(square) == 0 or chess.square_rank(square) == 7):
                                        promotion_pending = (selected_square, square, drag_piece.color)
                                    else:
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
                                        
                                        if board.is_check():
                                            self.play_sound("check")
                                        
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
                                    if piece and piece.color == player_color:
                                        selected_square = square
                                        valid_moves = [move for move in board.legal_moves if move.from_square == square]
                                        self.play_sound("click")
                                    else:
                                        selected_square = None
                                        valid_moves = []
                        
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
                                
                                if drag_piece and drag_piece.piece_type == chess.PAWN and \
                                   (chess.square_rank(target) == 0 or chess.square_rank(target) == 7):
                                    promotion_pending = (drag_start, target, drag_piece.color)
                                elif move in valid_moves:
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
                                    
                                    if board.is_check():
                                        self.play_sound("check")
                                    
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
        
        return

if __name__ == "__main__":
    game = ChessGame()
    game.main_menu()