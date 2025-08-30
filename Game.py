import json
import os

import collections
# --- Q-learning IA ---
Q_FILE = "ia_qtable.json"
ALPHA = 0.5  # tasa de aprendizaje
GAMMA = 0.9  # factor de descuento

def load_qtable():
	if os.path.exists(Q_FILE):
		with open(Q_FILE, 'r') as f:
			return json.load(f)
	return {}

def save_qtable(qtable):
	with open(Q_FILE, 'w') as f:
		json.dump(qtable, f)

qtable = load_qtable()
import random

# --- IA básica ---
def evaluate_board(board):
	# Suma simple de material
	values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
	eval = 0
	for piece_type in values:
		eval += len(board.pieces(piece_type, chess.WHITE)) * values[piece_type]
		eval -= len(board.pieces(piece_type, chess.BLACK)) * values[piece_type]
	return eval


# Q-learning: elegir la mejor jugada conocida o explorar
def q_choose_move(board, color, epsilon=0.1):
	fen = board.fen()
	legal_moves = list(board.legal_moves)
	if random.random() < epsilon:
		return random.choice(legal_moves)
	# Buscar si hay experiencia previa en la Q-table
	q_values = [qtable.get(fen + str(move), None) for move in legal_moves]
	if any(q is not None for q in q_values):
		# Si hay experiencia, elegir la mejor jugada según Q
		best_q = -float('inf') if color == chess.BLACK else float('inf')
		best_moves = []
		for move, q in zip(legal_moves, q_values):
			q = q if q is not None else (0 if color == chess.BLACK else 0)
			if color == chess.BLACK:
				if q > best_q:
					best_q = q
					best_moves = [move]
				elif q == best_q:
					best_moves.append(move)
			else:
				if q < best_q:
					best_q = q
					best_moves = [move]
				elif q == best_q:
					best_moves.append(move)
		return random.choice(best_moves)
	else:
		# Si no hay experiencia, usar evaluate_board para elegir la jugada más prometedora
		best_score = -float('inf') if color == chess.WHITE else float('inf')
		best_moves = []
		for move in legal_moves:
			board.push(move)
			score = evaluate_board(board)
			board.pop()
			if color == chess.WHITE:
				if score > best_score:
					best_score = score
					best_moves = [move]
				elif score == best_score:
					best_moves.append(move)
			else:
				if score < best_score:
					best_score = score
					best_moves = [move]
				elif score == best_score:
					best_moves.append(move)
		return random.choice(best_moves)

# Estructura para "aprender" (registro de partidas)
game_memory = []
stats = collections.defaultdict(int)  # 'games', 'ia_wins', 'player_wins', 'draws'
# Chess game from scratch using pygame and python-chess
import pygame
import sys
import chess


# --- Configuración ---

# Tamaño mejorado para mejor presentación
BOARD_SIZE = 700
SIDEBAR_WIDTH = 320
WIDTH, HEIGHT = BOARD_SIZE + SIDEBAR_WIDTH, BOARD_SIZE
DIMENSION = 8
SQ_SIZE = BOARD_SIZE // DIMENSION
FPS = 60
IMAGES = {}


# Escala de grises para el tablero
LIGHT_SQUARE = (230, 230, 230)  # blanco suave
DARK_SQUARE = (80, 80, 80)      # gris oscuro

# Colores UI
SIDEBAR_COLOR = (36, 37, 46)
TEXT_COLOR = (230, 230, 230)
ACCENT_COLOR = (255, 204, 0)
HIST_BG = (50, 50, 65)
BUTTON_COLOR = (60, 60, 80)
BUTTON_HOVER = (90, 90, 120)
BUTTON_TEXT = (255, 255, 255)

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

def draw_pieces(screen, board):
	for square in chess.SQUARES:
		piece = board.piece_at(square)
		if piece:
			row = 7 - chess.square_rank(square)
			col = chess.square_file(square)
			color = 'w' if piece.color == chess.WHITE else 'b'
			# Siempre usar mayúscula para piezas excepto peones, que son minúscula
			if piece.piece_type == chess.PAWN:
				piece_str = color + 'p'
			else:
				piece_str = color + piece.symbol().upper()
			if piece_str in IMAGES:
				screen.blit(IMAGES[piece_str], (col*SQ_SIZE, row*SQ_SIZE))

def draw_sidebar(screen, board, move_log, mouse_pos, reset_hover, ia_color, player_color):
	# Fondo barra lateral
	pygame.draw.rect(screen, SIDEBAR_COLOR, (BOARD_SIZE, 0, SIDEBAR_WIDTH, HEIGHT))
	# Título
	font_title = pygame.font.SysFont('Arial', 30, bold=True)
	title_text = font_title.render("♟ AJEDREZ PREMIUM", True, ACCENT_COLOR)
	screen.blit(title_text, (BOARD_SIZE + 30, 20))
	# Línea decorativa
	pygame.draw.line(screen, ACCENT_COLOR, (BOARD_SIZE + 20, 60), (WIDTH - 30, 60), 3)
	# Turno
	font = pygame.font.SysFont('Arial', 22, bold=True)
	turn = 'Blancas' if board.turn == chess.WHITE else 'Negras'
	turn_text = font.render(f'Turno: {turn}', True, TEXT_COLOR)
	screen.blit(turn_text, (BOARD_SIZE + 30, 70))
	# Quién inicia
	starter = 'IA' if (ia_color == chess.WHITE and board.fullmove_number == 1 and board.turn == chess.WHITE) or (ia_color == chess.BLACK and board.fullmove_number == 1 and board.turn == chess.BLACK) else 'Jugador'
	font_starter = pygame.font.SysFont('Arial', 18, bold=True)
	starter_color = ACCENT_COLOR if starter == 'IA' else (180,255,180)
	screen.blit(font_starter.render(f'Inicia: {starter}', True, starter_color), (BOARD_SIZE + 30, 100))
	# Mostrar roles y colores
	font_role = pygame.font.SysFont('Arial', 18)
	ia_str = 'Blancas' if ia_color == chess.WHITE else 'Negras'
	player_str = 'Blancas' if player_color == chess.WHITE else 'Negras'
	screen.blit(font_role.render(f'IA: {ia_str}', True, ACCENT_COLOR), (BOARD_SIZE + 30, 130))
	screen.blit(font_role.render(f'Jugador: {player_str}', True, (180,255,180)), (BOARD_SIZE + 140, 130))
	# Botón reiniciar
	button_rect = pygame.Rect(BOARD_SIZE + 30, 160, 200, 40)
	color = BUTTON_HOVER if reset_hover else BUTTON_COLOR
	pygame.draw.rect(screen, color, button_rect, border_radius=10)
	font_btn = pygame.font.SysFont('Arial', 20, bold=True)
	btn_text = font_btn.render('Reiniciar partida', True, BUTTON_TEXT)
	screen.blit(btn_text, (button_rect.x + 20, button_rect.y + 8))
	# Historial
	font2 = pygame.font.SysFont('Consolas', 17)
	pygame.draw.rect(screen, HIST_BG, (BOARD_SIZE + 30, 220, SIDEBAR_WIDTH - 60, HEIGHT - 240), border_radius=10)
	hist_title = font2.render('Historial:', True, ACCENT_COLOR)
	screen.blit(hist_title, (BOARD_SIZE + 40, 230))
	# Movimientos
	for i, move in enumerate(move_log[-18:]):
		move_text = font2.render(move, True, TEXT_COLOR)
		screen.blit(move_text, (BOARD_SIZE + 45, 260 + i*22))
	# Progreso de la IA
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

	# Elegir aleatoriamente el color de la IA y del jugador
	ia_color = random.choice([chess.WHITE, chess.BLACK])
	player_color = chess.BLACK if ia_color == chess.WHITE else chess.WHITE

	game_memory = []
	running = True

	def reset_game():
		nonlocal board, move_log, dragging, drag_start, drag_piece, drag_pos, ia_color, player_color
		board = chess.Board()
		move_log.clear()
		dragging = False
		drag_start = None
		drag_piece = None
		drag_pos = (0, 0)
		ia_color = random.choice([chess.WHITE, chess.BLACK])
		player_color = chess.BLACK if ia_color == chess.WHITE else chess.WHITE
		game_memory.clear()

	while running:
		mouse_pos = pygame.mouse.get_pos()
		reset_hover = False

		# IA juega automáticamente cuando es su turno
		if board.turn == ia_color and not board.is_game_over():
			# Guardar posición y acción para Q-learning
			fen = board.fen()
			move = q_choose_move(board, ia_color)
			game_memory.append((fen, move.uci()))
			san = board.san(move)
			board.push(move)
			move_log.append(san)

		# Dibuja primero para obtener el button_rect actualizado
		draw_board(screen)
		if dragging and drag_piece:
			temp_board = board.copy()
			temp_board.remove_piece_at(drag_start)
			draw_pieces(screen, temp_board)
			x, y = drag_pos
			color = 'w' if drag_piece.color == chess.WHITE else 'b'
			if drag_piece.piece_type == chess.PAWN:
				piece_str = color + 'p'
			else:
				piece_str = color + drag_piece.symbol().upper()
			if piece_str in IMAGES:
				screen.blit(IMAGES[piece_str], (x - SQ_SIZE//2, y - SQ_SIZE//2))
		else:
			draw_pieces(screen, board)

		button_rect = draw_sidebar(screen, board, move_log, mouse_pos, False, ia_color, player_color)
		pygame.display.flip()

		# Ahora maneja los eventos con button_rect válido
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_r:
					# Guardar Q-learning al terminar la partida y actualizar stats
					if board.is_game_over():
						result = board.result()
						stats['games'] += 1
						if (result == '0-1' and ia_color == chess.BLACK) or (result == '1-0' and ia_color == chess.WHITE):
							stats['ia_wins'] += 1
						elif (result == '1-0' and ia_color == chess.BLACK) or (result == '0-1' and ia_color == chess.WHITE):
							stats['player_wins'] += 1
						else:
							stats['draws'] += 1
						reward = 1 if (result == '0-1' and ia_color == chess.BLACK) or (result == '1-0' and ia_color == chess.WHITE) else -1 if (result == '1-0' and ia_color == chess.BLACK) or (result == '0-1' and ia_color == chess.WHITE) else 0
						for i, (fen, move_uci) in enumerate(reversed(game_memory)):
							key = fen + move_uci
							old_q = qtable.get(key, 0)
							qtable[key] = old_q + ALPHA * (reward - old_q)
							reward *= GAMMA
						save_qtable(qtable)
					reset_game()
				elif event.key == pygame.K_u:
					if board.move_stack:
						board.pop()
						if move_log:
							move_log.pop()
			elif event.type == pygame.MOUSEBUTTONDOWN:
				if event.button == 1:
					x, y = event.pos
					if button_rect and button_rect.collidepoint(x, y):
						# Guardar Q-learning al terminar la partida y actualizar stats
						if board.is_game_over():
							result = board.result()
							stats['games'] += 1
							if (result == '0-1' and ia_color == chess.BLACK) or (result == '1-0' and ia_color == chess.WHITE):
								stats['ia_wins'] += 1
							elif (result == '1-0' and ia_color == chess.BLACK) or (result == '0-1' and ia_color == chess.WHITE):
								stats['player_wins'] += 1
							else:
								stats['draws'] += 1
							reward = 1 if (result == '0-1' and ia_color == chess.BLACK) or (result == '1-0' and ia_color == chess.WHITE) else -1 if (result == '1-0' and ia_color == chess.BLACK) or (result == '0-1' and ia_color == chess.WHITE) else 0
							for i, (fen, move_uci) in enumerate(reversed(game_memory)):
								key = fen + move_uci
								old_q = qtable.get(key, 0)
								qtable[key] = old_q + ALPHA * (reward - old_q)
								reward *= GAMMA
							save_qtable(qtable)
						reset_game()
					elif x < BOARD_SIZE and y < BOARD_SIZE:
						col = x // SQ_SIZE
						row = y // SQ_SIZE
						square = chess.square(col, 7 - row)
						piece = board.piece_at(square)
						# Solo permitir mover si es el turno del jugador
						if piece and piece.color == board.turn and board.turn == player_color:
							dragging = True
							drag_start = square
							drag_piece = piece
							drag_pos = (x, y)
			elif event.type == pygame.MOUSEBUTTONUP:
				if event.button == 1 and dragging:
					x, y = event.pos
					if x < BOARD_SIZE and y < BOARD_SIZE:
						col = x // SQ_SIZE
						row = y // SQ_SIZE
						target = chess.square(col, 7 - row)
						move = chess.Move(drag_start, target)
						# Promoción automática a dama
						if drag_piece and drag_piece.piece_type == chess.PAWN and (chess.square_rank(target) == 0 or chess.square_rank(target) == 7):
							move = chess.Move(drag_start, target, promotion=chess.QUEEN)
						if move in board.legal_moves:
							san = board.san(move)
							board.push(move)
							move_log.append(san)
					dragging = False
					drag_start = None
					drag_piece = None
			elif event.type == pygame.MOUSEMOTION:
				if dragging:
					drag_pos = event.pos

		clock.tick(FPS)

	pygame.quit()
	sys.exit()

if __name__ == '__main__':
	main()
