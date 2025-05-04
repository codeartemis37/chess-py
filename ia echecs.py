import pygame
import sys
import copy
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageOps
import random
from time import *

# Initialisation de Pygame
pygame.init()

# Constantes
WIDTH, HEIGHT = 400, 400
GRID_SIZE = 8
SQUARE_SIZE = WIDTH // GRID_SIZE

# Couleurs
LIGHT_GREEN = (144, 238, 144)  # Vert clair
OCHRE = (204, 119, 34)  # Ocre
HIGHLIGHT = (255, 255, 0)
PROMOTION_BG = (200, 200, 200)  # Fond pour la fenêtre de promotion

# URLs des images des piÃƒÂ¨ces
PIECE_URLS = {
    'P': "https://cdn-icons-png.flaticon.com/512/8260/8260946.png",
    'R': "https://cdn-icons-png.flaticon.com/128/1310/1310876.png",
    'N': "https://cdn-icons-png.flaticon.com/128/15731/15731815.png",
    'B': "https://cdn-icons-png.flaticon.com/512/10967/10967482.png",
    'Q': "https://cdn-icons-png.flaticon.com/512/11389/11389245.png",
    'K': "https://cdn-icons-png.flaticon.com/512/11389/11389221.png"
}

AI_ERROR_RATE = 0.1  # 10% de chance de faire une erreur
depth_ai_raisonment = 3

# Fonction pour inverser les couleurs d'une image
def invert_image(image):
    """Inverts the colors of a Pygame image, preserving transparency."""
    img = Image.frombytes('RGBA', (SQUARE_SIZE, SQUARE_SIZE), pygame.image.tostring(image, 'RGBA'))
    # Split into color and alpha channels
    r, g, b, a = img.split()

    # Invert the color channels
    r = ImageOps.invert(r)
    g = ImageOps.invert(g)
    b = ImageOps.invert(b)

    # Merge back with the alpha channel
    inverted = Image.merge('RGBA', (r, g, b, a))
    mode = inverted.mode
    size = inverted.size
    data = inverted.tobytes()
    inverted_pygame = pygame.image.fromstring(data, size, mode)
    return inverted_pygame

# Fonction pour charger une image
def load_image(piece, url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img = img.resize((SQUARE_SIZE, SQUARE_SIZE), Image.LANCZOS)

        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        pygame_img = pygame.image.fromstring(img.tobytes(), img.size, img.mode)
        return piece, pygame_img
    except requests.exceptions.RequestException as e:
        print(f"Error loading image for {piece}: {e}")
        return piece, None
    except Exception as e:
        print(f"Error processing image for {piece}: {e}")
        return piece, None

# Chargement des images des piÃƒÂ¨ces avec multithreading
PIECES = {}
with ThreadPoolExecutor(max_workers=len(PIECE_URLS)) as executor:
    future_to_piece = {executor.submit(load_image, piece, url): piece for piece, url in PIECE_URLS.items()}
    for future in as_completed(future_to_piece):
        piece, img = future.result()
        if img:
            PIECES[piece.lower()] = img
            PIECES[piece] = invert_image(img)
        else:
            print(f"Failed to load image for {piece}. Using placeholder.")
            PIECES[piece] = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
            PIECES[piece].fill((255,0,255))
            PIECES[piece.lower()] = PIECES[piece]

# Configuration de l'ÃƒÂ©cran
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Jeu d'ÃƒÂ©checs")

# Valeurs des piÃƒÂ¨ces (White negative, Black positive)
PIECE_VALUES = {
    'p': 100, 'n': 320, 'b': 330, 'r': 500, 'q': 900, 'k': 20000,
    'P': -100, 'N': -320, 'B': -330, 'R': -500, 'Q': -900, 'K': -20000,
    '.': 0
}

# Piece Square Tables (PSTs) - a simple form of chess knowledge
# These tables give a bonus/penalty for having a piece on a certain square
# More advanced engines have separate tables for different game stages

PAWN_PST = [
    [0,  0,  0,  0,  0,  0,  0,  0],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [5,  5, 10, 25, 25, 10,  5,  5],
    [0,  0,  0, 20, 20,  0,  0,  0],
    [5, -5, -10,  0,  0, -10, -5,  5],
    [5, 10, 10, -20, -20, 10, 10,  5],
    [0,  0,  0,  0,  0,  0,  0,  0]
]

KNIGHT_PST = [
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20,   0,   0,   0,   0, -20, -40],
    [-30,   0,  10,  15,  15,  10,   0, -30],
    [-30,   5,  15,  20,  20,  15,   5, -30],
    [-30,   0,  15,  20,  20,  15,   0, -30],
    [-30,   5,  10,  15,  15,  10,   5, -30],
    [-40, -20,   0,   5,   5,   0, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50]
]

BISHOP_PST = [
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10,   0,   0,   0,   0,   0,   0, -10],
    [-10,   0,   5,  10,  10,   5,   0, -10],
    [-10,   5,   5,  10,  10,   5,   5, -10],
    [-10,   0,  10,  10,  10,  10,   0, -10],
    [-10,  10,  10,  10,  10,  10,  10, -10],
    [-10,   5,   0,   0,   0,   0,   5, -10],
    [-20, -10, -10, -10, -10, -10, -10, -20]
]

ROOK_PST = [
    [0,  0,  0,  0,  0,  0,  0,  0],
    [5, 10, 10, 10, 10, 10, 10,  5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [0,  0,  0,  5,  5,  0,  0,  0]
]

QUEEN_PST = [
    [-20, -10, -10, -5, -5, -10, -10, -20],
    [-10,   0,   0,   0,   0,   0,   0, -10],
    [-10,   0,   5,   5,   5,   5,   0, -10],
    [-5,   0,   5,   5,   5,   5,   0, -5],
    [0,   0,   5,   5,   5,   5,   0, -5],
    [-10,   5,   5,   5,   5,   5,   0, -10],
    [-10,   0,   5,   0,   0,   0,   0, -10],
    [-20, -10, -10, -5, -5, -10, -10, -20]
]

KING_PST = [
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-20, -30, -30, -40, -40, -30, -30, -20],
    [-10, -20, -20, -20, -20, -20, -20, -10],
    [20,  20,   0,   0,   0,   0,  20,  20],
    [20,  30,  10,   0,   0,  10,  30,  20]
]

# État du jeu
game_state = {
    'board': [
        ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
        ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
        ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
    ],
    'white_can_castle_kingside': True,
    'white_can_castle_queenside': True,
    'black_can_castle_kingside': True,
    'black_can_castle_queenside': True,
    'en_passant_target': [],  # Case où la prise en passant est possible
    'selected': None,
    'player_turn': True,  # True pour Blanc, False pour Noir
    'promotion_position': None,  # Position où la promotion doit avoir lieu
    'promotion_choices': ['Q', 'R', 'B', 'N'] # Choix possibles pour la promotion
}

def is_valid_position(x, y):
    return 0 <= x < 8 and 0 <= y < 8

def get_piece_moves(board, x, y, game_state, in_recursion = False):
    piece = board[y][x].lower()
    color = 1 if board[y][x].isupper() else -1
    moves = []

    if piece == 'p':
        # Mouvement simple du pion
        nx, ny = x, (y - color)
        if is_valid_position(nx, ny) and board[ny][nx] == '.':
            moves.append((nx, ny))
            # Double mouvement depuis la position initiale
            if (color == -1 and y == 1) or (color == 1 and y == 6):
                nx, ny = x, y - (2 * color)
                if board[ny][nx] == '.':
                    moves.append((nx, ny))
        # Captures en diagonale
        for dx in [-1, 1]:
            nx, ny = x + dx, y - color
            if is_valid_position(nx, ny) and board[ny][nx] != '.' and board[ny][nx].isupper() != board[y][x].isupper():
                moves.append((nx, ny))
        
        # Prise en passant
        if game_state['en_passant_target'] and (nx, ny) in game_state['en_passant_target']:
            moves.append((nx, ny))

    elif piece == 'n':
        for dx, dy in [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]:
            nx, ny = x + dx, y + dy
            if is_valid_position(nx, ny) and (board[ny][nx] == '.' or board[ny][nx].isupper() != board[y][x].isupper()):
                moves.append((nx, ny))
    elif piece == 'k':
        for dx, dy in [(a, b) for a in [-1, 0, 1] for b in [-1, 0, 1] if (a, b) != (0, 0)]:
            nx, ny = x + dx, y + dy
            if is_valid_position(nx, ny) and (board[ny][nx].isupper() != board[y][x].isupper()):
                moves.append((nx, ny))
    elif piece in ['b', 'r', 'q']:
        directions = {
            'b': [(1, 1), (1, -1), (-1, 1), (-1, -1)],
            'r': [(1, 0), (-1, 0), (0, 1), (0, -1)],
            'q': [(1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)]
        }
        for dx, dy in directions[piece]:
            nx, ny = x + dx, y + dy
            while is_valid_position(nx, ny):
                if board[ny][nx] == '.':
                    moves.append((nx, ny))
                elif board[ny][nx].isupper() != board[y][x].isupper():
                    moves.append((nx, ny))
                    break
                else:
                    break
                if piece == 'k':
                    break
                nx, ny = nx + dx, ny + dy
    
    # Roque
    if piece == 'k' and not in_recursion:
        # Roque avec vérification de l'échec
        if color == 1:  # Blanc
            if (game_state['white_can_castle_kingside'] 
                and board[7][5] == '.' 
                and board[7][6] == '.' 
                and not is_check(board, game_state['player_turn'], game_state)):
                    moves.append((6, 7))
            if (game_state['white_can_castle_queenside'] 
                and board[7][3] == '.' 
                and board[7][2] == '.' 
                and board[7][1] == '.' 
                and not is_check(board, game_state['player_turn'], game_state)):
                    moves.append((2, 7))
        else:  # Noir
            if (game_state['black_can_castle_kingside'] 
                and board[0][5] == '.' 
                and board[0][6] == '.' 
                and not is_check(board, game_state['player_turn'], game_state)):
                    moves.append((6, 0))
            if (game_state['black_can_castle_queenside'] 
                and board[0][3] == '.' 
                and board[0][2] == '.' 
                and board[0][1] == '.' 
                and not is_check(board, game_state['player_turn'], game_state)):
                    moves.append((2, 0))
    return moves

def evaluate_board(board):
    score = 0
    for y in range(8):
        for x in range(8):
            piece = board[y][x]
            if piece != '.':
                piece_value = PIECE_VALUES[piece]
                # White is negative, Black is positive, so we want to *add*
                # the white score and *subtract* the black score to make it
                # white-relative
                #score -= piece_value # Flip the sign

                # Add piece square table value
                piece_lower = piece.lower()
                if piece_lower == 'p':
                    pst = PAWN_PST
                elif piece_lower == 'n':
                    pst = KNIGHT_PST
                elif piece_lower == 'b':
                    pst = BISHOP_PST
                elif piece_lower == 'r':
                    pst = ROOK_PST
                elif piece_lower == 'q':
                    pst = QUEEN_PST
                elif piece_lower == 'k':
                    pst = KING_PST
                else:
                    pst = None

                if pst:
                    if piece.isupper():  # White piece
                        score += pst[y][x]  # Add for white
                    else:  # Black piece
                        score -= pst[7-y][7-x]  # Subtract for black, flip indices
                score += piece_value
    return -score

def get_all_possible_moves(board, maximizing_player, game_state):
    moves = []
    for y in range(8):
        for x in range(8):
            piece = board[y][x]
            if piece != '.':
                if (maximizing_player and piece.isupper()) or (not maximizing_player and piece.islower()):
                    for to_x, to_y in get_piece_moves(board, x, y, game_state):
                        moves.append((x, y, to_x, to_y))
    return moves


def apply_error_to_evaluation(evaluation, error_rate):
    if random.random() < error_rate:
        # Appliquer une erreur aléatoire à l'évaluation
        error = random.uniform(-200, 200)  # Erreur entre -200 et 200 points
        return evaluation + error
    return evaluation

def minimax(board, depth, maximizing_player, game_state, alpha=float('-inf'), beta=float('inf')):
    if depth == 0 or is_game_over(board):
        evaluation = evaluate_board(board)
        return apply_error_to_evaluation(evaluation, AI_ERROR_RATE), None

    possible_moves = get_all_possible_moves(board, maximizing_player, game_state)

    if maximizing_player:
        max_eval = float('-inf')
        best_move = None
        for move in possible_moves:
            new_board = copy.deepcopy(board)
            new_game_state = copy.deepcopy(game_state)
            make_move(new_board, move, new_game_state)
            eval, _ = minimax(new_board, depth - 1, False, new_game_state, alpha, beta)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        best_move = None
        for move in possible_moves:
            new_board = copy.deepcopy(board)
            new_game_state = copy.deepcopy(game_state)
            make_move(new_board, move, new_game_state)
            eval, _ = minimax(new_board, depth - 1, True, new_game_state, alpha, beta)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move

def draw_board():
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            color = LIGHT_GREEN if (x + y) % 2 == 0 else OCHRE
            pygame.draw.rect(screen, color, (x * SQUARE_SIZE, y * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

def draw_pieces(board):
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            piece = board[y][x]
            if piece != '.':
                screen.blit(PIECES[piece], (x * SQUARE_SIZE, y * SQUARE_SIZE))

def highlight_square(x, y):
    pygame.draw.rect(screen, HIGHLIGHT, (x * SQUARE_SIZE, y * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 4)

def get_square_under_mouse():
    x, y = pygame.mouse.get_pos()
    return x // SQUARE_SIZE, y // SQUARE_SIZE

def handle_promotion(board, x, y, piece):
    board[y][x] = piece
    game_state['promotion_position'] = None

def draw_promotion_window(x, y, color):
    """Draws a window allowing the player to choose a piece to promote to."""
    window_width = 4 * SQUARE_SIZE
    window_height = SQUARE_SIZE
    window_x = x * SQUARE_SIZE - (window_width - SQUARE_SIZE) // 2
    window_y = y * SQUARE_SIZE - (window_height - SQUARE_SIZE) // 2

    # Background for the promotion window
    pygame.draw.rect(screen, PROMOTION_BG, (window_x, window_y, window_width, window_height))

    # Display the choices
    for i, piece in enumerate(game_state['promotion_choices']):
        piece_to_draw = piece.upper() if color == 1 else piece.lower()
        screen.blit(PIECES[piece_to_draw], (window_x + i * SQUARE_SIZE, window_y))

def make_move(board, move, game_state):
    from_x, from_y, to_x, to_y = move
    piece = board[from_y][from_x]
    board[to_y][to_x] = piece
    board[from_y][from_x] = '.'
    piece_moved = board[to_y][to_x]

    # Roque
    if piece_moved == 'K' and from_x == 4 and from_y == 7:
        if to_x == 6:  # Roque côté roi
            board[7][5] = 'R'
            board[7][7] = '.'
        elif to_x == 2:  # Roque côté dame
            board[7][3] = 'R'
            board[7][0] = '.'
        game_state['white_can_castle_kingside'] = False
        game_state['white_can_castle_queenside'] = False
    elif piece_moved == 'k' and from_x == 4 and from_y == 0:
        if to_x == 6:  # Roque côté roi
            board[0][5] = 'r'
            board[0][7] = '.'
        elif to_x == 2:  # Roque côté dame
            board[0][3] = 'r'
            board[0][0] = '.'
        game_state['black_can_castle_kingside'] = False
        game_state['black_can_castle_queenside'] = False
    
    # Prise en passant
    if piece_moved.lower() == 'p':
        if (to_x, to_y) in game_state['en_passant_target']:
            if piece_moved.isupper():  # Blanc
                board[to_y + 1][to_x] = '.'
            else:  # Noir
                board[to_y - 1][to_x] = '.'
    
    # Mise à jour de l'état du roque
    if piece_moved == 'K':
        game_state['white_can_castle_kingside'] = False
        game_state['white_can_castle_queenside'] = False
    elif piece_moved == 'k':
        game_state['black_can_castle_kingside'] = False
        game_state['black_can_castle_queenside'] = False
    elif piece_moved == 'R' and from_x == 7 and from_y == 7:
        game_state['white_can_castle_kingside'] = False
    elif piece_moved == 'R' and from_x == 0 and from_y == 7:
        game_state['white_can_castle_queenside'] = False
    elif piece_moved == 'r' and from_x == 7 and from_y == 0:
        game_state['black_can_castle_kingside'] = False
    elif piece_moved == 'r' and from_x == 0 and from_y == 0:
        game_state['black_can_castle_queenside'] = False
    
    # Promotion
    if piece_moved == 'P' and to_y == 0:
        game_state['promotion_position'] = (to_x, to_y, 1)  # 1 pour blanc
    elif piece_moved == 'p' and to_y == 7:
        game_state['promotion_position'] = (to_x, to_y, -1)  # -1 pour noir
    
    # Mise à jour de la cible de prise en passant
    if piece_moved.lower() == 'p' and abs(from_y - to_y) == 2:
        game_state['en_passant_target'].append((to_x, (from_y + to_y) // 2))

def find_king_position(board, is_white):
    """Trouve la position du roi sur le plateau"""
    king = 'K' if is_white else 'k'
    for y in range(8):
        for x in range(8):
            if board[y][x] == king:
                return (x, y)
    return None

def is_check(board, player_turn, game_state):
    """Vérifie si le roi du joueur actuel est en échec"""
    king_pos = find_king_position(board, player_turn)
    if not king_pos:
        return False
    
    kx, ky = king_pos
    
    # Vérifier toutes les pièces adverses
    for y in range(8):
        for x in range(8):
            piece = board[y][x]
            if piece != '.' and piece.isupper() != (player_turn == game_state['player_turn']):
                moves = get_piece_moves(board, x, y, game_state, True)
                if (kx, ky) in moves:
                    return True
    return False

def is_game_over(board):
    white_king = black_king = False
    for row in board:
        for piece in row:
            if piece == 'K':
                white_king = True
            elif piece == 'k':
                black_king = True
    return not (white_king and black_king)

def set_ai_error_rate():
    global AI_ERROR_RATE
    while True:
        try:
            rate = float(input("Entrez le taux d'erreur de l'IA (0.0 à 1.0): "))
            if 0.0 <= rate <= 1.0:
                AI_ERROR_RATE = rate
                print(f"Taux d'erreur de l'IA défini à {rate}")
                break
            else:
                print("Le taux doit être entre 0.0 et 1.0")
        except ValueError:
            print("Veuillez entrer un nombre valide")


def play_game():
    # Initialisation de l'état du jeu
    board = game_state['board']
    selected = game_state['selected']
    player_turn = game_state['player_turn']
    promotion_position = game_state['promotion_position']

    running = True
    waiting_for_promotion = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.MOUSEBUTTONUP:  # Changé de MOUSEBUTTONDOWN à MOUSEBUTTONUP
                x, y = get_square_under_mouse()
                
                # Gestion de la promotion
                if waiting_for_promotion:
                    promo_x, promo_y, color = game_state['promotion_position']
                    window_width = 4 * SQUARE_SIZE
                    window_x = promo_x * SQUARE_SIZE - (window_width - SQUARE_SIZE) // 2
                    if y == promo_y and window_x <= x * SQUARE_SIZE < window_x + window_width:
                        piece_index = (x * SQUARE_SIZE - window_x) // SQUARE_SIZE
                        chosen_piece = game_state['promotion_choices'][piece_index]
                        
                        # Appliquer la promotion
                        piece_to_promote = chosen_piece.upper() if color == 1 else chosen_piece.lower()
                        handle_promotion(board, promo_x, promo_y, piece_to_promote)
                        game_state['player_turn'] = not game_state['player_turn']
                        waiting_for_promotion = False
                elif player_turn:
                    if selected is None:
                        # Sélectionner une pièce si c'est une pièce blanche
                        if board[y][x].isupper():
                            selected = (x, y)
                    else:
                        move = (selected[0], selected[1], x, y)
                        # Valider le mouvement
                        possible_moves = get_piece_moves(board, selected[0], selected[1], game_state)
                        if (x, y) in possible_moves:
                            make_move(board, move, game_state)
                            player_turn = False  # Passer au tour de l'IA
                        selected = None  # Désélectionner

        # Dessiner l'échiquier et les pièces
        screen.fill(LIGHT_GREEN)
        draw_board()
        draw_pieces(game_state['board'])
        
        # Vérification et affichage de l'échec
        if is_check(game_state['board'], game_state['player_turn'], game_state):
            king_pos = find_king_position(game_state['board'], game_state['player_turn'])
            if king_pos:
                x, y = king_pos
                pygame.draw.rect(screen, (255, 0, 0), 
                               (x * SQUARE_SIZE, y * SQUARE_SIZE, 
                                SQUARE_SIZE, SQUARE_SIZE), 3)
        
        # Mettre en évidence la case sélectionnée
        if selected:
            highlight_square(selected[0], selected[1])
            for move in get_piece_moves(board, selected[0], selected[1], game_state):
                highlight_square(move[0], move[1])

        # Dessiner la fenêtre de promotion si nécessaire
        if game_state['promotion_position']:
            waiting_for_promotion = True
            promo_x, promo_y, color = game_state['promotion_position']
            draw_promotion_window(promo_x, promo_y, color)

        pygame.display.flip()

        # Tour de l'IA (Noir)
        if not player_turn and not waiting_for_promotion:
            ai_move_eval, ai_move = minimax(board, depth_ai_raisonment, False, game_state)  # False pour les noirs
            if ai_move:
                make_move(board, ai_move, game_state)
            player_turn = True  # Retour au tour du joueur
        
        if is_game_over(board):
            print("Game Over!")
            running = False
            break

        # Mise à jour de l'état du jeu global
        game_state['board'] = board
        game_state['selected'] = selected
        game_state['player_turn'] = player_turn
        game_state['promotion_position'] = promotion_position
        

    sleep(5)
    pygame.quit()
    sys.exit()

# Appelez cette fonction avant de commencer le jeu
if __name__ == "__main__":
    set_ai_error_rate()
    play_game()
