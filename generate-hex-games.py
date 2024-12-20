import argparse
import random
from collections import deque

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-dim", type=int, default=3, help="Dimension of the Hex board (N for an N×N board)")
    parser.add_argument("--num-games", type=int, default=10, help="Number of games to generate")
    return parser.parse_args()

def neighbors(r, c, n):
    # Hex board neighbors (6 directions)
    # Directions on a hex grid (using "offset" coordinates):
    # Up-Left, Up-Right, Left, Right, Down-Left, Down-Right
    dirs = [(-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0)]
    for dr, dc in dirs:
        nr, nc = r+dr, c+dc
        if 0 <= nr < n and 0 <= nc < n:
            yield nr, nc

def check_winner(board, player, n):
    # player: 0 for 'O', 1 for 'X'
    # 'X' tries to connect top to bottom
    # 'O' tries to connect left to right
    # We'll run a BFS/DFS from starting nodes on one side to see if we can reach the opposite side.

    if player == 1:
        # Check if 'X' (player 1) connects top to bottom
        # Start from top row cells that are 'X'
        start_positions = [(0, c) for c in range(n) if board[0][c] == 'X']
        target_row = n - 1
        visited = [[False]*n for _ in range(n)]
        queue = deque(start_positions)
        for r, c in start_positions:
            visited[r][c] = True

        while queue:
            r, c = queue.popleft()
            if r == target_row:
                return True
            for nr, nc in neighbors(r, c, n):
                if board[nr][nc] == 'X' and not visited[nr][nc]:
                    visited[nr][nc] = True
                    queue.append((nr, nc))
        return False
    else:
        # Check if 'O' (player 0) connects left to right
        start_positions = [(r, 0) for r in range(n) if board[r][0] == 'O']
        target_col = n - 1
        visited = [[False]*n for _ in range(n)]
        queue = deque(start_positions)
        for r, c in start_positions:
            visited[r][c] = True

        while queue:
            r, c = queue.popleft()
            if c == target_col:
                return True
            for nr, nc in neighbors(r, c, n):
                if board[nr][nc] == 'O' and not visited[nr][nc]:
                    visited[nr][nc] = True
                    queue.append((nr, nc))
        return False

def simulate_game(n):
    # Represent board as a 2D array of chars: ' ', 'X', 'O'
    board = [[' ' for _ in range(n)] for _ in range(n)]
    empty_positions = [(r, c) for r in range(n) for c in range(n)]
    random.shuffle(empty_positions)

    # Players: 0 -> 'O', 1 -> 'X'
    # 'X' tries to connect top-bottom, 'O' tries to connect left-right
    current_player = 1  # start with X (as in many examples)
    # If you want O to start, set current_player = 0
    
    for pos in empty_positions:
        r, c = pos
        board[r][c] = 'X' if current_player == 1 else 'O'
        # Check if current player won
        if check_winner(board, current_player, n):
            # Return board and winner
            winner = 1 if current_player == 1 else 0
            return board, winner
        current_player = 1 - current_player
    
    # Theoretically Hex can't end in a draw if the board is filled.
    # But if something goes wrong, we return None
    return None, None

def board_to_string(board):
    # Flatten board rows into a single string
    return ''.join(''.join(row) for row in board)

def main():
    args = parse_args()
    n = args.board_dim
    num_games = args.num_games

    # Print header
    print("board,winner")

    games_generated = 0
    while games_generated < num_games:
        final_board, winner = simulate_game(n)
        if final_board is not None:
            # Convert to string
            board_str = board_to_string(final_board)
            # print line: "board,winner"
            print(f"{board_str},{winner}")
            games_generated += 1

if __name__ == "__main__":
    main()

