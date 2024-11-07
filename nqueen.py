def solveNQueens(n: int, first_queen_col: int):
    col = set()
    posDiag = set()
    negDiag = set()

    res = []
    board = [["."] * n for _ in range(n)]

    def backtrack(r):
        if r == n:
            res.append(["".join(row) for row in board])
            return

        for c in range(n):
            if c in col or (r + c) in posDiag or (r - c) in negDiag:
                continue

            col.add(c)
            posDiag.add(r + c)
            negDiag.add(r - c)
            board[r][c] = "Q"

            backtrack(r + 1)

            col.remove(c)
            posDiag.remove(r + c)
            negDiag.remove(r - c)
            board[r][c] = "."

    # Place the first queen in the specified column of the first row
    col.add(first_queen_col)
    posDiag.add(0 + first_queen_col)
    negDiag.add(0 - first_queen_col)
    board[0][first_queen_col] = "Q"

    backtrack(1)  # Start with the second row
    return res

# Get input from the user
if __name__ == "__main__":
    n = int(input("Enter the size of the board (n): "))
    first_queen_col = int(input(f"Enter the column (0 to {n-1}) to place the first queen in the first row: "))

    # Validate the input for first_queen_col
    if first_queen_col < 0 or first_queen_col >= n:
        print("Error: The column must be within the board size.")
    else:
        solutions = solveNQueens(n, first_queen_col)
        if solutions:
            print("One of the possible solutions:")
            for row in solutions[0]:
                print(" ".join(row))
        else:
            print("No solutions found.")
