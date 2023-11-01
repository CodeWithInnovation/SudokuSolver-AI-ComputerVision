def solve(puzzle):     
    # Find an empty cell
    empty = find_empty(puzzle)
    
    # If there are no empty cells, the puzzle is solved
    if not empty:
        return True
    
    row, col = empty

    # Try values from 1 to 9 in the empty cell
    for num in range(1, 10):
        if is_valid(puzzle, num, (row, col)):
            puzzle[row][col] = num

            # Recursively attempt to solve the puzzle
            if solve(puzzle):
                return True

            # If the current configuration leads to an invalid solution, backtrack
            puzzle[row][col] = 0

    # If no value works, return False to trigger backtracking
    return False

def find_empty(puzzle):
    for i in range(9):
        for j in range(9):
            if puzzle[i][j] == 0:
                return (i, j)
    return None

def is_valid(puzzle, num, pos):
    # Check row
    for i in range(9):
        if puzzle[pos[0]][i] == num:
            return False

    # Check column
    for i in range(9):
        if puzzle[i][pos[1]] == num:
            return False

    # Check 3x3 box
    box_x = pos[1] // 3
    box_y = pos[0] // 3

    for i in range(box_y * 3, box_y * 3 + 3):
        for j in range(box_x * 3, box_x * 3 + 3):
            if puzzle[i][j] == num:
                return False

    return True