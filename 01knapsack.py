def knapsack_01(n, values, weights, W):
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                dp[i][w] = 0
            elif weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]

    selected_items = []
    i, w = n, W
    while i > 0 and w > 0:
        if dp[i][w] != dp[i - 1][w]:
            selected_items.append(i - 1)
            w -= weights[i - 1]
        i -= 1

    return dp[n][W], selected_items

# Get input from the user
if __name__ == "__main__":
    n = int(input("Enter the number of items: "))
    values = []
    weights = []

    for i in range(n):
        value = int(input(f"Enter the value of item {i + 1}: "))
        weight = int(input(f"Enter the weight of item {i + 1}: "))
        values.append(value)
        weights.append(weight)

    W = int(input("Enter the maximum capacity of the knapsack: "))

    # Validate the input lengths
    if len(values) != n or len(weights) != n:
        print("Error: The number of values and weights must match the number of items.")
    else:
        max_value, selected_items = knapsack_01(n, values, weights, W)
        print("Maximum value:", max_value)
        print("Selected items (0-indexed):", selected_items)
