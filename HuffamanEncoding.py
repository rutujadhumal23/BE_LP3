import heapq


class Node:
    def __init__(self, freq, symbol, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right
        self.huff = ""

    def __lt__(self, other):
        return self.freq < other.freq


def printNodes(node, val=""):
    newval = val + node.huff
    if node.left:
        printNodes(node.left, newval)
    if node.right:
        printNodes(node.right, newval)
    else:
        print(f"{node.symbol} -> {newval}")


# Get input from the user
chars = input("Enter the characters (no spaces): ")
freqs_input = input("Enter their frequencies separated by spaces: ").split()

# Convert frequencies to integers and check if they match the length of characters
try:
    freqs = list(map(int, freqs_input))
    if len(chars) != len(freqs):
        print("Error: The number of characters and frequencies must match.")
    else:
        nodes = []

        # Create a node for each character and push it to the heap
        for i in range(len(chars)):
            heapq.heappush(nodes, Node(freqs[i], chars[i]))

        # Build the Huffman Tree
        while len(nodes) > 1:
            left = heapq.heappop(nodes)
            right = heapq.heappop(nodes)
            left.huff = "0"
            right.huff = "1"
            newnode = Node(left.freq + right.freq, left.symbol + right.symbol, left, right)
            heapq.heappush(nodes, newnode)

        # Print the Huffman Codes
        print("Huffman Codes for the input characters are:")
        printNodes(nodes[0])

except ValueError:
    print("Error: Frequencies must be integers.")
