import numpy as np
from multiprocessing import Pool
from PIL import Image

# Load image and convert to grayscale
def load_image(path):
    img = Image.open(path).convert('L')  # 'L' mode = grayscale
    return np.array(img)

# Edge detection rule for a single pixel
def detect_edge(args):
    grid, x, y, threshold = args
    rows, cols = grid.shape
    center = grid[x][y]
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if abs(int(center) - int(grid[nx][ny])) > threshold:
                    return 255  # Edge
    return 0  # Non-edge

# Parallel cellular edge detection
def parallel_edge_detection(image, threshold=20):
    rows, cols = image.shape
    args = [(image, x, y, threshold) for x in range(rows) for y in range(cols)]
    with Pool() as pool:
        edges = pool.map(detect_edge, args)
    return np.array(edges).reshape((rows, cols))

# Save or display result
def save_edge_image(edge_array, output_path='Lab7\\edges_output_img.png'):
    edge_img = Image.fromarray(edge_array.astype(np.uint8))
    edge_img.save(output_path)
    edge_img.show()

# Example usage
if __name__ == '__main__':
    image = load_image('Lab7\\image.png')  # Replace with actual image path
    edges = parallel_edge_detection(image, threshold=30)
    save_edge_image(edges)