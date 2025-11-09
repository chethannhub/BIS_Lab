import numpy as np
from multiprocessing import Pool

def edge_rule(subgrid):
    out = np.zeros_like(subgrid)
    rows, cols = subgrid.shape
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            gx = (
                subgrid[i-1, j+1] + 2*subgrid[i, j+1] + subgrid[i+1, j+1] -
                subgrid[i-1, j-1] - 2*subgrid[i, j-1] - subgrid[i+1, j-1]
            )
            gy = (
                subgrid[i-1, j-1] + 2*subgrid[i-1, j] + subgrid[i-1, j+1] -
                subgrid[i+1, j-1] - 2*subgrid[i+1, j] - subgrid[i+1, j+1]
            )
            
            grad = np.sqrt(gx**2 + gy**2)
            out[i, j] = 255 if grad > 13 else 0

    return out

def get_chunks_with_overlap(img, num_workers):
    height = img.shape[0]
    chunk_size = height // num_workers
    chunks = []
    for i in range(num_workers):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_workers - 1 else height

        if i > 0:
            start -= 1
        if i < num_workers - 1:
            end += 1

        chunks.append(img[start:end, :])
    return chunks

def parallel_edge_detection(img, num_workers=4):
    chunks = get_chunks_with_overlap(img, num_workers)
    with Pool(num_workers) as p:
        processed_chunks = p.map(edge_rule, chunks)

   
    results = []
    for i, chunk in enumerate(processed_chunks):
        if i > 0:
            chunk = chunk[1:] 
        if i < num_workers - 1:
            chunk = chunk[:-1]  
        results.append(chunk)

    return np.vstack(results)


if __name__ == "__main__":
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    
    img = imageio.imread('Lab7\\image.png', mode='L')
    
    edges = parallel_edge_detection(img)
    
    plt.imshow(edges, cmap='gray')
    plt.show()