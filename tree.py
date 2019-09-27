import numpy as np
import pyvista as pv
from pyvista import examples
from operator import itemgetter

#mesh = examples.download_teapot()
#mesh.plot(cpos=[-1, 2, -5], show_edges=True)

# Configuration
'''
N = 3
GRID_SIZE = N * N * N
'''
# Extract points, bounds

# Define some helpers - ignore these and use your own data!
'''
dataset_teapot = examples.download_teapot()
dataset_bunny = examples.download_bunny_coarse()
'''
def generate_points(dataset = examples.download_teapot(), subset=1):
    """A helper to make a 3D NumPy array of points (n_points by 3)"""
    ids = np.random.randint(low=0, high=dataset.n_points-1,
                            size=int(dataset.n_points * subset))

    bounds = dataset.bounds # x, y, z
    center = dataset.center

    points = dataset.points[ids]

    # Center Align
    center = np.array(center).reshape(-1, 3)
    points = points - center

    # Bounds Align
    bounds[0] = bounds[0] - center[0][0]
    bounds[1] = bounds[1] - center[0][0]
    bounds[2] = bounds[2] - center[0][1]
    bounds[3] = bounds[3] - center[0][1]
    bounds[4] = bounds[4] - center[0][2]
    bounds[5] = bounds[5] - center[0][2]
    return points, bounds



# 함수화 작업 필요함.
def generate_grid(points, bounds, N = 3):
    grid = np.ndarray((N,N,N), dtype = dict).tolist()
    grid_bounds = np.ndarray((N,N,N), dtype = list).tolist()
    for i in range(N):
        xAxis = (bounds[1] - bounds[0]) / N * (i + 1) + bounds[0]
        for j in range(N):
            yAxis = (bounds[3] - bounds[2]) / N * (j + 1) + bounds[2]
            for k in range(N):
                zAxis = (bounds[5] - bounds[4]) / N * (k + 1) + bounds[4]
                grid_bounds[i][j][k] = [xAxis - (bounds[1] - bounds[0]) / N, xAxis, yAxis - (bounds[3] - bounds[2]) / N, yAxis, zAxis - (bounds[5] - bounds[4]) / N, zAxis]

    for point in points:
        for i in range(N): # x
            for j in range(N): # y  
                for k in range(N): # z   
                    
                    # Grid initialize
                    if grid[i][j][k] is None:
                            grid[i][j][k] = {
                                "element": [],
                                "child": [None] * 2,
                            }
                    if point[0] >= grid_bounds[i][j][k][0] and point[0] < grid_bounds[i][j][k][1] and point[1] >= grid_bounds[i][j][k][2] and point[1] < grid_bounds[i][j][k][3] and point[2] >= grid_bounds[i][j][k][4] and point[2] < grid_bounds[i][j][k][5]:
                        grid[i][j][k]["element"].append(point)
    return grid, grid_bounds


# Test
'''
teapot_points, teapot_bounds = generate_points(dataset_teapot)
bunny_points, bunny_bounds = generate_points(dataset_bunny)

teapot_grid, teapot_grid_bounds = generate_grid(teapot_points, teapot_bounds, N)
bunny_grid, bunny_grid_bounds = generate_grid(bunny_points, bunny_bounds, N)
'''
def geneate_grid_clustering(source_grid, destination_grid, N = 3):
    for i in range(N):
        for j in range(N):
            for k in range(N):
                sorted_clustering(source_grid[i][j][k], destination_grid[i][j][k])
                

def sorted_clustering(source_grid, destination_grid):
    # Grid 에 아무것도 없을 때 (0:N or N:0 matching)
    if (not source_grid["element"]) ^ (not destination_grid["element"]):
        print("Error : N의 값을 낮춰야한다.")
        raise NotImplementedError
    source_grid["element"] = sorted(source_grid["element"], key = itemgetter(0, 1, 2))
    generate_tree(source_grid)
    destination_grid["element"] = sorted(destination_grid["element"], key = itemgetter(0, 1, 2))
    generate_tree(destination_grid)

def generate_tree(tree):
    if len(tree["element"]) is not 1 and tree is not None:
        tree["child"][0] = {
            "element": None,
            "child": [None] * 2
        }
        tree["child"][1] = {
            "element": None,
            "child": [None] * 2
        }
        tree["child"][0]["element"] = tree["element"][: len(tree["element"]) // 2]
        tree["child"][1]["element"] = tree["element"][len(tree["element"]) // 2 :]
        tree["element"] = None
    
        generate_tree(tree["child"][0])
        generate_tree(tree["child"][1])

