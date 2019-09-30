import numpy as np
import math
import pyvista as pv
import tree as T
import assignment as AS
import time
import pickle
from tqdm import tqdm_notebook
from pathlib import Path

from scipy.optimize import linear_sum_assignment

from pyvista import examples
from operator import itemgetter

dataset_teapot = examples.download_teapot()
dataset_bunny = examples.download_bunny_coarse()

teapot_points, temp = T.generate_points(dataset_teapot)
bunny_points, temp = T.generate_points(dataset_bunny)

source = teapot_points
destination = bunny_points * 10




source_points, destination_points = [], []
src_pts_path = Path("source_pts.pkl")
dst_pts_path = Path("dst_pts.pkl")

if src_pts_path.exists():
    with open(src_pts_path, "rb") as fp:
        source_points = pickle.load(fp)
    with open(dst_pts_path, "rb") as fp:
        destination_points = pickle.load(fp)

else :
    start = time.time()
    Morphing = AS.Assignment(source, destination)
    Morphing.calculate()
    source_points, destination_points = Morphing.get_result()
    print("time : ", time.time() - start)


if not src_pts_path.exists():
    with open(src_pts_path, "wb") as fp:
        pickle.dump(source_points, fp)

    with open("dst_pts.pkl", "wb") as fp:
        pickle.dump(destination_points, fp)



# Test

FRAME = 240
filename = "test.mp4"


# Frame Image
start_dataset = pv.PolyData(np.array(source_points))


source_dataset = pv.PolyData(np.array(source_points))
#source_dataset.plot(show_edges = True)


destination_dataset = pv.PolyData(np.array(destination_points))
#destination_dataset.plot(show_edges = True)

plotter = pv.Plotter()
plotter.open_movie(filename)
plotter.set_position([-1, 2, -5])
plotter.enable_eye_dome_lighting()
plotter.add_mesh(start_dataset, color='red', show_edges = True)

plotter.show(auto_close = False)
plotter.write_frame()

for i in range(FRAME):
    start_dataset.points = destination_dataset.points * i / FRAME + source_dataset.points * (FRAME - i) / FRAME
    plotter.write_frame()

# Test
