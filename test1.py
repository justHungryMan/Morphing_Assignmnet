import numpy as np
import math
import pyvista as pv
import tree as T
import assignment as AS
from tqdm import tqdm_notebook

from scipy.optimize import linear_sum_assignment

from pyvista import examples
from operator import itemgetter

dataset_teapot = examples.download_teapot()
dataset_bunny = examples.download_bunny_coarse()

teapot_points, temp = T.generate_points(dataset_teapot)
bunny_points, temp = T.generate_points(dataset_bunny)

source = teapot_points
destination = bunny_points


Morphing = AS.Assignment(source, destination)
Morphing.calculate()
destination_points, source_points = Morphing.get_result()


# Test

FRAME = 6000
filename = "test.mp4"


# Frame Image
start_dataset = pv.PolyData(np.array(source_points))


source_dataset = pv.PolyData(np.array(source_points))
#source_dataset.plot(show_edges = True)


destination_dataset = pv.PolyData(np.array(destination_points))
#destination_dataset.plot(show_edges = True)

plotter = pv.Plotter()
plotter.open_movie(filename)
plotter.add_mesh(start_dataset, color='red')

plotter.show(auto_close = False)
plotter.write_frame()

for i in range(FRAME):
    start_dataset.points = destination_dataset.points * i / FRAME + source_dataset.points * (FRAME - i) / FRAME
    plotter.write_frame()

# Test
