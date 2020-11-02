from eddies_detection import  get_traj_with_parcels, get_traj_with_scipy, find_eddies

from plot_tools import StreamPlot

date = 0
stream_data_fname = "../data/data.nc"
runtime = 400
delta_time = 5
particle_grid_step = 2


figure = StreamPlot()
a = get_traj_with_parcels(date, runtime, delta_time, particle_grid_step, stream_data_fname)
figure.trajectories_from_list([sl.coord_list for sl in a],line_style='k')
figure.show()
                    
figure = StreamPlot()
b = get_traj_with_scipy(date, runtime, delta_time, particle_grid_step, stream_data_fname)
figure.plot_trajectories([sl.coord_list for sl in b],line_style='c')

bb = find_eddies(b)
for eddy in bb:
    figure.plot_trajectories(eddy.sl_list)
figure.plot_eddies(bb)

figure.show()