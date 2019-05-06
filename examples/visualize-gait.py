import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

def plot_gait_params(gait_params):
	data = [[  -1,  -1, -1,  -1,   -1,   -1],
			[   1,  95, 50, 110, 0.55, 0.35],
			[1.32, 125, 80, 160, 0.65, 0.45]]

	rows = ('Value', 'Min Accept', 'Max Accept')
	columns = ('Gait Circle', 'Gait Frequency', 'Gait Length', 'Gait Speed', 'Gait Support', 'Gait Swing')
	data[0][0] = gait_params['fore_param'][0,0]['gait_circle'][0,0]
	data[0][1] = gait_params['fore_param'][0,0]['gait_frequency'][0,0]
	data[0][2] = gait_params['fore_param'][0,0]['gait_length'][0,0]
	data[0][3] = gait_params['fore_param'][0,0]['gait_speed'][0,0]
	data[0][4] = gait_params['fore_param'][0,0]['gait_support'][0,0]
	data[0][5] = gait_params['fore_param'][0,0]['gait_swing'][0,0]

	plt.figure(figsize=[15, 9])
	plt.subplots_adjust(0.06, 0.05, 0.94, 0.95, 0.2, 0.4)
	gs = gridspec.GridSpec(3, 4)

	plt.subplot(gs[0,0:3])
	the_table = plt.table(cellText=data,
						rowLabels=rows,
						colLabels=columns,
						colWidths=[0.125]*6,
						loc='center',
						cellLoc='center')
	the_table.auto_set_font_size(False)
	the_table.set_fontsize(11)
	the_table.scale(1, 1)
	table_cells = the_table.properties()['child_artists']
	for cell in table_cells: cell.set_height(0.2)
	plt.axis('off')
	plt.draw()

	time = gait_params['fore_param'][0,0]['gait_hip'][0,0]['angle_frameTime_sequence'][0]
	angle = gait_params['fore_param'][0,0]['gait_hip'][0,0]['angle_sequence'][0]
	plt.subplot(gs[1,0])
	plt.title('Left Hip')
	plt.plot(time, angle)
	plt.xlabel('Time(ms)')
	plt.ylabel('Angle(degree)')
	plt.axis('on')
	plt.draw()

	time = gait_params['fore_param'][0,0]['gait_knee'][0,0]['angle_frameTime_sequence'][0]
	angle = gait_params['fore_param'][0,0]['gait_knee'][0,0]['angle_sequence'][0]
	plt.subplot(gs[1,1])
	plt.title('Left Knee')
	plt.plot(time, angle)
	plt.xlabel('Time(ms)')
	plt.ylabel('Angle(degree)')
	plt.axis('on')
	plt.draw()

	time = gait_params['fore_param'][0,0]['gait_ankle'][0,0]['angle_frameTime_sequence'][0]
	angle = gait_params['fore_param'][0,0]['gait_ankle'][0,0]['angle_sequence'][0]
	plt.subplot(gs[1,2])
	plt.title('Left Ankle')
	plt.plot(time, angle)
	plt.xlabel('Time(ms)')
	plt.ylabel('Angle(degree)')
	plt.axis('on')
	plt.draw()

	time = gait_params['back_param'][0,0]['gait_hip'][0,0]['angle_frameTime_sequence'][0]
	angle = gait_params['back_param'][0,0]['gait_hip'][0,0]['angle_sequence'][0]
	plt.subplot(gs[2,0])
	plt.title('Right Hip')
	plt.plot(time, angle)
	plt.xlabel('Time(ms)')
	plt.ylabel('Angle(degree)')
	plt.axis('on')
	plt.draw()

	time = gait_params['back_param'][0,0]['gait_knee'][0,0]['angle_frameTime_sequence'][0]
	angle = gait_params['back_param'][0,0]['gait_knee'][0,0]['angle_sequence'][0]
	plt.subplot(gs[2,1])
	plt.title('Right Knee')
	plt.plot(time, angle)
	plt.xlabel('Time(ms)')
	plt.ylabel('Angle(degree)')
	plt.axis('on')
	plt.draw()

	time = gait_params['back_param'][0,0]['gait_ankle'][0,0]['angle_frameTime_sequence'][0]
	angle = gait_params['back_param'][0,0]['gait_ankle'][0,0]['angle_sequence'][0]
	plt.subplot(gs[2,2])
	plt.title('Right Ankle')
	plt.plot(time, angle)
	plt.xlabel('Time(ms)')
	plt.ylabel('Angle(degree)')
	plt.axis('on')
	plt.draw()

	plt.show()

if __name__ == '__main__':
	gait_param_path = 'f:/Lab/dataset/Gait-Database/2019-03-25_22-45/gaitparams.mat'
	gait_params = sio.loadmat(gait_param_path)
	plot_gait_params(gait_params)