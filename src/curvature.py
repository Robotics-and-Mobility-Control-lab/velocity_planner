import math
import csv
import numpy as np
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

file_choice = "path1_result"

x_data = []
y_data = []

if file_choice == "path1_result":
    with open("/home/ej/kiapi_plan_ws/src/kiapi_pkg/path_data/path1_result.txt", 'r') as file:
        for line in file:
            values = line.split()
            x_data.append(float(values[0]))
            y_data.append(float(values[1]))

else:
    raise ValueError("Invalid file choice.")


distances = [0.0]
for i in range(1, len(x_data)):
    dx = x_data[i] - x_data[i-1]
    dy = y_data[i] - y_data[i-1]
    distance = np.sqrt(dx**2 + dy**2)
    distances.append(distances[-1] + distance)

distances = np.array(distances)
interp_indices = np.arange(0, max(distances), 0.01)

x_spline = splrep(distances, x_data, s=3, k=5)
y_spline = splrep(distances, y_data, s=3, k=5)

x_fit = splev(interp_indices, x_spline)
y_fit = splev(interp_indices, y_spline)
x_derivative = splev(interp_indices, x_spline, der=1)
y_derivative = splev(interp_indices, y_spline, der=1)
x_second_derivative = splev(interp_indices, x_spline, der=2)
y_second_derivative = splev(interp_indices, y_spline, der=2)

curvature = -(x_second_derivative * y_derivative - y_second_derivative * x_derivative) / (y_derivative**2 + x_derivative**2)**1.5
print(float(curvature[-1]))

for value in curvature:
    print(float(value))


fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(3,2,width_ratios=[1,1], height_ratios=[1,1,1])

ax1 = plt.subplot(gs[0:3, 0])
ax1.scatter(x_data, y_data, c='blue', s=10, label='Original Data')
ax1.plot(x_fit, y_fit, 'r--', label='Fitted Curve')
ax1.set_title('Original Data vs Fitted Curve')
ax1.set_xlabel('X [m]')
ax1.set_ylabel('Y [m]')
ax1.legend()
ax1.grid()
ax1.set_aspect('equal')  # X축과 Y축 비율을 동일하게 유지

ax2 = plt.subplot(gs[1,1])
ax2.plot(interp_indices, curvature, 'g-')
ax2.set_title('Curvature')
ax2.set_xlabel("Distance [m]")
ax2.set_ylabel("Curvature [1/m]")
ax2.grid()


plt.tight_layout()
plt.show()