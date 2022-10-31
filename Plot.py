import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("/Users/susanamunguia/PycharmProjects/our_first_project/Lorenz_x_new", index_col=0)
t_new = df['t_new']

x_original = df['x_original']
y_original = df['y_original']
z_original = df['z_original']

y_pred_x = df['y_pred_x']
y_pred_y = df['y_pred_y']
y_pred_z = df['y_pred_z']

y_pred_new_x = df['y_pred_new_x']
y_pred_new_y = df['y_pred_new_y']
y_pred_new_z = df['y_pred_new_z']

#plt.plot(t_new, x_original, color='blue', label='True Dynamics')
#plt.plot(t_new, y_pred_x, color='red', linestyle='dashed', label='Predicted Dynamics-NN')
#plt.plot(t_new, y_pred_new_x, color='orange', linestyle='dotted', label='Predicted Dynamics-using new ouput as new input data')

#plt.plot(t_new, y_original, color='blue', label='True Dynamics')
#plt.plot(t_new, y_pred_y, color='red', linestyle='dashed', label='Predicted Dynamics-NN')
#plt.plot(t_new, y_pred_new_y, color='orange', linestyle='dotted', label='Predicted Dynamics-using new ouput as new input data')

plt.plot(t_new, z_original, color='blue', label='True Dynamics')
plt.plot(t_new, y_pred_z, color='red', linestyle='dashed', label='Predicted Dynamics-NN')
plt.plot(t_new, y_pred_new_z, color='orange', linestyle='dotted', label='Predicted Dynamics-using new ouput as new input data')


# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Time")
plt.ylabel("z(t)")
plt.title("Lorenz system, z(t)")

# Adding legend, which helps us recognize the curve according to it's color
plt.legend()

# To load the display window
plt.show()



