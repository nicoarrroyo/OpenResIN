import numpy as np

x1 = np.array([[1, 2, 3], 
               [3, 4, 5], 
               [5, 6, 7]])

x2 = np.array([[5, 6, 7], 
               [7, 8, 9], 
               [9, np.nan, 11]])

x_stack = np.stack((x1, x2))
x_mean = np.nanmean(x_stack, axis=0)
print(x_mean)

x_list = []
x_list.append(x1)
x_list.append(x2)
x_list_stack = np.stack(x_list)
x_list_mean = np.nanmean(x_list_stack, axis=0)
