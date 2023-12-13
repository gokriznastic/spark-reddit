import matplotlib.pyplot as plt

# Data for preprocessing time
pct_values = [1, 5, 10, 25, 50, 75, 100]

local1_preprocessing = [32, 187, 340, 552, 1542, 2353, 3246]
local20_preprocessing = [11, 19, 27, 52, 93, 132, 171]

# Data for training time
local1_training = [52, 323, 626, 1098, 3033, 4648, 6582]
local20_training = [6, 21, 36, 84, 165, 245, 324]

# Data for total execution time
local1_total = [123, 688, 1300, 2217, 6114, 9353, 12572]
local20_total = [33, 70, 105, 219, 405, 589, 775]

# Plotting preprocessing time
plt.figure(figsize=(10, 6))
plt.plot(pct_values, local1_preprocessing, marker='o', linestyle='-', label='Non-parellal (Preprocessing)', color='blue')
plt.plot(pct_values, local20_preprocessing, marker='o', linestyle='-', label='Parellal (Preprocessing)', color='orange')

# Plotting training time
plt.plot(pct_values, local1_training, marker='o', linestyle='-', label='Non-parellal (Training)', color='green')
plt.plot(pct_values, local20_training, marker='o', linestyle='-', label='Parellal (Training)', color='red')

# Plotting total execution time
plt.plot(pct_values, local1_total, marker='o', linestyle='-', label='Non-parellal (Total)', color='purple')
plt.plot(pct_values, local20_total, marker='o', linestyle='-', label='Parellal (Total)', color='brown')

# Adding labels and title
plt.xlabel('Size (in percentage) of data')
plt.ylabel('Time (s)')
plt.title('Execution Time vs Percentage of Dataset')
plt.legend()
plt.grid(True)
plt.savefig("parallelism_total.png")
