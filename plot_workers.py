import matplotlib.pyplot as plt

# Function to convert hours, minutes, and seconds to seconds
def convert_to_seconds(hours, minutes, seconds):
    return hours * 3600 + minutes * 60 + seconds

# Data for preprocessing time
workers = [1, 5, 10, 20, 40, 100]

preprocessing_time = [convert_to_seconds(0, 9, 33),
                      convert_to_seconds(0, 2, 32),
                      convert_to_seconds(0, 1, 25),
                      convert_to_seconds(0, 0, 55),
                      convert_to_seconds(0, 0, 55),
                      convert_to_seconds(0, 1, 6)]

# Data for training time
training_time = [convert_to_seconds(0, 19, 13),
                 convert_to_seconds(0, 5, 2),
                 convert_to_seconds(0, 2, 40),
                 convert_to_seconds(0, 1, 34),
                 convert_to_seconds(0, 1, 26),
                 convert_to_seconds(0, 1, 49)]

# Data for total execution time
total_time = [convert_to_seconds(0, 38, 17),
              convert_to_seconds(0, 11, 31),
              convert_to_seconds(0, 6, 11),
              convert_to_seconds(0, 3, 38),
              convert_to_seconds(0, 3, 13),
              convert_to_seconds(0, 3, 44)]

# Plotting preprocessing time
plt.figure(figsize=(10, 6))
plt.plot(workers, preprocessing_time, marker='o', linestyle='-', label='Preprocessing Time', color='blue')

# Plotting training time
plt.plot(workers, training_time, marker='o', linestyle='-', label='Training Time', color='green')

# Plotting total execution time
plt.plot(workers, total_time, marker='o', linestyle='-', label='Total Execution Time', color='purple')

# Adding labels and title
plt.xlabel('Number of Workers')
plt.ylabel('Time Taken (seconds)')
plt.title('Time Taken vs Number of Workers')
plt.legend()
plt.grid(True)
plt.savefig('parallelism_workers.png')