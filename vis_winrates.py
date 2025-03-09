import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create a figure with two vertically stacked subplots that share the x-axis
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(10, 8))

def update(frame):
    try:
        # Load data as a 2D array to avoid issues with a single row.
        data = np.loadtxt("winrate.txt", ndmin=2)
        if data.size == 0:
            return  # Skip update if file is empty
    except Exception as e:
        print("Error reading file:", e)
        return

    # Create an index array from 0 to number of rows - 1
    index = np.arange(data.shape[0])
    column1 = data[:, 0]
    column2 = data[:, 1]

    # Clear previous plots on both axes
    ax1.cla()
    ax2.cla()

    # Top subplot: Plot winrate (column 1) vs index with fixed y-axis range
    ax1.plot(index, column1, color='blue')
    ax1.set_ylabel("Win rate", color='blue')
    ax1.set_ylim(0.65, 0.94)
    ax1.tick_params(axis="y", labelcolor='blue')
    ax1.set_title("Win rate vs Opponent")
    ax1.grid(True)

    # Bottom subplot: Plot metric (column 2) vs index
    ax2.plot(index, column2, color='orange')
    ax2.set_ylabel("Probability", color='orange')
    ax2.tick_params(axis="y", labelcolor='orange')
    ax2.set_title("Probability to sample Opponent")
    ax2.grid(True)

# Set up the animation to update every 1000 milliseconds (1 second)
ani = FuncAnimation(fig, update, interval=1000, cache_frame_data=False)

plt.tight_layout()
plt.show()
