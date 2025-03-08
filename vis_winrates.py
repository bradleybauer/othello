import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create the figure and two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

def update(frame):
    try:
        # Force at least a 2D array; this prevents issues when there's only one row.
        data = np.loadtxt("winrate.txt", ndmin=2)
        
        # If the file is empty, data.size will be 0; skip updating in that case.
        if data.size == 0:
            return
    except Exception as e:
        print("Error reading file:", e)
        return

    # Create an index array based on the number of rows
    index = np.arange(data.shape[0])
    
    # Extract columns
    column1 = data[:, 0]
    column2 = data[:, 1]
    
    # Clear previous plots
    ax1.cla()
    ax2.cla()
    
    # Plot column 1 vs index with fixed y-axis limits
    ax1.plot(index, column1, marker='o')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Column 1 (Winrate)')
    ax1.set_title('Column 1 vs Index')
    ax1.grid(True)
    ax1.set_ylim(0.65, .97)  # Fixed y-axis range
    
    # Plot column 2 vs index
    ax2.plot(index, column2, marker='o', color='orange')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Column 2 (Metric)')
    ax2.set_title('Column 2 vs Index')
    ax2.grid(True)

# Set up the animation to update every second and disable frame data caching to remove the warning.
ani = FuncAnimation(fig, update, interval=1000, cache_frame_data=False)

plt.tight_layout()
plt.show()
