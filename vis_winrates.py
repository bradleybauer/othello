import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

def update(frame):
    try:
        # Load data from the file (update if file changes or grows)
        data = np.loadtxt("winrate.txt")
    except Exception as e:
        print("Error reading file:", e)
        return
    
    # Create an index array based on the number of rows
    index = np.arange(len(data))
    
    # Extract columns from the data
    column1 = data[:, 0]
    column2 = data[:, 1]
    
    # Clear previous plots
    ax1.cla()
    ax2.cla()
    
    # Plot column 1 vs index
    ax1.plot(index, column1, marker='o')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Column 1 (Winrate)')
    ax1.set_title('Column 1 vs Index')
    ax1.grid(True)
    
    # Plot column 2 vs index
    ax2.plot(index, column2, marker='o', color='orange')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Column 2 (Metric)')
    ax2.set_title('Column 2 vs Index')
    ax2.grid(True)

# Set up the animation to update every 1000 milliseconds (1 second)
ani = FuncAnimation(fig, update, interval=1000)

plt.tight_layout()
plt.show()
