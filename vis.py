import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

def read_histogram_data(filename):
    histogram = []
    try:
        with open(filename, 'r') as file:
            # Read and discard the header line
            header = file.readline()
            for line in file:
                line = line.strip()
                if line:
                    try:
                        # Each line is interpreted as a frequency for a bin
                        value = float(line)
                        histogram.append(value)
                    except ValueError:
                        print(f"Skipping invalid entry: {line}")
    except FileNotFoundError:
        print(f"File {filename} not found.")
    return histogram

filename = 'win_rate_data.txt'
last_mtime = os.path.getmtime(filename)

fig, ax = plt.subplots()

def update(frame):
    global last_mtime
    try:
        current_mtime = os.path.getmtime(filename)
        if current_mtime != last_mtime:
            # Load the updated histogram data
            data = read_histogram_data(filename)
            ax.clear()  # Clear previous plot
            
            # Use the bin indices as x-axis labels
            indices = list(range(len(data)))
            ax.bar(indices, data, edgecolor='black')
            ax.set_xlabel("Bin Index")
            ax.set_ylabel("Frequency")
            ax.set_title("Unnormalized Histogram")
            
            last_mtime = current_mtime
            print("Histogram updated with", len(data), "bins.")
    except Exception as e:
        print(f"Error: {e}")

# Use FuncAnimation for smooth updating every 1000 milliseconds (1 second)
ani = animation.FuncAnimation(fig, update, interval=1000)

plt.show()
