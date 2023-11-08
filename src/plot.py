import os
import matplotlib.pyplot as plt

# Specify the folder containing the txt files
data_folder = '../report'


plt.rcParams['font.size'] = 3

# Get a list of all the txt files in the folder
txt_files = [f for f in os.listdir(data_folder) if f.endswith('.txt')]

# Iterate through each txt file
for txt_file in txt_files:
    file_path = os.path.join(data_folder, txt_file)

    # Read the data from the txt file
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Parse the data assuming it's a column of numbers
        data = [float(line.strip()) for line in lines]

    # Create a new figure with higher resolution and larger size
    plt.figure(figsize=(8, 4), dpi=400)  # Increase dpi value for higher resolution

    # Plot the data, with x axis scaled down by 0.001
    plt.plot([i*0.001 for i in range(len(data))], data, linewidth=0.5)

    # Set the title of the plot
    plt.title(txt_file)

    # Display the plot
    plt.show()

    # Close the figure to free up memory
    plt.close()
