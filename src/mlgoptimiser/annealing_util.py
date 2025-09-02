from . import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_energy_plot(energy, qenergy):
    # Create lists for x and y values for the first dataset
    x_values = []
    y_values_energy = []

    for key, val in energy.items():
        if int(key) >= 1:
            x_values.append(int(key))
            y_values_energy.append(val[0])  # Assuming val is a list or tuple

    # Create y values for the second dataset
    y_values_qenergy = []
    x_values_qenergy = []
    for key, val in qenergy.items():
        if int(key) >= 1:
            y_values_qenergy.append(val)  # Assuming val is a single value
            x_values_qenergy.append(key)

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot the first dataset on the primary y-axis
    ax1.scatter(
        x_values,
        y_values_energy,
        color="blue",
        edgecolor="black",
        alpha=0.7,
        s=100,
        label="Energy",
    )
    ax1.set_xlabel("X Values", fontsize=12)
    ax1.set_ylabel("Energy", fontsize=12, color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Create a secondary y-axis for the second dataset
    ax2 = ax1.twinx()
    ax2.scatter(
        x_values_qenergy,
        y_values_qenergy,
        color="red",
        edgecolor="black",
        alpha=0.7,
        s=100,
        label="QEnergy",
    )
    ax2.set_ylabel("QEnergy", fontsize=12, color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # Add a title
    plt.title("Energy and QEnergy vs. X Values", fontsize=14)

    # Save and show the plot
    plt.savefig("energy_plot.png")


def get_disp(filename):
    disp = 0.0
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("  Largest displacement in region 2"):
                disp = line.strip().split()[-2]
    return float(disp)
