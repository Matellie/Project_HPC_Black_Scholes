import matplotlib.pyplot as plt
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Plot Monte Carlo Simulated Stock Price Paths.")
parser.add_argument("--filename", type=str, required=True, help="Path to the CSV file containing simulation data.")
args = parser.parse_args()

data = pd.read_csv(args.filename, header=None)

plt.figure(figsize=(10, 6))
for col in data.columns:
    plt.plot(data[col])

plt.title("Monte Carlo Simulated Stock Price Paths")
plt.xlabel("Time Intervals")
plt.ylabel("Stock Price")
plt.legend(loc="upper left", fontsize="small", ncol=2)
plt.grid()
plt.show()
