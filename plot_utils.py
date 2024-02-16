import matplotlib.pyplot as plt

def plot_series(time, series, format="-", start=0, end=None, label=None):
  plt.figure(figsize=(10, 6))


  if type(series) is tuple:
    for series_i in series:
      plt.plot(time[start:end], series_i[start:end], format)
  else:
    plt.plot(time[start:end], series[start:end], format)

  plt.xlabel("Time")

  plt.ylabel("Value")

  if label:
    plt.legend(fontsize=14, labels=label)

  plt.grid(True)

  plt.show()