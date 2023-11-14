import numpy as np
import roughpy as rp

rng = np.random.default_rng(1635134)

# Sample times
times = np.linspace(0.1,10,100)
times2 = 2*times

dt = np.abs(times[0]-times[1])
d2t = np.abs(times2[0]-times2[1])

# Moderate length 2D paths
data = np.concatenate([times.reshape(-1,1), times2.reshape(-1,1), (times*d2t - 2*times*dt).reshape(-1,1)], axis=1)
print(data)

interval = rp.RealInterval(0, 1)
print("The interval of definition", interval)

ctx = rp.get_context(width=2, depth=6, coeffs=rp.DPReal)

stream = rp.LieIncrementStream.from_increments(data, indices=times, ctx=ctx)

sig = stream.signature(interval)
logsig = stream.log_signature(interval)

print(f"{sig=!s}")
print(f"{logsig=!s}")

### plotting

import matplotlib.pyplot as plt
import math

resolution = 14

def plot_cycles(interval):
    fig, axes = plt.subplots()
    range_start = math.ceil(interval.inf()*2**resolution)
    range_end = math.ceil(interval.sup()*2**resolution)

    increments = np.vstack([np.array(lsig)
                            for i in range(range_start, range_end)
                            if (lsig:=stream.log_signature(rp.DyadicInterval(i, resolution), depth=1)).is_zero()])
    print(f"Using {len(increments)} increments")
    axes.plot(increments[:, 0], increments[:, 1])
    axes.set(xlabel="Voltage", ylabel="Charge", title=f"plot of charge against Voltage over {interval}")
    fig.savefig("stream.pdf")

plot_cycles(rp.RealInterval(10, 10.5))
