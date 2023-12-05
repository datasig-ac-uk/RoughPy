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
