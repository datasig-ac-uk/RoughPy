A stream means an object that provides the signature or log-signature over any interval.
Streams are parametrised sequential data viewed via Rough Path theory as a rough path.

To create a **stream**, you need **data**. 
For example, we can create data using a linearly spaced time array.
 - [ ] TODO - is this the simplest data we can come up with? Change this to time, temperature, paracetamol intake (3 columns).

```python 
times = np.linspace(0.1,1,10)
times2 = 2*times

dt = np.abs(times[0] - times[1])
d2t = np.abs(times2[0]-times2[1]

data = np.concatenate([times.reshape(-1,1), 
                       times2.reshape(-1,1), 
                       (times*d2t - 2*times*dt).reshape(-1,1)], axis=1)
```

To create a **stream**, we should also provide a **context**. For more information, see Contexts. 
 - [ ] TODO: Link contexts page

For example, we can create the following context, with Width 2, Depth 6, and Real coefficients.
- [ ] TODO: Comment on max depth in relation to data?

```python
context = roughpy.get_context(width=2, depth=6, coeffs=roughpy.DPReal)
```

You can then create a **stream** using your chosen **data** and **context** with **Lie** increments.

```python
stream = roughpy.LieIncrementStream.from_increments(data, indices=times, ctx=context)
```

You can also plot a stream.

```python

```

- [ ] TODO: Show sig and log sig
- [ ] TODO: Plot the stream? Cell 11 in electricity data example?
---
- [ ] TODO: Tick data, constructor
- [ ] TODO: Language (English alphabet) example