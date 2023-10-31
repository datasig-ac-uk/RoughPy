# Contexts

- [ ] Providing a Width, Depth, and a coefficient field
- [ ] Provides access to the Baker-Campbell-Hausdorff formula
- [ ] Allows us to transfer between signatures and log signatures (t2l, l2t functions) 

---

We can create a **context** for a **stream** in the following way. The example below has **width** 2, **depth** 6, and **coefficients** over the Reals.

```python
context = roughpy.get_context(width=2, depth=6, coeffs=rp.DPReal)
```