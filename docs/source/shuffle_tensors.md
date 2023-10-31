# Shuffle Tensors

```python
from roughpy import ShuffleTensor

d1 = tensor_data()
shuffle_tensor1 = roughpy.ShuffleTensor(d1, ctx=tensor_context)

shuffle_tensor2 = ShuffleTensor([1 * Monomial(f"x{i}") for i in range(7)], width=2,
                    depth=2, dtype=roughpy.RationalPoly)
```