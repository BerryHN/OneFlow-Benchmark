# Example 1:
import oneflow as flow
import numpy as np
import oneflow.typing as tp



x = np.array([1, 2, 3, 4], dtype=np.float32)

# flow.enable_eager_execution(True)
@flow.global_function()
def masked_fill_Job(x: tp.Numpy.Placeholder(x.shape), mask: tp.Numpy.Placeholder((4, ),dtype=flow.int8)
)->tp.Numpy:
    out = flow.masked_fill(x, mask, value=5)
    return out

mask = np.array([1, 0, 0, 1], dtype=np.int8)

print(masked_fill_Job(x, mask))
