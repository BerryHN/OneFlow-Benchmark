import torch_mingpt
import models
import torch
import numpy as np

import oneflow as flow
import oneflow.typing as tp
import numpy as np
mconf = models.GPTConfig(vocab_size=10, block_size=10,
                  n_layer=2, n_head=2, n_embd=8)

#print(mconf.n_embd)

flow.enable_eager_execution(True)
#x = torch.ones(1,1,4)

pymodel = torch_mingpt.CausalSelfAttention(mconf)
#print(pymodel(x))


#models.causal_self_attention(x, mconf, debug=True)

@flow.global_function()
def test_job(
    x: tp.Numpy.Placeholder((1, 2), dtype=flow.int8)
) -> None:
    # do something with images or labels
    #models.Causal_Self_Attention(x, config=mconf)
    #models.Block(x, config=mconf)
    models.GPT(x,mconf)
    
if __name__ == "__main__":
    
    x = np.ones( [1,2]).astype(np.int8)
    
    print('_______________________________________________')

    test_job(x)
