import models
import numpy as np
import os
import oneflow as flow
import oneflow.typing as tp


mconf = models.GPTConfig(vocab_size=100, block_size=128,
                  n_layer=8, n_head=8, n_embd=512)

batch_size = 32

#flow.enable_eager_execution(True)

class Snapshot(object):
    def __init__(self, model_save_dir, model_load_dir):
        self._model_save_dir = model_save_dir
        self._check_point = flow.train.CheckPoint()
        if model_load_dir:
            assert os.path.isdir(model_load_dir)
            print("Restoring model from {}.".format(model_load_dir))
            self._check_point.load(model_load_dir)
        else:
            self._check_point.init()
            #self.save('initial_model')
            print("Init model on demand.")

    def save(self, name):
        snapshot_save_path = os.path.join(self._model_save_dir, "snapshot_{}".format(name))
        if not os.path.exists(snapshot_save_path):
            os.makedirs(snapshot_save_path)
        print("Saving model to {}.".format(snapshot_save_path))
        self._check_point.save(snapshot_save_path)

@flow.global_function()
def test_att(
    x: tp.Numpy.Placeholder((batch_size, 
    mconf.block_size, mconf.n_embd), dtype=flow.float32)
) -> tp.Numpy:
    y = models.Causal_Self_Attention(x,mconf)
    return y
    
@flow.global_function()
def test_GPT(
    x: tp.Numpy.Placeholder((batch_size, mconf.block_size), dtype=flow.int32)
) -> tp.Numpy:
    y,_ = models.GPT(x,mconf)
    return y

if __name__ == "__main__":
    #snapshot = Snapshot('./log/', None)
    
    x = np.ones( (batch_size, mconf.block_size,mconf.n_embd)).astype(np.float32)
    
    words_tokens = np.ones((batch_size, mconf.block_size)).astype(np.int32)
    print('_______________________________________________')
    gpt_output = test_GPT(words_tokens)
    print(gpt_output)
    print(test_att(x))
