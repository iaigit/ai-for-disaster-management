import imp
from .engine_step.val_step import val_engine

def test(test_dataloader, model, loss_fn):
    loss = val_engine(test_dataloader, model, loss_fn)
    print('Test Loss: {:.4f}'.format(loss))

    return loss