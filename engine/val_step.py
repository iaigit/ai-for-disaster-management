from tqdm import tqdm

def val_engine(dataloader, model, loss_fn):
    model.eval()
    loss_one_step = 0

    for data, target in tqdm(dataloader):
        preds = model(data)
        loss = loss_fn(preds, target)
        loss_one_step += loss.item()

    return loss_one_step / len(dataloader)
    