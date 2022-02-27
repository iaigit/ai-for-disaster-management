from tqdm import tqdm

def train_engine(dataloader, model, loss_fn, optim):
    model.train()
    loss_one_step = 0

    for data, target in tqdm(dataloader):
        preds = model(data)
        loss = loss_fn(preds, target)
        optim.zero_grad()
        loss.backward()
        optim.step()
        loss_one_step += loss.item()

    return loss_one_step / len(dataloader)
    