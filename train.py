import copy
from tqdm.auto import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models import R3DModel
import util


# params
seed = 88
epochs = 200
batch_size = 1
lr = 0.001
weight_decay = 5e-4
dropout = 0.05
device = 'cpu'

def same_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

same_seed(seed)

# Model and optimizer
model = R3DModel(dropout=dropout).to(device)

optimizer = optim.Adam(model.parameters(),
                       lr=lr, weight_decay=weight_decay)

loss_fct = torch.nn.CrossEntropyLoss()

params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 6,
          'drop_last': True}


def train(model, data):
    train_loader = DataLoader(data, **params)

    model.train()

    acc_train = 0
    loss_history = 0
    cnt = 0

    for batched_vids, labels in tqdm(train_loader):
        batched_vids, labels = batched_vids.to(device), labels.to(device)
        logits = model(batched_vids)
        loss = loss_fct(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (torch.argmax(logits, dim=1) == labels).float().mean()
        acc_train = acc_train + acc.item()
        loss_history = loss_history + loss.item()

        cnt = cnt + 1


    return loss_history / cnt, acc_train / cnt


def val_test(model, data):
    val_test_dataloader = DataLoader(data, **params)

    model.eval()

    acc_val_test = 0
    loss_history = 0
    cnt = 0

    with torch.no_grad():
        for batched_vids, labels in tqdm(val_test_dataloader):
            batched_vids, labels = batched_vids.to(device), labels.to(device)
            logits = model(batched_vids)
            loss = loss_fct(logits, labels)

            acc = (torch.argmax(logits, dim=1) == labels).float().mean()
            acc_val_test = acc_val_test + acc.item()
            loss_history = loss_history + loss.item()
            cnt = cnt + 1


    return loss_history/cnt, acc_val_test / cnt

def main():
    max_acc = 0.7
    model_max = copy.deepcopy(model)

    # logs
    writer = SummaryWriter('./output/logs')

    # 加载数据
    print("Collecting data!")
    raw_dataset = util.vidData('./test_dataset')
    print("Data collected!")

    # 0.8, 0.1, 0.1 splitting
    allNum = len(raw_dataset)
    train_size = int(allNum * 0.8)
    val_size = int(allNum * 0.1)
    test_size = allNum - train_size - val_size
    training_data, test_data, val_data = torch.utils.data.random_split(raw_dataset, [train_size, test_size, val_size])

    print('Start Training...')
    for epoch in range(epochs):
        print('-------- Epoch ' + str(epoch + 1) + ' --------')

        loss_train, acc_train = train(model, training_data)

        loss_val, acc_val = val_test(model, val_data)

        print('Training Loss: {:.4f}'.format(loss_train))
        print('Training acc: {:.4f}'.format(acc_train))
        writer.add_scalar('Train Loss', loss_train, global_step=epoch)
        print('Val Loss: {:.4f}'.format(loss_val))
        print('Val acc: {:.4f}'.format(acc_val))
        writer.add_scalar('Val Loss', loss_val, global_step=epoch)

        if acc_val > max_acc:
            model_max = copy.deepcopy(model)
            max_acc = acc_val

            # 保存
            model_max.eval()
            torch.save(model_max, './output/models/model' + str(epoch) + '_' + str(max_acc) + '.pth')

    writer.close()

    loss_test, acc_test = val_test(model_max, test_data)
    print('test Loss: {:.6f}'.format(loss_test))
    print('test acc: {:.6f}'.format(acc_test))
    print("finished!")

if __name__ == '__main__':
    main()
