#!/usr/bin/python

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import util

# params
seed = 88

def same_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

same_seed(seed)


def getArg():
    parser = argparse.ArgumentParser(description='Script argparse setting')
    parser.add_argument('-d', dest='device', default='cpu', help='the device: cpu or cuda, default is cpu')
    parser.add_argument('-t', dest='type', default='r2plus1d', help='the model type: r3d, mc3 or r2plus1d, default is r2plus1d')
    parser.add_argument('-i', dest='input', default='./test/test.mp4',
                        help='Path to input file, default is ./test/test.mp4. If want to predict all data, please input all.')
    args = parser.parse_args()
    device, type, input = args.device, args.type, args.input

    return device, type, input

def test(model, data, device):
    params = {'batch_size': 16,
              'shuffle': True,
              'num_workers': 6,
              'drop_last': True}
    test_dataloader = DataLoader(data, **params)
    loss_fct = torch.nn.CrossEntropyLoss()
    model.eval()

    acc_test = 0
    loss_history = 0
    cnt = 0

    with torch.no_grad():
        for batched_vids, labels in test_dataloader:
            batched_vids, labels = batched_vids.to(device), labels.to(device)
            logits = model(batched_vids)
            loss = loss_fct(logits, labels)

            acc = (torch.argmax(logits, dim=1) == labels).float().mean()
            acc_test = acc_test + acc.item()
            loss_history = loss_history + loss.item()
            cnt = cnt + 1

    return loss_history / cnt, acc_test / cnt


def main(device, type, input):
    if device == 'cpu':
        if type == 'r3d':
            model = torch.load('test/modelr3d.pth', map_location=device)
        elif type == 'mc3':
            model = torch.load('test/modelmc3.pth', map_location=device)
        elif type == 'r2plus1d':
            model = torch.load('test/modelr2plus1d.pth', map_location=device)
    elif device == 'cuda':
        if type == 'r3d':
            model = torch.load('test/modelr3d.pth', map_location=device)
        elif type == 'mc3':
            model = torch.load('test/modelmc3.pth', map_location=device)
        elif type == 'r2plus1d':
            model = torch.load('test/modelr2plus1d.pth', map_location=device)
    model.to(device)

    if input == 'all':
        # 加载数据
        print("Collecting data!")
        raw_dataset = util.vidData('./3_class_dataset')
        print("Data collected!")

        # 0.8, 0.1, 0.1 splitting
        allNum = len(raw_dataset)
        train_size = int(allNum * 0.8)
        val_size = int(allNum * 0.1)
        test_size = allNum - train_size - val_size
        training_data, test_data, val_data = torch.utils.data.random_split(raw_dataset, [train_size, test_size, val_size])

        loss_train, acc_train = test(model, training_data, device)
        loss_val, acc_val = test(model, val_data, device)
        loss_test, acc_test = test(model, test_data, device)

        print('Training Loss: {:.4f}'.format(loss_train))
        print('Training acc: {:.4f}'.format(acc_train))
        print('Val Loss: {:.4f}'.format(loss_val))
        print('Val acc: {:.4f}'.format(acc_val))
        print('test Loss: {:.6f}'.format(loss_test))
        print('test acc: {:.6f}'.format(acc_test))

        print('Finished!')

    else:
        model.eval()
        vid = util.getOneVid(input).unsqueeze(0).to(device)
        logits = model(vid)
        predict = torch.argmax(logits, dim=1).flatten().item()
        if predict == 0:
            level = 'light'
        elif predict == 1:
            level = 'moderate'
        elif predict == 2:
            level = 'vigorous'
        print('Class is {}'.format(level))
        print('Finished!')

if __name__ == "__main__":
    device, type, input = getArg()
    main(device, type, input)


print()