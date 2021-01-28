import torch
import torch.nn.functional as F
import numpy as np
from sklearn import metrics


def evaluate(model, dataloader):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for d, q, s, d_l, q_l in iter(dataloader):
            d = d.cuda()
            q = q.cuda()
            s = s.cuda()
            d_l = d_l.cuda()
            q_l = q_l.cuda()

            outputs = model(d, q, d_l, q_l)
            loss = F.cross_entropy(outputs, s)
            loss_total += loss
            labels = s.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.  append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    acc = metrics.accuracy_score(labels_all, predict_all)
    return acc, loss_total / len(dataloader)


def train(config, train_loader, model, test_loader, optimizer):
    total_batch = 0
    for epoch in range(config.num_epoch):
        for d, q, s, d_l, q_l in iter(train_loader):
            d = d.cuda()
            q = q.cuda()
            s = s.cuda()
            d_l = d_l.cuda()
            q_l = q_l.cuda()

            output = model(d, q, d_l, q_l)
            model.zero_grad()
            loss = F.cross_entropy(output, s)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                test_acc, test_loss = evaluate(model, test_loader)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Val Loss: {2:>5.2},  Val Acc: {3:>6.2%}'
                print(msg.format(total_batch, loss.item(), test_loss, test_acc))
                model.train()
            total_batch += 1
    torch.save(model.state_dict(), './saved/rc-cnn.ckpt')
