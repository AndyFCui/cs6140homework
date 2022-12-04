import torch as to
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


class Predictor(to.nn.Module):
    def __init__(self, h1, h2):
        super().__init__()
        if h2 != 0:
            self.net = to.nn.Sequential(
                to.nn.Linear(2, h1),
                to.nn.Tanh(),
                to.nn.Linear(h1, h2),
                to.nn.Tanh(),
                to.nn.Linear(h2, 1),
                to.nn.Tanh()
            )
        else:
            self.net = to.nn.Sequential(
                to.nn.Linear(2, h1),
                to.nn.Tanh(),
                to.nn.Linear(h1, 1),
                to.nn.Tanh(),
            )

    def forward(self, x):
        y = self.net(x)
        return y


def train_model(model_name, model, X, y, X_val, y_val):
    model.train()

    acc_list = []
    b_acc_list = []
    auc_list = []
    acc_val_list = []
    b_acc_val_list = []
    auc_val_list = []

    ds = to.utils.data.TensorDataset(to.tensor(X).float(), to.tensor(y).float())

    k_fold_num = 5
    kf = KFold(n_splits=k_fold_num)

    loss_obj = to.nn.BCELoss()

    def loss_fcn(c, label):
        return loss_obj((c.flatten() + 1) * 0.5, (label + 1) * 0.5)

    def reset_model_param_fcn(m):
        if isinstance(m, to.nn.Linear):
            m.reset_parameters()

    for k, (train_idx, test_idx) in enumerate(kf.split(ds)):
        sp_train = to.utils.data.SubsetRandomSampler(train_idx)
        sp_test = to.utils.data.SubsetRandomSampler(test_idx)
        dl_train = to.utils.data.DataLoader(ds, batch_size=100,
                                                sampler=sp_train)
        dl_test = to.utils.data.DataLoader(ds, batch_size=100,
                                           sampler=sp_test)

        model.apply(reset_model_param_fcn)
        opt = to.optim.Adam(model.parameters(),
                            lr=1e-3,
                            betas=(0.9, 0.999),
                            weight_decay=1e-2)

        num_epoch = 3000
        eval_loss_hist = []
        min_eval_loss = None
        for epoch in range(num_epoch):
            model.train()
            for p, label in dl_train:
                c = model(p)
                loss = loss_fcn(c, label)
                opt.zero_grad()
                loss.backward()
                opt.step()
            with to.no_grad():
                loss_acc = 0.0
                loss_cnt = 0
                model.eval()
                for p, label in dl_test:
                    c = model(p)
                    loss = loss_fcn(c, label)
                    loss_acc += loss.item()
                    loss_cnt += label.size(0)
                loss_ave = loss_acc / loss_cnt
                eval_loss_hist.append(loss_ave)
                if min_eval_loss is None or min_eval_loss > loss_ave:
                    min_eval_loss = loss_ave
                    to.save(model.state_dict(), f"{model_name}.best_model.{k}fold.pt")
                if len(eval_loss_hist) > 10:
                    eval_loss_hist = eval_loss_hist[1:-1]
                    eval_loss_hist_np = np.array(eval_loss_hist)
                    diff = np.diff(eval_loss_hist_np)
                    if np.sum(diff < 0) < 4:
                        break

                # print("epoch#{} loss(eval): {}".format(
                #     epoch,
                #     loss_ave
                # ))

        with to.no_grad():
            state_dict = to.load(f"{model_name}.best_model.{k}fold.pt")
            model.load_state_dict(state_dict)
            model.eval()
            acc, b_acc, auc = evaluate_model(model, X, y)
            acc_list.append(acc)
            b_acc_list.append(b_acc)
            auc_list.append(auc)

            acc_val, b_acc_val, auc_val = evaluate_model(model, X_val, y_val)
            print("val: acc: {}, b_acc: {}, auc: {}".format(
                acc_val, b_acc_val, auc_val
            ))
            acc_val_list.append(acc_val)
            b_acc_val_list.append(b_acc_val)
            auc_val_list.append(auc_val)

    acc_list = np.array(acc_list)
    b_acc_list = np.array(b_acc_list)
    auc_list = np.array(auc_list)

    acc_val_list = np.array(acc_val_list)
    b_acc_val_list = np.array(b_acc_val_list)
    auc_val_list = np.array(auc_val_list)

    print(f"model({model_name}) gives acc: {acc_list.mean()}, b_acc: {b_acc_list.mean()}, auc: {auc_list.mean()}")
    print(f"(val) model({model_name}) gives acc: {acc_val_list.mean()}, b_acc: {b_acc_val_list.mean()}, auc: {auc_val_list.mean()}")


def evaluate_model(model, X, y):
    model.eval()
    y_pred = model(to.tensor(X).float())
    y_pred = ((y_pred + 1) * 0.5).flatten().numpy()
    y_pred = np.round(y_pred).astype(np.int)
    y = np.round((y + 1) * 0.5).astype(np.int)
    # print("y_pred")
    # print(y_pred)
    # print("y")
    # print(y)
    acc = accuracy_score(y, y_pred)
    b_acc = balanced_accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_pred)
    return acc, b_acc, auc
