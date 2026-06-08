"""Optuna 贝叶斯调参: 搜索 lr, weight_decay, dropout, attention_type"""
import optuna
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np, os, sys, datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import HandCharNet, HandCharNetCA, HandCharNetECA, HandCharNetCBAM, HandCharNetGroupSE
from data_utils import load_split_data, CharDataset, collate_with_augment
from project_constants import DEVICE, NUM_CLASSES, BATCH_SIZE
from training_utils import evaluate, LabelSmoothing, train_one_epoch

TUNE_EPOCHS = 20
N_TRIALS = 30

# === 预加载数据 (所有 trial 共享，避免重复磁盘 I/O) ===
_FULL_TRAIN, _, _, _, _ = load_split_data()
_RNG = np.random.RandomState(42)
_ALL_IDX = _RNG.permutation(len(_FULL_TRAIN))
_SPLIT = int(len(_ALL_IDX) * 0.8)
_TR_IDX = _ALL_IDX[:_SPLIT]
_VL_IDX = _ALL_IDX[_SPLIT:]


def _make_loader(indices, train_flag):
    ds = CharDataset([_FULL_TRAIN[i] for i in indices], train=train_flag)
    kwargs = dict(batch_size=BATCH_SIZE, shuffle=train_flag, num_workers=0)
    if train_flag:
        kwargs["collate_fn"] = collate_with_augment
    return DataLoader(ds, **kwargs)


def objective(trial):
    lr = trial.suggest_float("lr", 5e-4, 3e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    attention = trial.suggest_categorical("attention", ["se", "eca", "cbam", "ca", "gse"])

    train_dl = _make_loader(_TR_IDX, True)
    val_dl = _make_loader(_VL_IDX, False)

    model_map = {"se": HandCharNet, "eca": HandCharNetECA, "cbam": HandCharNetCBAM,
                 "ca": HandCharNetCA, "gse": HandCharNetGroupSE}
    net = model_map[attention](NUM_CLASSES, dropout=dropout).to(DEVICE)

    crit = LabelSmoothing(NUM_CLASSES)
    opt = AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = CosineAnnealingLR(opt, T_max=TUNE_EPOCHS)
    best_acc = 0.0
    for _ in range(TUNE_EPOCHS):
        train_one_epoch(net, train_dl, opt, crit)
        acc = evaluate(net, val_dl)
        sched.step()
        if acc > best_acc:
            best_acc = acc
    return best_acc


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    best_params = study.best_params
    print("\nBest params:", best_params, "val=%.4f" % study.best_value)
    with open("output/tune_%s.txt" % timestamp, "w", encoding="utf-8") as f_out:
        f_out.write("best_params: %s\n" "best_val: %.4f\n" "\n" "All trials:\n" %
                     (best_params, study.best_value))
        for t in study.trials:
            value = t.value if t.value is not None else 0.0
            f_out.write("%.4f  %s\n" % (value, t.params))
    print("Saved: output/tune_%s.txt" % timestamp)
