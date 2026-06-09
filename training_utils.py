"""Shared training and evaluation helpers."""
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from project_constants import CONFUSABLE_PAIRS, DEVICE, LEARNING_RATE, NUM_CLASSES


class LabelSmoothing(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1 - smoothing

    def forward(self, predictions, targets):
        log_prob = functional.log_softmax(predictions, -1)
        with torch.no_grad():
            smooth = torch.full_like(log_prob, self.smoothing / (self.num_classes - 1))
            smooth.scatter_(1, targets.unsqueeze(1), self.confidence)
        return (-smooth * log_prob).sum(-1).mean()


def _mixup_or_cutmix(images, labels, mixup_alpha=0.0, cutmix_alpha=0.0, p_cutmix=0.5):
    """随机选 mixup 或 cutmix 混合一个 batch.

    返回 (mixed_images, y_a, y_b, lam). 计算损失方式:
      loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
    """
    batch_size = images.size(0)
    perm = torch.randperm(batch_size, device=images.device)
    y_a, y_b = labels, labels[perm]
    use_cutmix = (cutmix_alpha > 0) and (
        mixup_alpha <= 0 or torch.rand(1).item() < p_cutmix)
    use_mixup = (mixup_alpha > 0) and not use_cutmix
    if use_cutmix:
        lam = float(torch.distributions.Beta(cutmix_alpha, cutmix_alpha).sample().item())
        _, _, height, width = images.shape
        cut_ratio = (1.0 - lam) ** 0.5
        cut_h = int(height * cut_ratio)
        cut_w = int(width * cut_ratio)
        center_y = int(torch.randint(0, height, (1,)).item())
        center_x = int(torch.randint(0, width, (1,)).item())
        y1 = max(0, center_y - cut_h // 2)
        y2 = min(height, center_y + cut_h // 2)
        x1 = max(0, center_x - cut_w // 2)
        x2 = min(width, center_x + cut_w // 2)
        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
        actual_area = (y2 - y1) * (x2 - x1)
        lam = 1.0 - actual_area / (height * width)
        return mixed, y_a, y_b, lam
    if use_mixup:
        lam = float(torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item())
        mixed = lam * images + (1 - lam) * images[perm]
        return mixed, y_a, y_b, lam
    return images, labels, labels, 1.0


def train_one_epoch(net, loader, optimizer, criterion,
                    mixup_alpha=0.0, cutmix_alpha=0.0):
    net.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        if mixup_alpha > 0 or cutmix_alpha > 0:
            mixed_images, y_a, y_b, lam = _mixup_or_cutmix(
                images, labels, mixup_alpha, cutmix_alpha)
            outputs = net(mixed_images)
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
        else:
            outputs = net(images)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # mixup 下 train acc 仅作参考 (用主要标签 y_a)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += len(labels)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(net, loader):
    net.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images = images.to(DEVICE)
        correct += (net(images).argmax(1).cpu() == labels).sum().item()
        total += len(labels)
    return correct / total


@torch.no_grad()
def evaluate_per_class(net, loader):
    net.eval()
    correct = defaultdict(int)
    total = defaultdict(int)
    for images, labels in loader:
        images = images.to(DEVICE)
        predictions = net(images).argmax(1).cpu()
        for i in range(len(labels)):
            lbl = labels[i].item()
            total[lbl] += 1
            if predictions[i] == lbl:
                correct[lbl] += 1
    return {cls_idx: correct[cls_idx] / total[cls_idx] for cls_idx in total}


def fit_best_model(net, train_loader, validation_loader, epochs, progress_label="",
                   mixup_alpha=0.0, cutmix_alpha=0.0):
    criterion = LabelSmoothing(NUM_CLASSES)
    optimizer = AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    best_accuracy = 0.0
    best_state = {key: value.cpu().clone() for key, value in net.state_dict().items()}

    for epoch_index in range(epochs):
        training_loss, _ = train_one_epoch(net, train_loader, optimizer, criterion,
                                            mixup_alpha=mixup_alpha,
                                            cutmix_alpha=cutmix_alpha)
        validation_accuracy = evaluate(net, validation_loader)
        scheduler.step()
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_state = {key: value.cpu().clone() for key, value in net.state_dict().items()}
        done = epoch_index + 1
        bar = "#" * (done * 20 // epochs) + "-" * (20 - done * 20 // epochs)
        prefix = "[%s] " % progress_label if progress_label else ""
        print("\r  %sEp%2d/%d [%s] loss=%.3f last_val=%.4f" %
              (prefix, done, epochs, bar, training_loss, validation_accuracy), end="", flush=True)
    print()

    net.load_state_dict(best_state)
    return best_accuracy, best_state


def make_pair_key(first_label, second_label):
    return "%s/%s" % (first_label, second_label)


@torch.no_grad()
def compute_pair_accuracy(trained_model, loader, index_to_label):
    trained_model.eval()
    stats = {(first, second): PairCounter() for first, second in CONFUSABLE_PAIRS}
    for images, labels in loader:
        images = images.to(DEVICE)
        predictions = trained_model(images).argmax(1).cpu()
        for i in range(len(labels)):
            true_label = index_to_label[labels[i].item()]
            predicted_label = index_to_label[predictions[i].item()]
            for first, second in CONFUSABLE_PAIRS:
                if true_label in (first, second) and predicted_label in (first, second):
                    counter = stats[(first, second)]
                    counter.attempts += 1
                    if predicted_label == true_label:
                        counter.successes += 1
    return {make_pair_key(first, second): counter.ratio()
            for (first, second), counter in stats.items()}


@dataclass
class PairCounter:
    attempts: int = 0
    successes: int = 0

    def ratio(self):
        return self.successes / self.attempts if self.attempts > 0 else 0.0
