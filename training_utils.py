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


def train_one_epoch(net, loader, optimizer, criterion):
    net.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
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


def fit_best_model(net, train_loader, validation_loader, epochs, progress_label=""):
    criterion = LabelSmoothing(NUM_CLASSES)
    optimizer = AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    best_accuracy = 0.0
    best_state = {key: value.cpu().clone() for key, value in net.state_dict().items()}

    for epoch_index in range(epochs):
        training_loss, _ = train_one_epoch(net, train_loader, optimizer, criterion)
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
