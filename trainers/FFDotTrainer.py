import torch
from torch.nn import BCELoss
from tqdm import tqdm

from classifiers.differential.FFDot import FFDot
from trainers.BaseFFTrainer import BaseFFTrainer
from losses.CrossEntropyLoss import CrossEntropyLoss


class FFDotTrainer(BaseFFTrainer):
    def __init__(self, model: FFDot, loss_fn=None, device="cuda" if torch.cuda.is_available() else "cpu", save_path=None):
        # If no loss function is provided, use BCELoss by default for FFDotTrainer
        super().__init__(model, device, save_path=save_path)
        self.lossfn = loss_fn if loss_fn else torch.nn.BCELoss()

    def train_epoch(self, train_dataloader) -> tuple[list[float], list[float]]:
        losses = []
        accuracies = []
        # Training loop
        for _, gt, test, label in tqdm(train_dataloader):

            gt = gt.to(self.device)
            test = test.to(self.device)
            label = label.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            prob = self.model(gt, test)

            # Compute loss
            loss = self.lossfn(prob, label)
            loss.backward()
            self.optimizer.step()

            # Compute accuracy
            acc = (prob.round() == label).float().mean()

            losses.append(loss.item())
            accuracies.append(acc.item())

        return accuracies, losses

    def val_epoch(
        self, val_dataloader, save_scores=False
    ) -> tuple[list[float], list[float], list[float], list[int], list[str]]:
        losses = []
        labels = []
        scores = []
        predictions = []
        file_names = []

        for file_name, gt, test, label in tqdm(val_dataloader):
            gt = gt.to(self.device)
            test = test.to(self.device)
            label = label.to(self.device)

            prob = self.model(gt, test)
            loss = self.lossfn(prob, label)

            if save_scores:
                file_names.extend(file_name)
            predictions.extend(prob.round().tolist())
            losses.append(loss.item())
            labels.extend(label.tolist())
            scores.extend(prob.tolist())

        return losses, labels, scores, predictions, file_names
