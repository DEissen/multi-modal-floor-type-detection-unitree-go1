import torch
from torchmetrics.classification import MulticlassConfusionMatrix


class ConfusionMatrix():
    """
        Custom confusion matrix class based on torchmetrics.classification.MulticlassConfusionMatrix but
        with some getter functions for most important metrics.

        Due to the used class form torch, row indices of the confusion matrix correspond to the true class labels 
        and column indices correspond to the predicted class labels:
            ---------------------------------
            |True Negatives | False Positives| \n
            |False Negatives| True Positives | \n
            ---------------------------------

    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confmatrix = MulticlassConfusionMatrix(self.num_classes)

    def reset(self):
        self.confmatrix = MulticlassConfusionMatrix(self.num_classes)

    def update(self, pred, labels):
        self.confmatrix(pred, labels)

    def get_result(self):
        return self.confmatrix.compute().numpy()
    
    def get_tp_tn_fn_fp_total(self):
        conf_matrix = self.confmatrix.compute()
        
        if self.num_classes == 2:
            tp = conf_matrix[1][1]
            fn = conf_matrix[1][0]
            fp = conf_matrix[0][1]
            total = torch.sum(conf_matrix)
            tn = total - tp - fn - fp
        else:
            tp = torch.diagonal(conf_matrix)
            fn = torch.sum(conf_matrix, 1) - tp
            fp = torch.sum(conf_matrix, 0) - tp
            total = torch.sum(conf_matrix)
            tn = total - tp - fn - fp

        return tp, tn, fn, fp, total

    def get_accuracy(self):
        tp, tn, fn, fp, total = self.get_tp_tn_fn_fp_total()

        if self.num_classes == 2:
            correct = tp+tn
        else:
            correct = tp

        accuracy = correct / total

        if self.num_classes == 2:
            accuracy = torch.sum(accuracy)

        return accuracy

    def get_error_rate(self):
        return 1 - self.get_accuracy()

    def get_sensitivity(self):
        tp, tn, fn, fp, total = self.get_tp_tn_fn_fp_total()

        sensitivity = tp / (tp + fn)

        return sensitivity

    def get_specificity(self):
        tp, tn, fn, fp, total = self.get_tp_tn_fn_fp_total()

        specificity = tn / (fp + tn)
    
        return specificity

    def get_balanced_accuracy(self):
        return (self.get_sensitivity() + self.get_specificity()) / 2
    
    def plot(self):
        self.confmatrix.plot()
