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
        """
            Constructor of ConfusionMatrix() class.

            Parameters:
                - num_classes (int): Number of classes present in the classification problem
        """
        self.num_classes = num_classes
        self.confmatrix = MulticlassConfusionMatrix(self.num_classes)

    def reset(self):
        """
            Method to reset the confusion matrix.
        """
        self.confmatrix = MulticlassConfusionMatrix(self.num_classes)

    def update(self, pred, labels):
        """
            Method to update the state of the confusion matrix based on the predictions and the ground truth/ labels.

            Parameters:
                - pred (torch.Tensor): Tensor with the predictions for the batch
                - labels (torch.Tensor): Tensor with the labels for the batch
        """
        self.confmatrix(pred, labels)

    def get_result(self):
        """
            Method to get the current state of the confusion matrix as a numpy array.

            Returns:
                - (np.array): Numpy array of the current state of the confusion matrix
        """
        return self.confmatrix.compute().numpy()

    def get_tp_tn_fn_fp_total(self):
        """
            Method to get the numbers of True Positives, True Negatives, False Negatives, False Positives and of all elements.

            Returns:
                - tp (torch.Tensor): Tensor with True Positives for each class (single number in case of self.num_classes == 2)
                - tn (torch.Tensor): Tensor with True Negatives for each class (single number in case of self.num_classes == 2)
                - fn (torch.Tensor): Tensor with False Negatives for each class (single number in case of self.num_classes == 2)
                - fp (torch.Tensor): Tensor with False Positives for each class (single number in case of self.num_classes == 2)
                - total (torch.Tensor): Sum of all elements.
        """
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
        """
            Method to get the accuracy.

            Returns:
                - accuracy (torch.Tensor): Tensor with accuracy (single number)
        """
        tp, tn, fn, fp, total = self.get_tp_tn_fn_fp_total()

        if self.num_classes == 2:
            correct = tp+tn
        else:
            correct = tp

        accuracy = correct / total
        
        accuracy = torch.sum(accuracy)

        return accuracy

    def get_error_rate(self):
        """
            Method to get the error rate for all classes.

            Returns:
                - (torch.Tensor): Tensor with error rate for each class (single number in case of self.num_classes == 2)
        """
        return 1 - self.get_accuracy()

    def get_sensitivity(self):
        """
            Method to get the sensitivity for all classes.

            Returns:
                - sensitivity (torch.Tensor): Tensor with sensitivity for each class (single number in case of self.num_classes == 2)
        """
        tp, tn, fn, fp, total = self.get_tp_tn_fn_fp_total()

        sensitivity = tp / (tp + fn)

        # nan occurs in case of division trough 0, thus nan shall be replaced with 1 which is the maximum value
        sensitivity = torch.nan_to_num(sensitivity, nan=1)

        # check whether tp was zero (must be handled differently for binary classification)
        if self.num_classes > 2:
            for index, value in enumerate(tp):
                if value == 0:
                    # when numerator is zero, nan shall be replaced with 0 instead
                    sensitivity[index] = 0
        else:
            if tp == 0:
                sensitivity = 0

        return sensitivity

    def get_specificity(self):
        """
            Method to get the specificity for all classes.

            Returns:
                - specificity (torch.Tensor): Tensor with specificity for each class (single number in case of self.num_classes == 2)
        """
        tp, tn, fn, fp, total = self.get_tp_tn_fn_fp_total()

        specificity = tn / (fp + tn)

        # nan occurs in case of division trough 0, thus nan shall be replaced with 1 which is the maximum value
        specificity = torch.nan_to_num(specificity, nan=1)

        # check whether tn was zero (must be handled differently for binary classification)
        if self.num_classes > 2:
            for index, value in enumerate(tn):
                if value == 0:
                    # when numerator is zero, nan shall be replaced with 0 instead
                    specificity[index] = 0
        else:
            if tn == 0:
                specificity = 0

        return specificity

    def get_balanced_accuracy(self):
        """
            Method to get the balanced accuracy for all classes.

            Returns:
                - (torch.Tensor): Tensor with balanced accuracy for each class (single number in case of self.num_classes == 2)
        """
        return (self.get_sensitivity() + self.get_specificity()) / 2

    def plot(self):
        """
            Method to plot the confusion matrix.
            NOTE: This does not always work properly!
        """
        self.confmatrix.plot()
