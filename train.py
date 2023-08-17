import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import datetime
import logging

# custom imports
from metrics.confusion_matrix import ConfusionMatrix


class Trainer():
    def __init__(self, model, ds_train, ds_val, sensors, config, log_interval, run_paths_dict, write_to_wandb):
        # store parameters as members
        self.model = model
        self.sensors = sensors
        self.config = config
        self.log_interval = log_interval
        self.run_paths_dict = run_paths_dict
        self.write_to_wandb = write_to_wandb

        # login to Weights & Biases
        if self.write_to_wandb:
            wandb.login()
            display_name = datetime.datetime.now().strftime(
                '%d.%m_%H:%M:%S')
            wandb.init(project="FA", entity="st177975", dir=self.run_paths_dict["wandb_path"],
                       config=self.config, name=display_name, resume="auto")

        # create data loader
        self.ds_train_loader = DataLoader(ds_train,
                                          batch_size=config["batch_size"], shuffle=True, drop_last=True)
        self.ds_val_loader = DataLoader(ds_val,
                                        batch_size=config["batch_size"], shuffle=True, drop_last=True)
        # calculate number of steps per epoch for logging
        self.steps_per_epoch = int(len(ds_train)/config["batch_size"])

        # loss and optimizer
        self.loss_object = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(
        ), lr=config["lr"], momentum=config["momentum"])

        # metrics
        self.train_confusion_matrix = ConfusionMatrix(
            self.config["num_classes"])
        self.val_confusion_matrix = ConfusionMatrix(self.config["num_classes"])

    def train(self):
        logging.info('######### Start training #########')
        for epoch_index in range(self.config["epochs"]):
            # reset metrics at beginning of epoch
            self.running_loss = 0.0
            self.train_confusion_matrix.reset()

            for step_index, (data_dict, labels) in enumerate(self.ds_train_loader, 0):

                self.train_step(data_dict, labels)

                # log metrics in log_interval
                if step_index % self.log_interval == self.log_interval - 1:
                    self.log_metrics(epoch_index, step_index)

            logging.info(
                f"\n[Epoch: {epoch_index+1:d}, Step: end] Train confusion matrix:\n{self.train_confusion_matrix.get_result()}")

        logging.info('######### Finished training #########')

    def train_step(self, data_dict, labels):
        # prepare data_dict for model
        if len(self.sensors) == 1:
            # extract input for uni-modal case
            inputs = data_dict[self.sensors[0]]
        else:
            # TODO: extract input for multi-modal case
            pass

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.model(inputs)
        loss = self.loss_object(outputs, labels)
        loss.backward()
        self.optimizer.step()

        # update metrics
        _, predicted = torch.max(outputs.data, 1)
        self.train_confusion_matrix.update(predicted, labels)
        self.running_loss += loss.item()

    def validation_for_logging(self):
        with torch.no_grad():
            for (data_dict, labels) in self.ds_val_loader:
                # prepare data_dict for model
                if len(self.sensors) == 1:
                    # extract input for uni-modal case
                    inputs = data_dict[self.sensors[0]]
                else:
                    # TODO: extract input for multi-modal case
                    pass

                # get predicitons
                outputs = self.model(inputs)

                # update metrics
                _, predicted = torch.max(outputs.data, 1)
                self.val_confusion_matrix.update(predicted, labels)

    def log_metrics(self, epoch_index, step_index):
        # get results for validation set
        self.validation_for_logging()

        # prepare step and epoch index for logging
        num_epoch = epoch_index + 1
        num_step = step_index + 1
        total_step = (num_epoch*self.steps_per_epoch) + num_step

        # prepare metrics for logging
        self.train_loss = self.running_loss / self.log_interval
        train_accuracy = self.train_confusion_matrix.get_accuracy()
        train_sensitivity = self.train_confusion_matrix.get_sensitivity()
        train_specificity = self.train_confusion_matrix.get_specificity()
        val_accuracy = self.val_confusion_matrix.get_accuracy()
        val_sensitivity = self.val_confusion_matrix.get_sensitivity()
        val_specificity = self.val_confusion_matrix.get_specificity()

        if self.config["num_classes"] == 2:
            train_balanced_accuracy = self.train_confusion_matrix.get_balanced_accuracy()
            val_balanced_accuracy = self.val_confusion_matrix.get_balanced_accuracy()
        else:
            train_balanced_accuracy = torch.sum(
                self.train_confusion_matrix.get_balanced_accuracy()) / self.config["num_classes"]
            val_balanced_accuracy = torch.sum(
                self.val_confusion_matrix.get_balanced_accuracy()) / self.config["num_classes"]

        # log metrics to console/ logging file
        logging_message = f'\n[Epoch: {num_epoch:d}, Step: {num_step:5d}]:\n' + f'loss: {self.train_loss:.3f}\n' + f'train acc: {train_accuracy*100:.2f} %\n' + f'train sensitivity: {train_sensitivity*100:.2f} %\n' + f'train specificity: {train_specificity*100:.2f} %\n' + f'train balanced acc: {train_balanced_accuracy*100:.2f} %\n' + \
            f'val acc: {val_accuracy*100:.2f} %\n' + f'val sensitivity: {val_sensitivity*100:.2f} %\n' + f'val specificity: {val_specificity*100:.2f} %\n' + \
            f'val balanced acc: {val_balanced_accuracy*100:.2f} %\n' + \
            f'val confusion matrix:\n{self.val_confusion_matrix.get_result()}'
        logging.info(logging_message)

        # log to wandb if configured
        if self.write_to_wandb:
            wandb.log({'Train/loss': self.train_loss,
                       "Train/acc": train_accuracy,
                       "Train/sensitivity": train_sensitivity,
                       "Train/specificity": train_specificity,
                       "Train/balanced_acc": train_balanced_accuracy,
                       "Val/acc": val_accuracy,
                       "Val/sensitivity": val_sensitivity,
                       "Val/specificity": val_specificity,
                       "Val/balanced_acc": val_balanced_accuracy}, step=total_step)

        # reset metrics
        self.val_confusion_matrix.reset()
        self.running_loss = 0.0
