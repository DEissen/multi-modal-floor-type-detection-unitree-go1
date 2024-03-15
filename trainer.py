import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import os

# custom imports
from metrics.confusion_matrix import ConfusionMatrix
from custom_utils.utils import get_run_name_for_logging, get_git_revision_hash


class Trainer():
    """
        Trainer class to train any provided torch model based on the config_dict with ds_train and evaluate the model additionally with ds_val.
        Details about present sensors in dataset must be provided in the sensors list. 
        Results and logs of the run will be stored in dir specified by run_paths_dict.
    """

    def __init__(self, model, ds_train, ds_val, sensors, config_dict, run_paths_dict, perform_wandb_sweep):
        """
            Init method of Trainer() class.

            Parameters:
                - model (torch.nn): Model which shall be trained
                - ds_train (torch.utils.data.Dataset): Dataset to use for training
                - ds_val (torch.utils.data.Dataset): Dataset to use for evaluation during training
                - sensors (list): List containing all sensors which are present in the datasets
                - config_dict (dict): Dict containing the configuration for the training
                - run_paths_dict (dict): Dict containing all important paths for saving results and logs
                - perform_wandb_sweep (bool): Boolean flag if data shall be written to Weights & Biases for hyperparameter sweep
        """
        # store parameters as members
        self.model = model
        self.sensors = sensors
        self.config_dict = config_dict
        self.log_interval = self.config_dict["train_log_interval"]
        self.run_paths_dict = run_paths_dict
        self.perform_wandb_sweep = perform_wandb_sweep

        # member for early stopping detection
        self.best_accuracy = 0
        self.early_stopping_counter = 0

        # login to Weights & Biases
        if self.config_dict["use_wandb"]:
            wandb.login()

            # create name of run based on current time and number of sensors
            display_name = get_run_name_for_logging(sensors, model)

            wandb.init(project="MA", entity="st177975", dir=self.run_paths_dict["wandb_path"],
                       config=self.config_dict, name=display_name, resume="auto")

        # create data loader
        self.ds_train_loader = DataLoader(ds_train,
                                          batch_size=config_dict["batch_size"], shuffle=True, drop_last=True)
        self.ds_val_loader = DataLoader(ds_val,
                                        batch_size=config_dict["batch_size"], shuffle=True, drop_last=True)
        # calculate number of steps per epoch for logging
        self.steps_per_epoch = int(len(ds_train)/config_dict["batch_size"])

        # initialize loss and optimizer
        self.train_loss_object = nn.CrossEntropyLoss()
        self.val_loss_object = nn.CrossEntropyLoss()
        if self.config_dict["optimizer"].lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=config_dict["lr"])
        elif self.config_dict["optimizer"].lower() == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(
            ), lr=config_dict["lr"], momentum=config_dict["momentum"])
        else:
            raise Exception(
                f"The optimizer '{self.config_dict['optimizer']}' is not supported!")

        # initialize metrics
        self.train_confusion_matrix = ConfusionMatrix(
            self.config_dict["num_classes"])
        self.val_confusion_matrix = ConfusionMatrix(
            self.config_dict["num_classes"])

    def train(self):
        """
            Method to start the training of the model, including evaluation and logging during training.
        """
        logging.info('######### Start training #########')

        # log checked out git hash of training in run_paths logs path
        git_hash_file_path = os.path.join(
            self.run_paths_dict["logs_path"], "git_hash.txt")

        git_hash = get_git_revision_hash()
        logging.info(f"Training is done based on commit '{git_hash}'")

        with open(git_hash_file_path, mode="w") as f:
            f.write(git_hash)

        # check and log used device for training
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        logging.info(
            f"Using {device} device to train the following model:\n{self.model}")

        # set model to training mode to be sure dropout and BN layers work as expected
        self.model.train()

        # outer loop for epochs
        for epoch_index in range(self.config_dict["epochs"]):
            # reset metrics at beginning of epoch
            self.running_loss = 0.0
            self.train_confusion_matrix.reset()

            # iterate over all batches of the dataset once
            for step_index, (data_dict, labels) in enumerate(self.ds_train_loader, 0):
                # update parameters for this batch
                self.train_step(data_dict, labels)

                # log metrics in log_interval
                if step_index % self.log_interval == self.log_interval - 1:
                    self.log_metrics(epoch_index, step_index)

            # log current results of the confusion matrix for the training dataset for this epoch
            logging.info(
                f"\n[Epoch: {epoch_index:d}, Step: end] Train confusion matrix:\n{self.train_confusion_matrix.get_result()}")

            # save model after each epoch
            force_save = False
            if epoch_index + 1 == self.config_dict["epochs"]:
                # force save for last epoch
                force_save = True
            self.save_current_model(epoch_index+1, force_save)

            # ### check for early stopping
            accuracy = self.train_confusion_matrix.get_accuracy()

            # check for increased performance
            if self.best_accuracy < (accuracy - 0.05):
                self.best_accuracy = accuracy
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

            if (self.early_stopping_counter == 2 and self.best_accuracy > 0.9) or self.early_stopping_counter == 3:
                logging.info(
                    f"Accuracy did not increase by at least 5% during for {self.early_stopping_counter} epochs, thus training will be stopped now!")
                logging.info('######### Finished training #########')
                wandb.finish()
                return

        logging.info('######### Finished training #########')
        wandb.finish()

    def train_step(self, data_dict, labels):
        """
            Method for training step (parameter update, ...) for one batch.

            Parameters:
                - data_dict (dict): Batch of data which is stored as a dict where the sensor names are the key
                - labels (list): Labels for the batch
        """
        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.model(data_dict)
        loss = self.train_loss_object(outputs, labels)
        loss.backward()
        self.optimizer.step()

        # update metrics
        _, predicted = torch.max(outputs.data, 1)
        self.train_confusion_matrix.update(predicted, labels)
        self.running_loss += loss.item()

    def log_metrics(self, epoch_index, step_index):
        """
            Method for logging current results to console and wandb (if configured).

            Parameters:
                - epoch_index (int): Number of the current epoch starting at 0
                - step_index (int: Number of the current batch of this epoch starting at 0
        """
        # evaluate model with validation set
        self.validation_for_logging()

        # prepare step and epoch index for logging
        num_epoch = epoch_index
        num_step = step_index + 1
        total_step = (num_epoch*self.steps_per_epoch) + num_step

        # prepare metrics for logging
        self.train_loss = self.running_loss / self.log_interval
        # values from confusion must be averaged for better logging in case of more than 2 classes
        if self.config_dict["num_classes"] == 2:
            averageing_info_string = ""
            train_accuracy = self.train_confusion_matrix.get_accuracy()
            train_sensitivity = self.train_confusion_matrix.get_sensitivity()
            train_specificity = self.train_confusion_matrix.get_specificity()
            train_balanced_accuracy = self.train_confusion_matrix.get_balanced_accuracy()
            val_accuracy = self.val_confusion_matrix.get_accuracy()
            val_sensitivity = self.val_confusion_matrix.get_sensitivity()
            val_specificity = self.val_confusion_matrix.get_specificity()
            val_balanced_accuracy = self.val_confusion_matrix.get_balanced_accuracy()
        else:
            averageing_info_string = "Averaged "
            train_accuracy = self.train_confusion_matrix.get_accuracy()
            train_sensitivity = torch.sum(
                self.train_confusion_matrix.get_sensitivity()) / self.config_dict["num_classes"]
            train_specificity = torch.sum(
                self.train_confusion_matrix.get_specificity()) / self.config_dict["num_classes"]
            train_balanced_accuracy = torch.sum(
                self.train_confusion_matrix.get_balanced_accuracy()) / self.config_dict["num_classes"]
            val_accuracy = self.val_confusion_matrix.get_accuracy()
            val_sensitivity = torch.sum(
                self.val_confusion_matrix.get_sensitivity()) / self.config_dict["num_classes"]
            val_specificity = torch.sum(
                self.val_confusion_matrix.get_specificity()) / self.config_dict["num_classes"]
            val_balanced_accuracy = torch.sum(
                self.val_confusion_matrix.get_balanced_accuracy()) / self.config_dict["num_classes"]

        # log metrics to console/ logging file
        # create logging message by list for better readability and easier way to add new messages
        logging_message = [f'\n[Epoch: {num_epoch:d}, Step: {num_step:5d}]:',
                           f'train loss: {self.train_loss:.3f}',
                           f'train acc: {train_accuracy*100:.2f} %',
                           f'{averageing_info_string}train sensitivity: {train_sensitivity*100:.2f} %',
                           f'{averageing_info_string}train specificity: {train_specificity*100:.2f} %',
                           f'{averageing_info_string}train balanced acc: {train_balanced_accuracy*100:.2f} %',
                           f'val loss: {self.val_loss:.3f}',
                           f'val acc: {val_accuracy*100:.2f} %',
                           f'{averageing_info_string}val sensitivity: {val_sensitivity*100:.2f} %',
                           f'{averageing_info_string}val specificity: {val_specificity*100:.2f} %',
                           f'{averageing_info_string}val balanced acc: {val_balanced_accuracy*100:.2f} %',
                           f'val confusion matrix:\n{self.val_confusion_matrix.get_result()}']
        logging_message = "\n".join(logging_message)
        logging.info(logging_message)

        # log to wandb if configured
        if self.config_dict["use_wandb"] or self.perform_wandb_sweep:
            wandb.log({'Train/loss': self.train_loss,
                       "Train/acc": train_accuracy,
                       "Train/sensitivity": train_sensitivity,
                       "Train/specificity": train_specificity,
                       "Train/balanced_acc": train_balanced_accuracy,
                       "Val/loss": self.val_loss,
                       "Val/acc": val_accuracy,
                       "Val/sensitivity": val_sensitivity,
                       "Val/specificity": val_specificity,
                       "Val/balanced_acc": val_balanced_accuracy}, step=total_step)

        # reset metrics
        self.val_confusion_matrix.reset()
        self.running_loss = 0.0

    def validation_for_logging(self):
        """
            Method to evaluate model for whole validation dataset and update results in self.val_confusion_matrix.
        """
        # set model to eval mode for correct behavior of dropout and BN layers
        self.model.eval()

        # reset validation loss
        self.val_loss = 0

        with torch.no_grad():
            for (data_dict, labels) in self.ds_val_loader:
                # get predicitons
                outputs = self.model(data_dict)

                # add loss of batch to validation loss
                self.val_loss += self.val_loss_object(outputs, labels)

                # update metrics
                _, predicted = torch.max(outputs.data, 1)
                self.val_confusion_matrix.update(predicted, labels)

        # calculate average validation loss
        self.val_loss = self.val_loss / len(self.ds_val_loader)

        # set model back to training at the end of validation for further training
        self.model.train()

    def save_current_model(self, suffix_for_filename, force_save):
        """
            Method to save the state_dict of self.model to the path from run_paths_dict if the training accuracy
            is better than 60%. Saving can be enforced by setting force_save to True.

            Parameters:
                - suffix_for_filename (str): Suffix to add the to filename (e.g. epoch, ...)
                - force_save (bool): Bool to enforce saving of model independent of training accuracy
        """
        current_train_acc = self.train_confusion_matrix.get_accuracy()

        if current_train_acc < 0.6 and not force_save:
            logging.info(
                f"State_dict not save, as accuracy for training dict is below 60% = not worth it")
            return

        # create save path
        save_path = os.path.join(
            self.run_paths_dict["model_ckpts"], f"{self.model._get_name()}_{suffix_for_filename}.pt")

        # save the current state_dict of the model to save_path
        torch.save(self.model.state_dict(), save_path)

        logging.info(
            f"Stored current state_dict of the model to '{save_path}'")
