import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import datetime


class Trainer():
    def __init__(self, model, ds_train, ds_val, sensors, config, log_interval, run_paths, write_to_wandb):
        # store parameters as members
        self.model = model
        self.sensors = sensors
        self.config = config
        self.log_interval = log_interval
        self.run_paths = run_paths
        self.write_to_wandb = write_to_wandb

        # login to Weights & Biases
        if self.write_to_wandb:
            wandb.login()
            display_name = datetime.datetime.now().strftime(
                '%d.%m_%H:%M:%S')
            wandb.init(project="FA", entity="st177975",
                       config=self.config, name=display_name)

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

    def train(self):
        for epoch_index in range(self.config["epochs"]):
            # reset metrics at beginning of epoch
            self.running_loss = 0.0

            for step_index, (data_dict, labels) in enumerate(self.ds_train_loader, 0):

                self.train_step(data_dict, labels)

                # log metrics in log_interval
                if step_index % self.log_interval == self.log_interval - 1:
                    self.plot_metrics(epoch_index, step_index)

        print('Finished Training')

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
        self.running_loss += loss.item()

    def plot_metrics(self, epoch_index, step_index):
        # prepare step and epoch index for logging
        num_epoch = epoch_index + 1
        num_step = step_index + 1
        total_step = (num_epoch*self.steps_per_epoch) + num_step

        # prepare metrics for logging
        self.train_loss = self.running_loss / self.log_interval

        # print metrics
        print('[%d, %5d] loss: %.3f' %
              (num_epoch, num_step, self.train_loss))

        # log to wandb if configured
        if self.write_to_wandb:
            wandb.log({'Train/loss': self.train_loss}, step=total_step)

        # reset train metrics
        self.running_loss = 0.0
