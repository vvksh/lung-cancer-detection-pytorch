import logging
import argparse
from datetime import datetime

from torch.utils.data import DataLoader

import log
from dsets import LunaDataset
import torch
from torch.optim import SGD
from util.utils import enumerate_with_estimate
from model import LunaModel
import metrics

logger = log.setup_custom_logger(__name__)

METRICS_LABEL_NDX=0
METRICS_PRED_NDX=1
METRICS_LOSS_NDX=2
METRICS_SIZE=3

class LunaTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is not None:
            sys_argv = sys_argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers',
                            help='Number of worker processes for data loading',
                            default=4,
                            type=int)

        parser.add_argument('--batch-size',
                            help='Batch size to use for training',
                            default=32,
                            type=int,
                            )
        parser.add_argument('--epochs',
                            help='Number of epochs to train for',
                            default=1,
                            type=int,
                            )

        self.cli_args = parser.parse_args(sys_argv)
        # to identify the training runs
        self.time_str = datetime.now().strftime('%Y-%m-%d_%H.%M.%S')


        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = self.init_model()
        self.optimizer = self.init_optimizer()
        self.total_training_samples_count = 0
        self.trn_writer, self.val_writer = metrics.init_tensorboard_writers(self.time_str)

    def init_model(self):
        model = LunaModel()
        if self.use_cuda:
            logger.info(f"Using cuda; {torch.cuda.device_count()}")
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def init_optimizer(self):
        return SGD(self.model.parameters(), lr=0.001, momentum=0.99)

    def init_dataloader(self, is_validation=False):
        logger.info(f"Initializing dataloader with batch_size {self.cli_args.batch_size},"
                 f" validation? :{is_validation}")
        train_ds = LunaDataset(
            val_stride=10,
            isValSet_bool=is_validation
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        dataloader = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda
        )

        return dataloader

    def do_training(self, epoch_ndx, train_dl):

        self.model.train() # just sets to train mode, doesnt do the actual training

        # initialize an empty metrics array
        trn_metrics_g = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device
        )

        # set up batch looping with time estimate
        batch_iter = enumerate_with_estimate(
            train_dl,
            f"E{epoch_ndx} Training",
            start_ndx=train_dl.num_workers,
        )

        for batch_ndx, batch_tup in batch_iter:
            if batch_ndx % 100 == 0:
                if batch_ndx % 100 == 0:
                    logger.info(f"Running on batch_ndx {batch_ndx}")
                self.optimizer.zero_grad()

                loss_var = self.compute_batch_loss(
                    batch_ndx,
                    batch_tup,
                    train_dl.batch_size,
                    trn_metrics_g
                )

                # update model weight
                loss_var.backward()
                self.optimizer.step()

        self.total_training_samples_count += len(train_dl.dataset)

        return trn_metrics_g.to('cpu')

    def compute_batch_loss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, label_t, _series_list, _center_list = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        logits_g, probability_g = self.model(input_g)

        loss_func = torch.nn.CrossEntropyLoss(reduction='none') # reduction='none' gives loss per sample

        loss_g = loss_func(
            logits_g,
            label_g[:,1], # index of the one-hot-encoded class
        )

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = \
            label_g[:, 1].detach()
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = \
            probability_g[:, 1].detach()
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = \
            loss_g.detach()

        return loss_g.mean() # recombine loss per sample into a single value

    def do_validation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval() # turns off training-time behavior
            val_metrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            batch_iter = enumerate_with_estimate(
                val_dl,
                f"E{epoch_ndx} Validation",
                start_ndx = val_dl.num_workers,
            )

            for batch_ndx, batch_tup in batch_iter:
                self.compute_batch_loss(
                    batch_ndx, batch_tup, val_dl.batch_size, val_metrics_g)

        return val_metrics_g.to('cpu')



    def main(self):
        logger.info(f"Starting {self.__class__.__name__}, {self.cli_args}")
        train_dl = self.init_dataloader()
        val_dl = self.init_dataloader(is_validation=True)
        for epoch_ndx in range(self.cli_args.epochs + 1):
            logger.info(f"Starting epoch {epoch_ndx}")
            trn_metrics_t = self.do_training(epoch_ndx=epoch_ndx, train_dl=train_dl)
            metrics.log_metrics(writer=self.trn_writer,
                                epoch_ndx=epoch_ndx,
                                mode_str='trn',
                                metrics_t=trn_metrics_t,
                                total_training_samples_count=self.total_training_samples_count)
            val_metrics_t = self.do_validation(epoch_ndx=epoch_ndx, val_dl=val_dl)
            metrics.log_metrics(writer=self.val_writer,
                                epoch_ndx=epoch_ndx,
                                mode_str='val',
                                metrics_t=val_metrics_t,
                                total_training_samples_count=self.total_training_samples_count)

if __name__ == '__main__':
    LunaTrainingApp().main()