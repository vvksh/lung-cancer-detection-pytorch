import logging
import argparse
from datetime import datetime

from torch.utils.data import DataLoader

import log
from dsets import LunaDataset
import torch
from torch.optim import SGD
import numpy as np
from util.utils import enumerate_with_estimate
from model import LunaModel


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

        return loss_g.mean() # recombine loss per sample into a single value

    def do_validation(self, epoch_ndx, val_dl):
        with torch.no_grad:
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

    def log_metrics(self,
                    epoch_ndx,
                    mode_str,
                    metrics_t,
                    classification_threshold=0.5):
        neg_label_mask = metrics_t[METRICS_LABEL_NDX] <= classification_threshold
        neg_pred_mask = metrics_t[METRICS_PRED_NDX] <= classification_threshold
        pos_label_mask = -neg_label_mask
        pos_pred_mask = -neg_pred_mask

        neg_count = int(neg_label_mask.sum()) # converts to a python int
        pos_count = int(pos_label_mask.sum())

        neg_correct = int((neg_label_mask & neg_pred_mask).sum())
        pos_correct = int((pos_label_mask & pos_pred_mask).sum())

        metrics_dict = {}

        metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/neg'] = metrics_t[METRICS_LOSS_NDX, neg_label_mask].mean()
        metrics_dict['loss/pos'] = metrics_t[METRICS_LOSS_NDX, pos_label_mask].mean()

        metrics_dict['correct/all'] = (pos_correct + neg_correct) / np.float32(metrics_t.shape[1]) *100

        metrics_dict['correct/neg'] = neg_correct/ np.float32(neg_count) * 100
        metrics_dict['correct/pos'] = pos_correct / np.float32(pos_count) * 100

        logger.info(
            ("E{} {:8} {loss/all:.4f} loss, "
             + "{correct/all:-5.1f}% correct, "
             ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            )
        )
        logger.info(
            ("E{} {:8} {loss/neg:.4f} loss, "
             + "{correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})"
             ).format(
                epoch_ndx,
                mode_str + '_neg',
                neg_correct=neg_correct,
                neg_count=neg_count,
                **metrics_dict,
            )
        )
        logger.info(
            ("E{} {:8} {loss/pos:.4f} loss, "
             + "{correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})"
             ).format(
                epoch_ndx,
                mode_str + '_pos',
                pos_correct=pos_correct,
                pos_count=pos_count,
                **metrics_dict,
            )
        )

        writer = getattr(self, mode_str + '_writer')

        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.total_training_samples_count)

        writer.add_pr_curve(
            'pr',
            metrics_t[METRICS_LABEL_NDX],
            metrics_t[METRICS_PRED_NDX],
            self.total_training_samples_count,
        )

        bins = [x / 50.0 for x in range(51)]

        negHist_mask = neg_label_mask & (metrics_t[METRICS_PRED_NDX] > 0.01)
        posHist_mask = pos_label_mask & (metrics_t[METRICS_PRED_NDX] < 0.99)

        if negHist_mask.any():
            writer.add_histogram(
                'is_neg',
                metrics_t[METRICS_PRED_NDX, negHist_mask],
                self.total_training_samples_count,
                bins=bins,
            )
        if posHist_mask.any():
            writer.add_histogram(
                'is_pos',
                metrics_t[METRICS_PRED_NDX, posHist_mask],
                self.total_training_samples_count,
                bins=bins,
            )


    def main(self):
        logger.info(f"Starting {self.__class__.__name__}, {self.cli_args}")
        train_dl = self.init_dataloader()
        val_dl = self.init_dataloader(is_validation=True)
        for epoch_ndx in range(self.cli_args.epochs + 1):
            logger.info(f"Starting epoch {epoch_ndx}")
            trn_metrics_t = self.do_training(epoch_ndx=epoch_ndx, train_dl=train_dl)
            self.log_metrics(epoch_ndx, 'trn', trn_metrics_t)
            val_metrics_t = self.do_validation(epoch_ndx=epoch_ndx, val_dl=val_dl)
            self.log_metrics(epoch_ndx, 'val', val_metrics_t)

if __name__ == '__main__':
    LunaTrainingApp().main()