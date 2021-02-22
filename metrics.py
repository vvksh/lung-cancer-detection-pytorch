import os

import log
from training import METRICS_PRED_NDX, METRICS_LOSS_NDX, METRICS_LABEL_NDX, METRICS_SIZE
from torch.utils.tensorboard import SummaryWriter
import numpy as np

logger = log.setup_custom_logger(__name__)

RUNS_DIR = 'runs'


def init_tensorboard_writers(time_str):
    log_dir = os.path.join(RUNS_DIR, time_str)
    trn_writer = SummaryWriter(log_dir=log_dir + '-trn_cls')
    val_writer = SummaryWriter(log_dir=log_dir + '-val_cls')
    return trn_writer, val_writer


def log_metrics(writer,
                epoch_ndx,
                mode_str,
                metrics_t,
                total_training_samples_count,
                classification_threshold=0.5,
                ):
    neg_label_mask = metrics_t[METRICS_LABEL_NDX] <= classification_threshold
    neg_pred_mask = metrics_t[METRICS_PRED_NDX] <= classification_threshold
    pos_label_mask = ~neg_label_mask
    pos_pred_mask = ~neg_pred_mask

    neg_count = int(neg_label_mask.sum())  # converts to a python int
    pos_count = int(pos_label_mask.sum())

    neg_correct = int((neg_label_mask & neg_pred_mask).sum())
    pos_correct = int((pos_label_mask & pos_pred_mask).sum())

    metrics_dict = {}

    metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_NDX].mean()
    metrics_dict['loss/neg'] = metrics_t[METRICS_LOSS_NDX, neg_label_mask].mean()
    metrics_dict['loss/pos'] = metrics_t[METRICS_LOSS_NDX, pos_label_mask].mean()

    metrics_dict['correct/all'] = (pos_correct + neg_correct) / np.float32(metrics_t.shape[1]) * 100

    metrics_dict['correct/neg'] = neg_correct / np.float32(neg_count) * 100
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

    for key, value in metrics_dict.items():
        writer.add_scalar(key, value, total_training_samples_count)

    writer.add_pr_curve(
        'pr',
        metrics_t[METRICS_LABEL_NDX],
        metrics_t[METRICS_PRED_NDX],
        total_training_samples_count,
    )

    bins = [x / 50.0 for x in range(51)]

    negHist_mask = neg_label_mask & (metrics_t[METRICS_PRED_NDX] > 0.01)
    posHist_mask = pos_label_mask & (metrics_t[METRICS_PRED_NDX] < 0.99)

    if negHist_mask.any():
        writer.add_histogram(
            'is_neg',
            metrics_t[METRICS_PRED_NDX, negHist_mask],
            total_training_samples_count,
            bins=bins,
        )
    if posHist_mask.any():
        writer.add_histogram(
            'is_pos',
            metrics_t[METRICS_PRED_NDX, posHist_mask],
            total_training_samples_count,
            bins=bins,
        )
