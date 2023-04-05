import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from copy import deepcopy
from torch import nn
import pandas as pd
import os
import torch.nn.functional as F


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Complementary Learning Systems Based Experience Replay')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    # Sample Selection
    parser.add_argument('--loss_margin', type=float, default=1)
    parser.add_argument('--loss_alpha', type=float, default=0.99)
    parser.add_argument('--std_margin', type=float, default=1)
    parser.add_argument('--task_warmup', type=int, default=1)
    parser.add_argument('--track_performance', type=int, default=0)
    parser.add_argument('--monitor_clean_noisy_performance', type=int, default=0)
    parser.add_argument('--enable_lars', type=int, default=1)
    parser.add_argument('--remove_loss_outliers', type=int, default=1)
    parser.add_argument('--enable_lll', type=int, default=1)

    return parser


# =============================================================================
# Mean-ER
# =============================================================================
class ESM(ContinualModel):
    NAME = 'esm'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(ESM, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

        self.task_loss = nn.CrossEntropyLoss(reduction='none')
        self.current_task = 0
        self.global_step = 0
        self.task_iter = 0

        # Running estimates
        self.loss_running_sum = 0
        self.loss_running_mean = 0
        self.loss_running_std = 0
        self.n_samples_seen = 0

    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()
        self.net.train()

        loss = 0
        out = self.net(inputs)
        task_loss = self.task_loss(out, labels)

        ignore_mask = torch.zeros_like(labels) > 0

        if self.loss_running_mean > 0:
            sample_weight = torch.where(
                task_loss >= self.args.loss_margin * self.loss_running_mean,
                self.loss_running_mean / task_loss,
                torch.ones_like(task_loss)
            )
            ce_loss = (sample_weight.detach() * task_loss).mean()

        else:
            ce_loss = task_loss.mean()


        if hasattr(self, 'writer'):
            self.writer.add_scalar(f'Task {self.current_task}/l_ce', ce_loss.item(), self.iteration)
            self.writer.add_scalar(f'Overall/l_ce', ce_loss.item(), self.global_step)

        loss += ce_loss

        # =====================================================================
        # Apply Buffer loss
        # =====================================================================
        if not self.buffer.is_empty():

            buf_data = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, return_non_aug=True)

            if len(buf_data) == 4:
                buf_inputs, non_aug_buf_inputs, buf_labels, task_id = buf_data
            else:
                buf_inputs, non_aug_buf_inputs, buf_labels, task_id, _ = buf_data

            buf_out = self.net(buf_inputs)
            l_buf_ce = torch.mean(self.task_loss(buf_out, buf_labels))

            l_buf = l_buf_ce
            loss += l_buf

            if hasattr(self, 'writer'):
                self.writer.add_scalar(f'Task {self.current_task}/l_buf_ce', l_buf_ce.item(), self.iteration)
                self.writer.add_scalar(f'Task {self.current_task}/l_buf', l_buf.item(), self.iteration)
                self.writer.add_scalar(f'Overall/l_buf_ce', l_buf_ce.item(), self.global_step)
                self.writer.add_scalar(f'Overall/l_buf', l_buf.item(), self.global_step)

        # Log values
        if hasattr(self, 'writer'):
            self.writer.add_scalar(f'Task {self.current_task}/loss', loss.item(), self.iteration)
            self.writer.add_scalar(f'Overall/loss', loss.item(), self.global_step)

        loss.backward()
        self.opt.step()

        self.buffer.add_data(
            examples=not_aug_inputs[~ignore_mask],
            labels=labels[~ignore_mask],
            timestamps=torch.ones_like(labels[~ignore_mask]) * self.current_task,
        )

        # Update the ema model
        self.global_step += 1
        self.task_iter += 1

        if not self.warmup_phase:
            self.update_running_loss_ema(task_loss.detach())


        return loss.item()


    def update_running_loss_ema(self, batch_loss):
        alpha = min(1 - 1 / (self.global_step + 1), self.args.loss_alpha)
        self.loss_running_mean = alpha * self.loss_running_mean + (1 - alpha) * batch_loss.mean()
        self.loss_running_std = alpha * self.loss_running_std + (1 - alpha) * batch_loss.std()

    def end_task(self, dataset) -> None:
        self.current_task += 1
        self.task_iter = 0


    def begin_task(self, dataset):
        if self.args.task_warmup > 0:
            self.warmup_phase = 1
            print('Enabling Warmup phase')
        else:
            self.warmup_phase = 0

    def end_epoch(self, epoch, dataset) -> None:
        if (epoch >= self.args.task_warmup) and self.warmup_phase:
            self.warmup_phase = 0
            print('Disable Warmup phase')
