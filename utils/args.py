# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--experiment_id', type=str, default='cl')
    parser.add_argument('--backbone', type=str, default='default')
    parser.add_argument('--dataset', type=str, required=True, choices=DATASET_NAMES)
    parser.add_argument('--mnist_seed', type=int, default=0)
    parser.add_argument('--corrupt_perc', type=float, default=0)
    parser.add_argument('--return_index', type=bool, default=False)
    parser.add_argument('--tiny_imagenet_path', type=str, default='data')
    parser.add_argument('--model', type=str, required=True, help='Model name.', choices=get_all_models())
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--n_epochs', type=int, required=True)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--num_tasks', type=int, default=20)
    parser.add_argument('--cifar100_num_tasks', type=int, default=5)
    parser.add_argument('--deg_inc', type=float, default=8)
    parser.add_argument('--optim_wd', type=float, default=0.,
                        help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')

def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=None,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')

    parser.add_argument('--csv_log', action='store_true',
                        help='Enable csv logging')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--validation', action='store_true',
                        help='Test on the validation set')
    parser.add_argument('--output_dir', type=str, default='experiments')


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, required=True,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int, required=True,
                        help='The batch size of the memory buffer.')


def add_gcil_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments required for GCIL-CIFAR100 Dataset.
    :param parser: the parser instance
    """
    # arguments for GCIL-CIFAR100
    parser.add_argument('--gil_seed', type=int, default=1993, help='Seed value for GIL-CIFAR task sampling')
    parser.add_argument('--pretrain', action='store_true', default=False, help='whether to use pretrain')
    parser.add_argument('--phase_class_upper', default=50, type=int, help='the maximum number of classes')
    parser.add_argument('--epoch_size', default=1000, type=int, help='Number of samples in one epoch')
    parser.add_argument('--pretrain_class_nb', default=0, type=int, help='the number of classes in first group')
    parser.add_argument('--weight_dist', default='unif', type=str, help='what type of weight distribution assigned to classes to sample (unif or longtail)')


def add_dale_nn_args(parser: ArgumentParser) -> None:
    """
    Model Parameters for Dale NNs.
    :param parser: the parser instance
    """
    parser.add_argument('--input_units', type=int, default=28 * 28)
    parser.add_argument('--output_units', type=int, default=10)
    parser.add_argument('--network_type', type=str, default='dense')
    parser.add_argument('--layer_type', type=str, default='DANN')
    parser.add_argument('--n_e',  type=int, default=500, help='Number of excitatory units in each layer')
    parser.add_argument('--n_i',  type=int, default=50, help='Number of inhibitory units in each layer')
    parser.add_argument('--n_hidden', default=3, type=int, help='the number of classes in first group')
    parser.add_argument('--i_iid_i', default=False, type=bool)
    parser.add_argument('--c_sgd', default=True, type=bool)
    parser.add_argument('--NGD', default=False, type=bool)


def add_active_dendrites_args(parser: ArgumentParser) -> None:
    """
    Model Parameters for Active Dendrites
    :param parser: the parser instance
    """
    parser.add_argument('--input_units', type=int, default=784)
    parser.add_argument('--output_units', type=int, default=10)
    parser.add_argument('--hidden_sizes',  type=int, nargs='*', default=[2048, 2048], help='Number of units in each layer')
    parser.add_argument('--num_segments', type=int, default=10)
    parser.add_argument('--dim_context',  type=int, default=784)
    parser.add_argument('--kw',  type=bool, default=True)
    parser.add_argument('--kw_percent_on', type=float, nargs='*', default=[0.05, 0.05])
    parser.add_argument('--context_percent_on', type=float, default=0.1)
    parser.add_argument('--dendrite_weight_sparsity', type=float, default=0)
    parser.add_argument('--weight_sparsity', type=float, default=0.50)
    parser.add_argument('--weight_init', type=str, default='modified')
    parser.add_argument('--dendrite_init', type=str, default='modified')
    parser.add_argument('--freeze_dendrites', type=bool, default=False)
    parser.add_argument('--dendritic_layer_class', type=str, default='absolute_max_gating')
    parser.add_argument('--output_nonlinearity', type=str, default=None)
    parser.add_argument('--learn_context',  type=bool, default=False)
    parser.add_argument('--use_context_network', action='store_true', default=False)
    parser.add_argument('--context_net_hidden_units', type=int, nargs='*', default=[500, 500])
    parser.add_argument('--context_train_epochs', type=int, default=1)
    parser.add_argument('--context_loss_temp', type=int, default=0.07)
    parser.add_argument('--context_lr', type=float, default=0.1)


def add_dale_active_dendrites_args(parser: ArgumentParser) -> None:
    """
    Model Parameters for Active Dendrites
    :param parser: the parser instance
    """
    parser.add_argument('--input_units', type=int, default=784)
    parser.add_argument('--output_units', type=int, default=10)
    parser.add_argument('--n_e',  type=int, default=1844, help='Number of excitatory units in each layer')
    parser.add_argument('--n_i',  type=int, default=204, help='Number of inhibitory units in each layer')
    parser.add_argument('--n_hidden', default=2, type=int, help='the number of classes in first group')
    parser.add_argument('--num_segments', type=int, default=10)
    parser.add_argument('--dim_context',  type=int, default=784)
    parser.add_argument('--kw',  type=bool, default=True)
    parser.add_argument('--kw_percent_on', type=float, nargs='*', default=[0.05, 0.05])
    parser.add_argument('--context_percent_on', type=float, default=0.1)
    parser.add_argument('--dendrite_weight_sparsity', type=float, default=0)
    parser.add_argument('--weight_sparsity', type=float, default=0.50)
    parser.add_argument('--weight_init', type=str, default='modified')
    parser.add_argument('--dendrite_init', type=str, default='modified')
    parser.add_argument('--freeze_dendrites', type=bool, default=False)
    parser.add_argument('--output_nonlinearity', type=str, default=None)
    parser.add_argument('--learn_context',  action='store_true', default=False)
    parser.add_argument('--use_shunting', action='store_true', default=False)
    parser.add_argument('--output_inhib_units',  type=int, default=1)
    parser.add_argument('--i_iid_i', type=bool, default=False)
    parser.add_argument('--use_context_network', action='store_true', default=False)
    parser.add_argument('--context_net_hidden_units', type=int, nargs='*', default=[500, 500])
    parser.add_argument('--context_train_epochs', type=int, default=1)
    parser.add_argument('--context_loss_temp', type=int, default=0.07)
    parser.add_argument('--context_lr', type=float, default=0.1)
