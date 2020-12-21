# coding: utf-8
"""
Collection of helper functions
"""
import copy
import glob
import os
import os.path
import errno
import shutil
import random
import logging
from typing import Optional, List
import numpy as np
import pkg_resources

import torch
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter

from torchtext.data import Dataset
import yaml
from joeynmt.vocabulary import Vocabulary
from joeynmt.plotting import plot_heatmap
from constants import DEFAULT_ENCODING

class ConfigurationError(Exception):
    """ Custom exception for misspecifications of configuration """


def make_model_dir(model_dir: str, overwrite=False) -> str:
    """
    Create a new directory for the model.

    :param model_dir: path to model directory
    :param overwrite: whether to overwrite an existing directory
    :return: path to model directory
    """
    if os.path.isdir(model_dir):
        if not overwrite:
            raise FileExistsError(
                "Model directory exists and overwriting is disabled.")
        # delete previous directory to start with empty dir again
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    return model_dir


def make_logger(log_dir: str = None, mode: str = "train",return_log:bool=False) -> str:
    """
    Create a logger for logging the training/testing process.

    :param log_dir: path to file where log is stored as well
    :param mode: log file name. 'train', 'test' or 'translate'
    :return: joeynmt version number
    """
    logger = logging.getLogger("") # root logger
    try:
        version = pkg_resources.require("joeynmt")[0].version
    except:
        version='1.0.0'
    # add handlers only once.
    if len(logger.handlers) == 0:
        logger.setLevel(level=logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s')

        if log_dir is not None:
            if not os.path.exists(log_dir):
                make_model_dir(log_dir)
            log_file = f'{log_dir}/{mode}.log'

            fh = logging.FileHandler(log_file)
            fh.setLevel(level=logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        logger.info("This is Joey-NMT (version %s) and logged in (%s).", version,log_file)

    if return_log:
        return version,logger
    else:
        return version


def log_cfg(cfg: dict, prefix: str = "cfg") -> None:
    """
    Write configuration to log.

    :param cfg: configuration to log
    :param prefix: prefix for logging
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    for k, v in cfg.items():
        if isinstance(v, dict):
            p = '.'.join([prefix, k])
            log_cfg(v, prefix=p)
        else:
            p = '.'.join([prefix, k])
            # logger.warning("{:34s} : {}".format(p, v))
            logger.info("{:34s} : {}".format(p, v))


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Produce N identical layers. Transformer helper function.

    :param module: the module to clone
    :param n: clone this many times
    :return cloned modules
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def subsequent_mask(size: int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0


def set_seed(seed: int) -> None:
    """
    Set the random seed for modules torch, numpy and random.

    :param seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)


def log_data_info(train_data: Dataset, valid_data: Dataset, test_data: Dataset,
                  src_vocab: Vocabulary, trg_vocab: Vocabulary) -> None:
    """
    Log statistics of data and vocabulary.

    :param train_data:
    :param valid_data:
    :param test_data:
    :param src_vocab:
    :param trg_vocab:
    """
    logger = logging.getLogger(__name__)
    logger.info(
        "Data set sizes: \n\ttrain %d,\n\tvalid %d,\n\ttest %d",
            len(train_data), len(valid_data),
            len(test_data) if test_data is not None else 0)

    logger.info("First training example:\n\t[SRC] %s\n\t[TRG] %s",
        " ".join(vars(train_data[0])['src']),
        " ".join(vars(train_data[0])['trg']))

    logger.info("First 10 words (src): %s", " ".join(
        '(%d) %s' % (i, t) for i, t in enumerate(src_vocab.itos[:10])))
    logger.info("First 10 words (trg): %s", " ".join(
        '(%d) %s' % (i, t) for i, t in enumerate(trg_vocab.itos[:10])))

    logger.info("Number of Src words (types): %d", len(src_vocab))
    logger.info("Number of Trg words (types): %d", len(trg_vocab))


def load_config(path="configs/default.yaml",encoding=DEFAULT_ENCODING) -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, 'r',encoding =encoding ) as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def bpe_postprocess(string, bpe_type="subword-nmt") -> str:
    """
    Post-processor for BPE output. Recombines BPE-split tokens.

    :param string:
    :param bpe_type: one of {"sentencepiece", "subword-nmt"}
    :return: post-processed string
    """
    if bpe_type == "sentencepiece":
        ret = string.replace(" ", "").replace("▁", " ").strip()
    elif bpe_type == "subword-nmt":
        ret = string.replace("@@ ", "").strip()
    else:
        ret = string.strip()
    return ret


def store_attention_plots(attentions: np.array, targets: List[List[str]],
                          sources: List[List[str]],
                          output_prefix: str, indices: List[int],
                          tb_writer: Optional[SummaryWriter] = None,
                          steps: int = 0) -> None:
    """
    Saves attention plots.

    :param attentions: attention scores
    :param targets: list of tokenized targets
    :param sources: list of tokenized sources
    :param output_prefix: prefix for attention plots
    :param indices: indices selected for plotting
    :param tb_writer: Tensorboard summary writer (optional)
    :param steps: current training steps, needed for tb_writer
    :param dpi: resolution for images
    """
    for i in indices:
        if i >= len(sources):
            continue
        plot_file = "{}.{}.pdf".format(output_prefix, i)
        src = sources[i]
        trg = targets[i]
        attention_scores = attentions[i].T
        try:
            fig = plot_heatmap(scores=attention_scores, column_labels=trg,
                               row_labels=src, output_path=plot_file,
                               dpi=100)
            if tb_writer is not None:
                # lower resolution for tensorboard
                fig = plot_heatmap(scores=attention_scores, column_labels=trg,
                                   row_labels=src, output_path=None, dpi=50)
                tb_writer.add_figure("attention/{}.".format(i), fig,
                                     global_step=steps)
        # pylint: disable=bare-except
        except:
            print("Couldn't plot example {}: src len {}, trg len {}, "
                  "attention scores shape {}".format(i, len(src), len(trg),
                                                     attention_scores.shape))
            continue


def get_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """
    Returns the latest checkpoint (by time) from the given directory.
    If there is no checkpoint in this directory, returns None

    :param ckpt_dir:
    :return: latest checkpoint file
    """
    list_of_files = glob.glob("{}/*.ckpt".format(ckpt_dir))
    latest_checkpoint = None
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=os.path.getctime)

    # check existence
    if latest_checkpoint is None:
        raise FileNotFoundError("No checkpoint found in directory {}."
                                .format(ckpt_dir))
    return latest_checkpoint


def load_checkpoint(path: str, use_cuda: bool = True) -> dict:
    """
    Load model from saved checkpoint.

    :param path: path to checkpoint
    :param use_cuda: using cuda or not
    :return: checkpoint (dict)
    """
    assert os.path.isfile(path), "Checkpoint %s not found" % path
    checkpoint = torch.load(path, map_location='cuda' if use_cuda else 'cpu')
    return checkpoint


# from onmt
def tile(x: Tensor, count: int, dim=0) -> Tensor:
    """
    Tiles x on dimension dim count times. From OpenNMT. Used for beam search.

    :param x: tensor to tile
    :param count: number of tiles
    :param dim: dimension along which the tensor is tiled
    :return: tiled tensor
    """
    if isinstance(x, tuple):
        h, c = x
        return tile(h, count, dim=dim), tile(c, count, dim=dim)

    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
        .transpose(0, 1) \
        .repeat(count, 1) \
        .transpose(0, 1) \
        .contiguous() \
        .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def freeze_params(module: nn.Module) -> None:
    """
    Freeze the parameters of this module,
    i.e. do not update them during training

    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False


def symlink_update(target, link_name):
    try:
        os.symlink(target, link_name)
    except FileExistsError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


if __name__ == '__main__':
    version,logger = make_logger('log4', mode='train',return_log=True)
    logger.info(logger)
    del logger
    # version,logger = make_logger('log4', mode='train',return_log=False)

    # cfg_dic={'a':"AAA",'b':"BBB",'c_dict':{"c1":"C1_v","c2":"C2_v"}}
    # log_cfg(cfg_dic)

    input=np.ones((2,3))
    print(input)
    mask=subsequent_mask(3)
    print(mask)
    cfg=load_config(r"/home/zjk/fairseq_code_analysis/joeynmt/configs/small.yaml")
    print(cfg)
    log_cfg(cfg)





