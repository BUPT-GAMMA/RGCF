import os
import random
import numpy as np
import torch
import logging
import gc
import pdb
import time
from graphgym.cmd_args import parse_args
from graphgym.config import (cfg, assert_cfg, dump_cfg,
                             update_out_dir, get_parent_dir)
from graphgym.loader import create_dataset, create_loader
from graphgym.logger import setup_printing, create_logger
from graphgym.optimizer import create_optimizer, create_scheduler
from graphgym.model_builder import create_model
from graphgym.train import train
from graphgym.sampler import get_info_dict
from graphgym.utils.agg_runs import agg_runs
from graphgym.utils.comp_budget import params_count
from graphgym.utils.device import auto_select_device
from graphgym.contrib.train import *
from graphgym.register import train_dict

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    
    # Load cmd line args
    args = parse_args()

    # Repeat for different random seeds
    for i in range(args.repeat):
        # Load config file
        t1 = time.time()
        cfg.merge_from_file(args.cfg_file)
        cfg.merge_from_list(args.opts)

        assert_cfg(cfg)
        # Set Pytorch environment
        torch.set_num_threads(cfg.num_threads)
        # if 'condense' in cfg.out_dir:
        #     fname = args.cfg_file.split('/')
        #     out_dir_parent = f'{cfg.out_dir}/{fname[-3]}/{fname[-2]}'
        #     # out_dir_parent = os.path.dirname(args.cfg_file)
        # else:
        out_dir_parent = cfg.out_dir
        cfg.seed = i+1
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        update_out_dir(out_dir_parent, args.cfg_file)

        dump_cfg(cfg)
        setup_printing()

        if cfg.skip_existing_exp:
            test_file = f'{cfg.out_dir}/test/stats.json'
            if os.path.isfile(test_file):
                logging.info('Skip existing exp!')
                continue

        # select gpu device according to dataset size
        elif cfg.dataset.name in ['yelp2020', 'amazon-movies', 'ml-1m', 'amazon-cd'] and cfg.gnn.layer_type == 'gatconv':
            auto_select_device(strategy=args.gpu_strategy, required_mem_min=16000)
        elif cfg.dataset.name in ['yelp2020', 'amazon-movies', 'ml-1m', 'amazon-cd']:
            auto_select_device(strategy=args.gpu_strategy)
        else:
            auto_select_device(strategy=args.gpu_strategy)
            # cfg.optim.max_epoch = 50
        # Set learning environment
        cfg.train.ckpt_period=50
        datasets = create_dataset()
        if cfg.model.loss_type == 'ranking':
            sample_info_dict = get_info_dict(datasets)
        else:
            sample_info_dict = None
        loaders = create_loader(datasets)
        meters = create_logger(datasets, loaders)
        model = create_model(datasets)
        optimizer = create_optimizer(model.parameters())
        scheduler = create_scheduler(optimizer)
        # Print model info
        logging.info(model)
        logging.info(cfg)
        logging.info(cfg.out_dir)
        cfg.params = params_count(model)
        logging.info('Num parameters: {}'.format(cfg.params))
        # Start training
        if cfg.train.mode == 'standard':
            train(meters, loaders, model, optimizer, scheduler, args.exp_num, sample_info_dict)
        else:
            train_dict[cfg.train.mode](
                meters, loaders, model, optimizer, scheduler, sample_info_dict)
        #logging.info(f"one time train time{time.time()-t1}")
        logging.info(f'sum time one train{time.time() - t1}')
    # Aggregate results from different seeds
    agg_runs(get_parent_dir(out_dir_parent, args.cfg_file), cfg.metric_best)
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, '{}_done'.format(args.cfg_file))
    if os.path.exists('{}/ckpt'.format(cfg.out_dir)):
        for i in os.listdir('{}/ckpt'.format(cfg.out_dir)):
            os.remove(os.path.join('{}/ckpt'.format(cfg.out_dir), i))
            print('dont save model because of device space')
