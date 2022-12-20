import os
import torch
from util.parser import get_parse
from trainerflow import TrainerFlow
import numpy as np

def main(args):
    set_seed(args)
    args = set_gpu(args)

    print(f'==> mode is [{args.mode}] ...')
    model = TrainerFlow(args)

    if args.mode == 'meta-train':
        if args.mixup_on:
            model.mixup_meta_train()
        else:
            model.meta_train()
    elif args.mode == 'meta-test':
        model.test_predictor()
    elif args.mode == 'search':
        model.test_full_arch_and_generate_topk()
        # model.get_topk_configs()

def set_seed(args):
    # Set the random seed for reproducible experiments
    if args.seed >= 0:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)


def set_gpu(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if args.gpu == None else args.gpu
    args.gpu = int(args.gpu)
    return args


if __name__ == '__main__':
    main(get_parse())
