import argparse

def str2bool(v):
    return v.lower() in ['t', 'true', True]

def str2list(v):
    if isinstance(v, list):
        return v
    else:
        return [item for item in v.split(',')]


def get_parse():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--space_embedding_dim', type=int, help='space architecture embedding dim', default=16)
    parser.add_argument('--support_split_ratio', type=float, help='space architecture', default=1.0)

    parser.add_argument('--gpu', type=str, default='0', help='set visible gpus')
    parser.add_argument('--seed', type=int, default=3, help='set seed')
    parser.add_argument('--mode', type=str, default='meta-train', help='meta-train|meta-test')
    parser.add_argument('--main_path', type=str, default='.')
    parser.add_argument('--data_path', type=str, default='./data_process/test_best.csv')
    parser.add_argument('--dataset_path', type=str, default='./data_process/result.txt')
    parser.add_argument('--output_path', type=str, default='task_num')
    parser.add_argument('--metrics', type=str2list, default=["spearman"],
                        help="metric for ranking correlation between real and estimated latencies of architectures.")
    parser.add_argument('--load_path', type=str, default='result_rank/checkpoint/max_corr.pt',
                        help='model checkpoint path')
    # Data & Meta-learning Settings
    parser.add_argument('--meta_train_datasets', type=str2list,
                        default='yelp2020,yahoo,amazon-cd,flixster,amazon-movies,douban,amazon-beauty,ml-100k,ml-1m')
    # yelp2020,yahoo,amazon-cd,flixster,amazon-movies,douban,amazon-beauty,ml-100k,ml-1m
    parser.add_argument('--meta_valid_datasets', type=str2list,
                        default='amazon-beauty')
    parser.add_argument('--meta_test_datasets', type=str2list,
                        default='amazon-sports,epinions,beeradvocate')
    parser.add_argument('--num_inner_tasks', type=int, default=9, help="the number of meta-batch")

    parser.add_argument('--num_samples', type=int, default=8, help="the number of training samples for each task")
    parser.add_argument('--num_query', type=int, default=20, help="the number of test samples for each task")
    parser.add_argument('--meta_lr', type=float, default=1e-4, help="meta-learning rate")
    parser.add_argument('--num_episodes', type=int, default=1500, help="the number of episodes during meta-training")
    parser.add_argument('--num_train_updates', type=int, default=2,
                        help="the number of inner gradient step during meta-training")
    parser.add_argument('--num_eval_updates', type=int, default=0,
                        help="the number of inner gradient step during meta-test")
    parser.add_argument('--alpha_on', type=str2bool, default=False, help="True:Meta-SGD/MAML")
    parser.add_argument('--inner_lr', type=float, default=1e-3, help="inner learning rate for MAML")
    parser.add_argument('--second_order', type=str2bool, default=True,
                        help="on/off computing second order gradient of bilevel optimization framework (MAML framework)")


    # Save / Log
    parser.add_argument('--save_path', type=str, default='results', help='')
    parser.add_argument('--save_summary_steps', type=int, default=50, help="the interval to print log")
    parser.add_argument('--arch_save_path', type=str, default='arch_path', help='')
    #encoder
    parser.add_argument('--dataset_on', type=str2bool, default=True, help="on/off dataset modulator")
    parser.add_argument('--dot_on', type=str2bool, default=False, help="on/off dataset modulator")
    parser.add_argument('--arch_embed_dim', type=int, default=16)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--cross_hid_dim', type=int, default=32)
    parser.add_argument('--dataset_hid_dim', type=int, default=16)
    parser.add_argument('--dataset_out_dim', type=int, default=16)
    parser.add_argument('--rank_on',  type=str2bool, default=True, help="Ranking loss")

    parser.add_argument('--space_dims', type=str2list, default="init_dim,msg,gnn_layer,layers_num,stage,inter_func,act,cpnt_num,cpnt_aggr", help="space dim")
    parser.add_argument('--performance_name', type=str, default='rmse', help='rmse|recall')
    parser.add_argument('--feature_type', type=str, default='full',
                        help="full|distribution|graph|bipartite")

    parser.add_argument('--alpha', type=float, default=0.5, help="parameter of mix up beta distribution")
    parser.add_argument('--mixup_on', type=str2bool, default=True, help="Manifold mix up")

    parser.add_argument('--patience', type=int, default=20, help="patience for early stopping")

    parser.add_argument('--task', type=str, default='rp', help='rp|ir')
    parser.add_argument('--load_pretrain', type=str2bool, default=False, help="on/off pretrain architecture embedding in search phrase")
    parser.add_argument('--lamda', type=float, default=0.6, help="trade off of listwise and pairwise")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parse()

    print(str(args))
