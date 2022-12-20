import os
import logging
from collections import OrderedDict
from collections import defaultdict
import yaml
from tqdm import tqdm
import json
from data_process.loader import Data
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F
from model.meta_learner import MetaLearner
from model.dataset_encoder import MetaFeatureReconstruct
from torch.utils.tensorboard import SummaryWriter
from util.util import metrics_fn, loss_fn, Log, set_logger, EarlyStopping, mix_up
from torch.nn.utils import clip_grad_norm
import numpy as np
import random

class TrainerFlow:
    def __init__(self, args):
        self.args = args
        self.mode = args.mode
        self.dataset_path = args.dataset_path
        self.data_path = args.data_path

        self.metrics = args.metrics
        # self.load_path = args.load_path
        self.save_summary_steps = args.save_summary_steps
        self.save_path = args.save_path
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.load_path = args.load_path
        # Data & Meta-learning Settings
        self.meta_train_datasets = args.meta_train_datasets
        self.meta_valid_datasets = args.meta_valid_datasets
        self.meta_test_datasets = args.meta_test_datasets
        self.num_inner_tasks = args.num_inner_tasks
        self.num_samples = args.num_samples
        self.num_query = args.num_query
        self.meta_lr = args.meta_lr

        self.num_episodes = args.num_episodes
        self.num_train_updates = args.num_train_updates
        self.num_eval_updates = args.num_eval_updates
        self.alpha_on = args.alpha_on
        self.inner_lr = args.inner_lr
        self.second_order = args.second_order
        self.arch_embed_dim = args.arch_embed_dim
        self.dataset_on = args.dataset_on
        self.dot_on = args.dot_on
        self.lamda = args.lamda
        self.n_heads = args.n_heads
        self.n_layers = args.n_layers
        self.cross_hid_dim = args.cross_hid_dim
        self.dataset_hid_dim = args.dataset_hid_dim
        self.dataset_out_dim = args.dataset_out_dim
        self.space_dims = args.space_dims
        self.performance_name = args.performance_name
        self.loss_fn = loss_fn['mse']
        self.feature_type = args.feature_type
        self.mu = 0.2
        self.reconstruct_loss = nn.MSELoss()
        self.output_path = args.output_path
        self.task = args.task
        self.load_pretrain = args.load_pretrain
        self.list_wise_loss = loss_fn['listMLE']
        if args.rank_on:
            if args.mixup_on :
                self.loss_fn_train = loss_fn['rank_bpr_mix']
            else:
                self.loss_fn_train = loss_fn['rank_bpr']
        else:
            self.loss_fn_train = loss_fn['mse']
        # Meta-learner

        self.arch_save_path = args.arch_save_path
        self.alpha = args.alpha

        self.patience = args.patience
        self.early_stopping = EarlyStopping(patience=self.patience, save_path=self.save_path)
        # Data

        self.data = Data(args.mode,
                         args.data_path,
                         args.dataset_path,
                         args.meta_train_datasets,
                         args.meta_valid_datasets,
                         args.meta_test_datasets,
                         args.num_inner_tasks,
                         args.support_split_ratio,
                         args.num_samples,
                         args.space_dims,
                         args.performance_name,
                         args.feature_type
                         )
        self.arch_num = self.data.space_dim_dict['sum_dim']
        self.dataset_input_dim = self.data.dataset_input_dim
        # Model
        if args.dataset_on == False:
            self.model = MetaLearner(self.arch_num,
                                     args.arch_embed_dim,
                                     args.hidden_dim,
                                     args.n_heads,
                                     args.n_layers,
                                     args.cross_hid_dim,
                                     dataset_on=args.dataset_on,
                                     dot_on=args.dot_on).cuda()
        else:
            self.model = MetaLearner(self.arch_num,
                                     args.arch_embed_dim,
                                     args.hidden_dim,
                                     args.n_heads,
                                     args.n_layers,
                                     args.cross_hid_dim,
                                     dataset_input_dim=self.dataset_input_dim,
                                     dataset_hid_dim=args.dataset_hid_dim,
                                     dataset_out_dim=args.dataset_out_dim,
                                     dot_on=args.dot_on).cuda()
        if self.mu:
            self.reconstruct_model = MetaFeatureReconstruct(self.dataset_input_dim, args.dataset_hid_dim, args.dataset_out_dim)
        self.model_params = list(self.model.parameters())
        if self.alpha_on:
            self.define_task_lr_params()
            self.model_params += list(self.task_lr.values())
        else:
            self.task_lr = None

        if self.mode == 'meta-train':
            set_logger(os.path.join(self.save_path, 'log.txt'))
            writer = SummaryWriter(log_dir=self.save_path)
            self.log = {
                'meta_train': Log(self.save_path,
                                  self.save_summary_steps,
                                  self.metrics,
                                  self.meta_train_datasets,
                                  'meta_train',
                                  writer),
                'meta_test': Log(self.save_path,
                                  self.save_summary_steps,
                                  self.metrics,
                                  self.meta_test_datasets,
                                  'meta_test',
                                  writer,
                                  ),
            }
            self.meta_optimizer = torch.optim.Adam(self.model_params, lr=self.meta_lr,weight_decay=0.00001)
            # self.scheduler = CosineAnnealingWarmRestarts(optimizer=self.meta_optimizer, T_0=50, T_mult=2)
            self.scheduler = None


    def define_task_lr_params(self):
        self.task_lr = OrderedDict()
        for key, val in self.model.named_parameters():
            self.task_lr[key] = nn.Parameter(
                1e-3 * torch.ones_like(val))


    def train_single_task(self, dataset_emb, xs, ys, num_updates):
        self.model.train()
        if self.dataset_on:
            dataset_emb = dataset_emb.cuda()
        xs, ys = xs.cuda(), ys.cuda()
        adapted_params = self.model.cloned_params()
        for n in range(num_updates):
            if self.dataset_on:
                ys_hat, _ = self.model(xs, dataset_emb, adapted_params, inner_loop=True)
            else:
                ys_hat = self.model(xs, adapted_params=adapted_params, inner_loop=True)
            loss = self.loss_fn_train(ys_hat, ys)
            grads = torch.autograd.grad(
                loss, adapted_params.values(), create_graph=(self.second_order), allow_unused=True)

            for (key, val), grad in zip(adapted_params.items(), grads):
                if self.task_lr is not None:  # Meta-SGD
                    task_lr = self.task_lr[key]
                else:
                    task_lr = self.inner_lr  # MAML
                if grad is not None:
                    adapted_params[key] = val - task_lr * grad
        return adapted_params


    def mixup_inner_loop(self, support_list, lam_list, inter_layer_list, num_updates, inter_index):
        adapted_state_dicts = []
        self.model.train()
        for i, j in enumerate(inter_index):
            adapted_params = self.model.cloned_params()
            ys_i = support_list[1][i].cuda()
            ys_j = support_list[1][j].cuda()
            xs_i = support_list[0][i].cuda()
            xs_j = support_list[0][j].cuda()
            dataset_emb_i = support_list[2][i]
            dataset_emb_j = support_list[2][j]
            for n in range(num_updates):
                if self.dataset_on:
                    ys_hat, _ = self.model.mix_forward(xs_i, xs_j, dataset_emb_i, dataset_emb_j, lam_list[i], inter_layer_list[i],adapted_params=adapted_params, inner_loop=True)
                else:
                    ys_hat, _ = self.model.mix_no_feature_forward(xs_i, xs_j,  lam_list[i],
                                                    inter_layer_list[i], adapted_params=adapted_params)
                loss = self.loss_fn_train(ys_hat, ys_i, ys_j, lam_list[i])
                grads = torch.autograd.grad(
                    loss, adapted_params.values(), create_graph=(self.second_order), allow_unused=True)

                for (key, val), grad in zip(adapted_params.items(), grads):
                    if self.task_lr is not None:  # Meta-SGD
                        task_lr = self.task_lr[key]
                    else:
                        task_lr = self.inner_lr  # MAML
                    if grad is not None:
                        adapted_params[key] = val - task_lr * grad
            adapted_state_dicts.append(adapted_params)
        return adapted_state_dicts

    def meta_train(self):
        print('==> start training...')
        # max_valid_corr = -1
        stop = False
        with tqdm(total=self.num_episodes) as t:
            for i_epi in range(self.num_episodes):
                # Run inner loops to get adapted parameters (theta_t`)
                adapted_state_dicts = []
                query_list_x = []
                query_list_y = []
                dataset_emb_list = []
                query_list = []
                episode = self.data.generate_episode()
                for i_task in range(self.num_inner_tasks):
                    # Perform a gradient descent to meta-learner on the task
                    (dataset_emb, xs, ys, xq, yq, _) = episode[i_task]
                    if self.task == 'rp':
                        ys = -ys
                        yq = -yq
                    if not self.dataset_on :
                        dataset_emb = None
                    query_list_x.append(xq)
                    query_list_y.append(yq)
                    dataset_emb_list.append(dataset_emb.cuda())
                    adapted_state_dict = self.train_single_task(dataset_emb, xs, ys, self.num_train_updates)
                    # Store adapted parameters
                    # Store dataloaders for meta-update and evaluation
                    adapted_state_dicts.append(adapted_state_dict)
                    query_list.append((dataset_emb, xq, yq))
                query_list_wise = [query_list_x, query_list_y, dataset_emb_list]
                # Update the parameters of meta-learner
                # Compute losses with adapted parameters along with corresponding tasks
                # Updated the parameters of meta-learner using sum of the losses
                meta_loss = 0
                for i_task in range(self.num_inner_tasks):
                    dataset_emb, xq, yq = query_list[i_task]
                    xq, yq = xq.cuda(), yq.cuda()
                    if self.dataset_on:
                        dataset_emb = dataset_emb.cuda()

                    adapted_state_dict = adapted_state_dicts[i_task]
                    yq_hat, re_loss = self.model(xq, dataset_emb, adapted_state_dict)
                    loss_t = self.loss_fn_train(yq_hat, yq)
                    if re_loss:
                        re_loss = 0
                        loss_t = loss_t+self.mu*re_loss
                    meta_loss += loss_t / float(self.num_inner_tasks)
                meta_loss = self.lamda * meta_loss + (1 - self.lamda) * self.calculate_list_wise_loss(query_list_wise)
                self.meta_optimizer.zero_grad()
                meta_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.meta_optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step(meta_loss)

                # Evaluate model on new tasks
                # Evaluate on train and test dataset given a number of tasks (args.num_steps)
                if (i_epi + 1) % self.save_summary_steps == 0:
                    logging.info(f"Episode {i_epi + 1}/{self.num_episodes}")
                postfix = {}
                for split in ['meta_train', 'meta_test']:
                    msg = f"[{split.upper()}] "
                    self._test_predictor(split, i_epi)
                    self.log[split].update_epi(i_epi)
                    for m in self.metrics + ['mse_loss']:
                        v = self.log[split].avg(i_epi, m)
                        postfix[f'{split}/{m}'] = f'{v:05.3f}'
                        msg += f"{m}: {v:05.3f}; "

                        # if m == 'spearman' and max_valid_corr < v:
                        #     max_valid_corr = v
                        if m == 'spearman' and split == 'meta_train':
                            save_dict = {'epi': i_epi,
                                         'model': self.model.cpu().state_dict()}

                            if self.args.alpha_on:
                                save_dict['task_lr'] = {k: v.cpu() for k, v in self.task_lr.items()}
                                for k, v in self.task_lr.items():
                                    self.task_lr[k].cuda()
                            stop = self.early_stopping.step(score=v, save_dict=save_dict)

                    self.model.cuda()
                    logging.info(msg)
                    t.set_postfix(postfix)
                    print('\n')
                t.update()
                if stop:
                    print('Early Stop!\tEpoch:' + str(episode))
                    break
        with open(os.path.join(self.save_path,'hp_msg.log'),'w') as f:
            f.write(str(self.args))
        print('==> Training done')


    def mixup_meta_train(self):
        print('==> start task manifold mix up training...')
        # max_valid_corr = -1
        stop = False
        with tqdm(total=self.num_episodes) as t:
            for i_epi in range(self.num_episodes):
                # Run inner loops to get adapted parameters (theta_t`)
                query_list_x = []
                query_list_y = []
                support_list_x = []
                support_list_y = []
                dataset_emb_list = []
                inter_layer_list = []
                lam_list = []
                episode = self.data.generate_episode()
                for i_task in range(self.num_inner_tasks):
                    (dataset_emb, xs, ys, xq, yq, _) = episode[i_task]
                    if self.task == 'rp':
                        ys = -ys
                        yq = -yq
                    support_list_x.append(xs)
                    support_list_y.append(ys)
                    query_list_x.append(xq)
                    query_list_y.append(yq)
                    dataset_emb_list.append(dataset_emb.cuda())
                    # support_list = [support_list_x, support_list_y, dataset_emb_list]
                    lam = np.random.beta(self.alpha, self.alpha)
                    inter_layer_list.append(random.randint(0, 2))
                    # query_list = [query_list_x, query_list_y, dataset_emb_list]
                    lam_list.append(lam)
                support_list = [support_list_x, support_list_y, dataset_emb_list]
                query_list = [query_list_x, query_list_y, dataset_emb_list]
                interpolation_index = torch.randperm(self.num_inner_tasks)
                adapted_state_dicts = self.mixup_inner_loop(support_list, lam_list, inter_layer_list, self.num_train_updates, interpolation_index)

                query_list.append((dataset_emb, xq, yq))

                meta_loss = 0
                for i, j in enumerate(interpolation_index):
                    xq_i = query_list[0][i].cuda()
                    xq_j = query_list[0][j].cuda()
                    dataset_emb_i = query_list[2][i]
                    dataset_emb_j = query_list[2][j]
                    yq_i = query_list[1][i].cuda()
                    yq_j = query_list[1][j].cuda()
                    adapted_state_dict = adapted_state_dicts[i]
                    if self.dataset_on:
                        yq_hat, re_loss = self.model.mix_forward(xq_i, xq_j, dataset_emb_i, dataset_emb_j, lam_list[i], inter_layer_list[i],
                                               adapted_params=adapted_state_dict, inner_loop=True)

                    else:
                        yq_hat, re_loss = self.model.mix_no_feature_forward(xq_i, xq_j, lam_list[i],
                                                        inter_layer_list[i],
                                                        adapted_params=adapted_state_dict)

                    loss_t = self.loss_fn_train(yq_hat, yq_i, yq_j,lam_list[i])
                    if re_loss:
                        re_loss = 0
                        loss_t = loss_t + self.mu * re_loss

                    meta_loss += loss_t / float(self.num_inner_tasks)
                meta_loss = self.lamda * meta_loss + (1-self.lamda) * self.calculate_list_wise_loss(query_list)
                # a = a + 0.6/self.num_episodes
                # b = b - 0.6/self.num_episodes

                self.meta_optimizer.zero_grad()
                meta_loss.backward()
                nn.utils.clip_grad_norm(self.model.parameters(), 1)
                self.meta_optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step(meta_loss)

                # Evaluate model on new tasks
                # Evaluate on train and test dataset given a number of tasks (args.num_steps)
                if (i_epi + 1) % self.save_summary_steps == 0:
                    logging.info(f"Episode {i_epi + 1}/{self.num_episodes}")
                postfix = {}
                for split in ['meta_train', 'meta_test']:
                    msg = f"[{split.upper()}] "
                    self._test_predictor(split, i_epi)
                    self.log[split].update_epi(i_epi)
                    for m in self.metrics + ['mse_loss']:
                        v = self.log[split].avg(i_epi, m)
                        postfix[f'{split}/{m}'] = f'{v:05.3f}'
                        msg += f"{m}: {v:05.3f}; "

                        if m == 'spearman' and split == 'meta_train':
                            # max_valid_corr = v
                            save_dict = {'epi': i_epi,
                                         'model': self.model.cpu().state_dict()}
                            if self.args.alpha_on:
                                save_dict['task_lr'] = {k: v.cpu() for k, v in self.task_lr.items()}
                                for k, v in self.task_lr.items():
                                    self.task_lr[k].cuda()
                            stop = self.early_stopping.step(v, save_dict)

                            # save_path = os.path.join(self.save_path, 'checkpoint', f'max_corr.pt')
                            # if not os.path.exists(save_path):
                            #     os.makedirs(os.path.join(self.save_path, 'checkpoint'))
                            # torch.save(save_dict, save_path)
                            # print(f'==> save {save_path}')
                    self.model.cuda()
                    logging.info(msg)
                    t.set_postfix(postfix)
                    print('\n')
                t.update()
                if stop:
                    print('Early Stop!\tEpoch:' + str(i_epi))
                    break
        with open(os.path.join(self.save_path,'hp_msg.log'),'w') as f:
            f.write(str(self.args))
        print('==> Training done')

    def calculate_list_wise_loss(self, query_list):
        batch_pred = []
        batch_ground_truth = []
        for i in range(self.num_inner_tasks):
            arch = query_list[0][i].cuda()
            ground_truth = query_list[1][i].cuda()
            dataset_emb = query_list[2][i].cuda()
            top_index = torch.topk(ground_truth.squeeze(), 5, largest=True).indices
            top_ground_truth = ground_truth[top_index]
            pred, _ = self.model(arch[top_index], dataset_emb)
            pred = pred.squeeze()
            batch_pred.append(pred)
            batch_ground_truth.append(top_ground_truth.squeeze())
        list_loss = self.list_wise_loss(batch_pred, batch_ground_truth)
        return list_loss

    def test_predictor(self):
        loaded = torch.load(self.load_path)
        print(f'==> load {self.load_path}')
        if 'epi' in loaded.keys():
            epi = loaded['epi']
            print(f'==> load {epi} model..')
        self.model.load_state_dict(loaded['model'])

        if self.alpha_on:
            for (k, v), (lk, lv) in zip(self.task_lr.items(), loaded['task_lr'].items()):
                self.task_lr[k] = lv.cuda()

        self._test_predictor('meta_test', None)

    def _test_predictor(self, split, i_epi=None):
        save_file_path = os.path.join(self.save_path, f'test_log.txt')
        f = open(save_file_path, 'a+')

        avg_metrics = {m: 0.0 for m in self.metrics}
        avg_metrics['mse_loss'] = 0.0

        tasks = self.data.generate_test_tasks(split)
        for (dataset_emb, xs, ys, xq, yq, dataset) in tasks:
            if self.task == 'rp':
                ys = -ys
                yq = -yq
            adapted_state_dict = \
                self.train_single_task(dataset_emb, xs, ys, self.num_eval_updates)

            xq, yq = xq.cuda(), yq.cuda()
            if self.dataset_on:
                dataset_emb = dataset_emb.cuda()
            else:
                dataset_emb = None
            yq_hat, _ = self.model(xq, dataset_emb, adapted_state_dict)
            loss = self.loss_fn(yq_hat, yq)
            if i_epi is not None:
                for metric in self.metrics:
                    self.log[split].update(i_epi, metric, dataset,
                                           val=metrics_fn[metric](yq_hat, yq)[0])

                    print(f'{dataset}_{metric}_{metrics_fn[metric](yq_hat, yq)[0]}')
                self.log[split].update(i_epi, 'mse_loss', dataset, val=loss.item())

            else:
                msg = f'[{split}/{dataset}] '
                for m in self.metrics:
                    msg += f'{m} {metrics_fn[m](yq_hat, yq)[0]:.3f} '
                    avg_metrics[m] += metrics_fn[m](yq_hat, yq)[0]
                msg += f'MSE {loss.item():.3f}'
                avg_metrics['mse_loss'] += loss.item()
                rank = torch.argsort(yq, dim=0,descending=False)
                rank_value = torch.ones_like(yq.squeeze(),dtype=torch.int64).cuda()
                rank_value[rank.squeeze()] = torch.tensor(range(len(yq))).cuda()

                f.write(msg + '\n')
                print(msg)

        if i_epi is None:
            nd = len(tasks)
            msg = f'[{split}/average] '
            for m in self.metrics:
                msg += f'{m} {avg_metrics[m] / nd:.3f} '
            mse_loss = avg_metrics['mse_loss']
            msg += f'MSE {mse_loss / nd:.3f} ({nd} datasets)'
            f.write(msg + '\n')
            print(msg)
        f.close()



    def load_model(self):
        loaded = torch.load(os.path.join(self.load_path))
        self.model.load_state_dict(loaded['model'])
        self.model.eval()
        self.model.cuda()
        if self.alpha_on:
            self.task_lr = {k: v.cuda() for k, v in loaded['task_lr'].items()}

    def test_full_arch_and_generate_topk(self):
        self.load_model()
        test_datasets = self.data.get_test_dataset_emb()
        arch_path = os.path.join(self.save_path, self.arch_save_path)
        arch_emb, arch_name = self.data.generate_arch_full(arch_path, self.load_pretrain)
        arch_emb = arch_emb.cuda()

        for dataset_emb, dataset in test_datasets:
            dataset_emb = dataset_emb.cuda()
            self.model(arch_emb, dataset_emb)
            y_predict, _ = self.model(arch_emb, dataset_feature=dataset_emb)

            if self.task == 'rp':
                top_index = list(torch.topk(y_predict.squeeze(), 5, largest=True).indices)
                file = f'configs/{dataset}.yaml'
            else:
                top_index = list(torch.topk(y_predict.squeeze(), 5, largest=True).indices)
                file = f'configs/{dataset}_ir.yaml'
            top_arch = [arch_name[i] for i in top_index]
            with open(file, "r") as f:
                configs = yaml.safe_load(f)
                print(f'dataset {dataset} top k')
                for rank, arch in enumerate(top_arch):
                    self.get_topk_configs(configs, list(arch), rank, dataset)
                print('\n')

    def test(self):
        test_datasets = self.data.get_test_dataset_emb()
        arch_path = os.path.join(self.save_path, self.arch_save_path)
        arch_emb, arch_name = self.data.generate_arch_full(arch_path, self.load_pretrain)
        arch_emb = arch_emb.cuda()

        for dataset_emb, dataset in test_datasets:
            dataset_emb = dataset_emb.cuda()
            self.model(arch_emb, dataset_emb)
            y_predict, _ = self.model(arch_emb, dataset_feature=dataset_emb)

            if self.task == 'rp':
                top_index = list(torch.topk(y_predict.squeeze(), 5, largest=True).indices)
                file = f'configs/{dataset}.yaml'
            else:
                top_index = list(torch.topk(y_predict.squeeze(), 5, largest=True).indices)
                file = f'configs/{dataset}_ir.yaml'
            top_arch = [arch_name[i] for i in top_index]
            with open(file, "r") as f:
                configs = yaml.safe_load(f)
                print(f'dataset {dataset} top k')
                for rank, arch in enumerate(top_arch):
                    self.get_topk_configs(configs, list(arch), rank, dataset)
                print('\n')

    def get_topk_configs(self, configs, arch, rank, dataset):
        save_configs = configs.copy()
        print(arch)
        save_configs['dataset']['encoder_dim'] = int(arch[0])
        save_configs['gnn']['msg'] = arch[1]
        save_configs['gnn']['layer_type'] = arch[2]
        save_configs['gnn']['layers_mp'] = int(arch[3])
        save_configs['gnn']['stage_type'] = arch[4]
        save_configs['model']['edge_decoding'] = arch[5]
        save_configs['gnn']['act'] = arch[6]
        save_configs['gnn']['component_num'] = int(arch[7])
        save_configs['gnn']['component_aggr'] = arch[8]
        if self.task == 'rp':
            config_dir = f'configs/{self.output_path}_{dataset}'
            save_configs['out_dir'] = f'results/{self.output_path}'
        else:
            config_dir = f'configs/{self.output_path}_{dataset}_ir'
            save_configs['out_dir'] = f'results/{self.output_path}_ir'
        if not os.path.exists(config_dir):
            os.mkdir(config_dir)
        out_file = os.path.join(config_dir, f"{self.save_path}_{dataset}_{rank}.yaml")
        # print(save_configs)
        # out_file = os.path.join(config_dir, f"top_{rank}.yaml")
        # print(save_configs)
        with open(out_file, "w") as f:
            yaml.dump(save_configs, f, default_flow_style=False)
