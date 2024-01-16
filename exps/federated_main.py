#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy, sys
import time
import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
import random
import torch.utils.model_zoo as model_zoo
from pathlib import Path

# __file__获取当前文件路径，pathlib.Path是当前文件的路径，parent获取父目录
# / ".." / "lib"将父目录与子目录连接起来，将路径解析为绝对路径
lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))    # 将指定的目录路径插入到 sys.path 的最前面，使得 Python 解释器在搜索模块时首先查找这个目录
mod_dir = (Path(__file__).parent / ".." / "lib" / "models").resolve()
if str(mod_dir) not in sys.path:
    sys.path.insert(0, str(mod_dir))

from resnet import resnet18
from options import args_parser
from update import LocalUpdate, save_protos, LocalTest, test_inference_new_het_lt
from models import CNNMnist, CNNFemnist
from utils import get_dataset, average_weights, exp_details, proto_aggregation, agg_func, average_weights_per, average_weights_sem

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# 任务异构场景
def FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list):
    # 展示数据的对象
    summary_writer = SummaryWriter('../tensorboard/'+ args.dataset +'_fedproto_' + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.stdev) + 'e_' + str(args.num_users) + 'u_' + str(args.rounds) + 'r')

    # 全局原型
    global_protos = []
    # 用户索引，0到19
    idxs_users = np.arange(args.num_users)
    # 训练损失，训练精度
    train_loss, train_accuracy = [], []

    # 按照轮数循环
    for round in tqdm(range(args.rounds)):
        # 局部权重，全局损失，局部原型，存储1轮里20个客户端的 局部权重、局部损失、局部损失
        local_weights, local_losses, local_protos = [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        # proto损失
        proto_loss = 0
        for idx in idxs_users:
            # 实例化LocalUpdate对象，位于update.py里
            # user_groups是一个字典，传入idx可以获得对应500个数据的索引（以CIFAR10为例）
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            # 局部模型更新权重
            w, loss, acc, protos = local_model.update_weights_het(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            # 对原型进行汇聚，最终输出一个字典，即一个类对应一个proto
            agg_protos = agg_func(protos)
            # 添加idx对应的权重、损失到local_weights和local_losses里
            # 更新原型
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))
            # 客户端idx对应的原型情况，一个类对应一个原型，CIFAR10就有10个原型
            local_protos[idx] = agg_protos
            # loss['2']对应的是原型损失
            summary_writer.add_scalar('Train/Loss/user' + str(idx + 1), loss['total'], round)
            summary_writer.add_scalar('Train/Loss1/user' + str(idx + 1), loss['1'], round)
            summary_writer.add_scalar('Train/Loss2/user' + str(idx + 1), loss['2'], round)
            summary_writer.add_scalar('Train/Acc/user' + str(idx + 1), acc, round)
            # 记录客户端idx对应的原型损失
            proto_loss += loss['2']

        # update global weights
        # 每一个全局round获得20个用户的局部weight，为全局weight
        local_weights_list = local_weights

        # 枚举20个用户，将local_weights_list的权重给local_model，然后更新local_model_list
        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model

        # update global weights
        # 原型聚合，将所有客户端对应的所有原型进行聚合
        global_protos = proto_aggregation(local_protos)

        # 计算平均损失，即20个客户端在一轮的平均损失
        loss_avg = sum(local_losses) / len(local_losses)
        # 记录的是所有轮round的平均损失
        train_loss.append(loss_avg)

    # 获得测试精度 和 损失
    acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos)
    print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
    print('For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_l), np.std(acc_list_l)))
    print('For all users (with protos), mean of proto loss is {:.5f}, std of test acc is {:.5f}'.format(np.mean(loss_list), np.std(loss_list)))

    # save protos
    if args.dataset == 'mnist':
        save_protos(args, local_model_list, test_dataset, user_groups_lt)

# 模型异构场景
def FedProto_modelheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list):
    summary_writer = SummaryWriter('../tensorboard/'+ args.dataset +'_fedproto_mh_' + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.stdev) + 'e_' + str(args.num_users) + 'u_' + str(args.rounds) + 'r')

    global_protos = []
    idxs_users = np.arange(args.num_users)

    train_loss, train_accuracy = [], []

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos = [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        proto_loss = 0
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss, acc, protos = local_model.update_weights_het(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            agg_protos = agg_func(protos)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))

            local_protos[idx] = agg_protos
            summary_writer.add_scalar('Train/Loss/user' + str(idx + 1), loss['total'], round)
            summary_writer.add_scalar('Train/Loss1/user' + str(idx + 1), loss['1'], round)
            summary_writer.add_scalar('Train/Loss2/user' + str(idx + 1), loss['2'], round)
            summary_writer.add_scalar('Train/Acc/user' + str(idx + 1), acc, round)
            proto_loss += loss['2']

        # update global weights
        local_weights_list = local_weights

        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model

        # update global protos
        global_protos = proto_aggregation(local_protos)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

    acc_list_l, acc_list_g = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos)
    print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
    print('For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_l), np.std(acc_list_l)))

if __name__ == '__main__':
    start_time = time.time()

    # 输出参数
    args = args_parser()
    exp_details(args)

    # set random seeds
    # 设置随机种子
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load dataset and user groups
    # n_list表示每个用户对应的类别数目，默认情况下ways为5，stdev为2，shots为100
    n_list = np.random.randint(max(2, args.ways - args.stdev), min(args.num_classes, args.ways + args.stdev + 1), args.num_users)
    # k_list表示每个用户对应的类别数
    if args.dataset == 'mnist':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev - 1, args.num_users)
    elif args.dataset == 'cifar10':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)
    elif args.dataset =='cifar100':
        k_list = np.random.randint(args.shots, args.shots + 1, args.num_users)
    elif args.dataset == 'femnist':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)
    # 获得数据集，测试数据集和用户索引和对应数据
    train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt = get_dataset(args, n_list, k_list)

    # Build models
    # 局部模型列表
    local_model_list = []
    # 为每个用户赋予初始的局部模型
    for i in range(args.num_users):
        if args.dataset == 'mnist':
            if args.mode == 'model_heter':
                if i<7:
                    args.out_channels = 18
                elif i>=7 and i<14:
                    args.out_channels = 20
                else:
                    args.out_channels = 22
            else:
                args.out_channels = 20

            local_model = CNNMnist(args=args)

        elif args.dataset == 'femnist':
            if args.mode == 'model_heter':
                if i<7:
                    args.out_channels = 18
                elif i>=7 and i<14:
                    args.out_channels = 20
                else:
                    args.out_channels = 22
            else:
                args.out_channels = 20
            local_model = CNNFemnist(args=args)

        elif args.dataset == 'cifar100' or args.dataset == 'cifar10':
            # 如果是任务异构，则args.stride一定是[2, 2]
            if args.mode == 'model_heter':
                if i<10:
                    args.stride = [1,4]
                else:
                    args.stride = [2,2]
            else:
                args.stride = [2, 2]
            # 初始化resnet18，如果pretrained=True则返回Imagenet上的预训练模型
            resnet = resnet18(args, pretrained=False, num_classes=args.num_classes)
            # 下载resnet18的初始化权重
            initial_weight = model_zoo.load_url(model_urls['resnet18'])
            # 设置局部模型，这个对应update.py里的model(images)
            local_model = resnet
            # 将resnet18初始化的权重保存到initial_weight_1
            initial_weight_1 = local_model.state_dict()
            # 将resnet18初始化的参数 赋给 下载resnet18的权重
            for key in initial_weight.keys():
                if key[0:3] == 'fc.' or key[0:5]=='conv1' or key[0:3]=='bn1':
                    initial_weight[key] = initial_weight_1[key]
            # 将下载resnet里的fc. conv1 bn1赋给初始化的resnet18
            local_model.load_state_dict(initial_weight)

        local_model.to(args.device)
        # 每个模型设置训练模式，然后将训练后的模型添加到list里
        # local_model.train()标记训练模式开启，训练模式下模型行为可能与测试模式下不同。如某些层如 dropout 层在训练模式下启用，而在测试模式下禁用
        local_model.train()
        local_model_list.append(local_model)

    # 根据异构性分别调用 任务异构/模型异构
    # 任务异构（本文称统计异构）：任务有不同的统计分布，设置标准差是1或2，在类空间和数据大小上创建异构性
    # 模型异构（在MNIST和FEMNIST里，卷积层的输出channel被设置为18、20、22），CIFAR10里 不同客户端卷积层的stride是不同的
    if args.mode == 'task_heter':
        FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list)
    else:
        FedProto_modelheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list)