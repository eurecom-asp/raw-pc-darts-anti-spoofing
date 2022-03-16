import os
import sys
import time
import glob
import numpy as np
import torch
from utils import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from ASVRawDataset import ASVRawDataset
from models.model_search import Network
from func.architect import Architect
from pathlib import Path
from func.functions import train, validate
from func.p2sgrad import P2SGradLoss


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ASVSpoof2019 model')
    parser.add_argument('--data', type=str, default='/path/to/your/LA', 
                    help='location of the data corpus')          
    parser.add_argument('--valid_freq', type=int, default=1, help='validate frequency')
    parser.add_argument('--report_freq', type=int, default=1000, help='report frequency in training')
    parser.add_argument('--layers', type=int, default=8, help='number of cells of the network')
    parser.add_argument('--init_channels', type=int, default=64, help='number of the initial channels of the network')
    parser.add_argument('--sinc_scale', type=str, default='mel', help='the type of sinc layer')
    parser.add_argument('--sinc_kernel', type=int, default=128, help='kernel size of sinc layer')
    parser.add_argument('--gru_hsize', type=int, default=1024, help='number of the features in the hidden state of gru layers')
    parser.add_argument('--gru_layers', type=int, default=3, help='number of gru layers of the network')
    parser.add_argument('--trainable', dest='is_trainable', action='store_true', help='whether using trainable sinc layer')
    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--warm_up_epoch', type=int, default=10, help='the network architecture will not change until this epoch')
    parser.add_argument('--sr', type=int, default=16000, help='default sampling rate')
    parser.add_argument('--lr', type=float, default=5e-5, help='intial learning rate')
    parser.add_argument('--no-mask', dest='is_mask', action='store_false', help='whether use freq mask, if not set, use mask by default')
    parser.add_argument('--weight_decay', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--comment', type=str, default='EXP', help='Comment to describe the saved mdoel')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
    parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    parser.add_argument('--rand', dest='is_rand', action='store_true', help='whether use rand start of input audio')
    
    # by default, use fixed first 4 seconds of the original audio file, not using randomly selected 4 seconds
    parser.set_defaults(is_rand=False)
    # use mask by default
    parser.set_defaults(is_mask=True)
    parser.set_defaults(is_trainable=False)

    args = parser.parse_args()
    args.comment = 'search-{}-{}'.format(args.comment, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.comment, scripts_to_save=glob.glob('*.py'))
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.comment, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    
    # models will be saved under this path
    model_save_path = os.path.join(args.comment, 'models')
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    train_protocol = 'ASVspoof2019.LA.cm.train.trn_h.txt'
    dev_protocol = 'ASVspoof2019.LA.cm.train.trn_t.txt'
    eval_protocol = 'ASVspoof2019.LA.cm.dev.trl.txt'

    OUTPUT_CLASSES = 2
    
    # set random seed
    if args.seed:
        cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)

    logging.info("args = %s", args)
    
    device = 'cuda'
    criterion = P2SGradLoss()
    criterion = criterion.cuda()

    model = Network(args.init_channels, args.layers, args, OUTPUT_CLASSES, criterion)
    model = model.cuda()
    architect = Architect(model, args)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.Adam(
        model.parameters(),
        args.lr,
        weight_decay=args.weight_decay)
    
    train_dataset = ASVRawDataset(Path(args.data), 'train', train_protocol, args.is_rand)
    dev_dataset = ASVRawDataset(Path(args.data), 'train', dev_protocol, is_rand=False)
    eval_dataset = ASVRawDataset(Path(args.data), 'dev', eval_protocol, is_rand=False)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )
    dev_loader = torch.utils.data.DataLoader(
        dataset=dev_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )
    eval_loader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )

    begin_epoch = 0
    writer_dict = {
        'writer': SummaryWriter(args.comment),
        'train_global_steps': begin_epoch * len(train_loader),
        'valid_global_steps': begin_epoch // args.valid_freq,
    }
    
    best_acc = 85
    for epoch in range(args.num_epochs):
        logging.info('epoch %d ', epoch)
        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        # training
        train_acc, train_loss = train(args, train_loader, dev_loader, model, architect, criterion,
                                    args.lr, optimizer, epoch, writer_dict)

        logging.info('train_acc %f', train_acc)
        logging.info('train_loss %f', train_loss)

        # validation
        if epoch % args.valid_freq == 0:
            valid_acc, valid_loss = validate(eval_loader, model, criterion, epoch, writer_dict, validate_type='dev')
            print('*'*50)
            logging.info('dev_acc %f', valid_acc)
            logging.info('dev_loss %f', valid_loss)
            if valid_acc > best_acc:
                logging.info('best model')
            print('*'*50)
        
        best_acc = max(valid_acc, best_acc)
        torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))
