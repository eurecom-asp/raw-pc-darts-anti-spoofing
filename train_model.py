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
from ASVRawDataset import ASVRawDataset
from models.model import Network
from func.architect import Architect
from pathlib import Path
from func.functions import train_from_scratch, validate
from utils.utils import Genotype
from func.p2sgrad import P2SGradLoss

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ASVSpoof2019 model')
    parser.add_argument('--data', type=str, default='/path/to/your/LA', 
                    help='location of the data corpus')          
    parser.add_argument('--valid_freq', type=int, default=1, help='validate frequency')
    parser.add_argument('--report_freq', type=int, default=1000, help='report frequency in training')
    parser.add_argument('--layers', type=int, default=8, help='number of cells of the network')
    parser.add_argument('--init_channels', type=int, default=64, help='number of the initial channels of the network')
    parser.add_argument('--gru_hsize', type=int, default=1024, help='number of the features in the hidden state of gru layers')
    parser.add_argument('--gru_layers', type=int, default=3, help='number of gru layers of the network')
    parser.add_argument('--sinc_scale', type=str, default='mel', help='the type of sinc layer')
    parser.add_argument('--sinc_kernel', type=int, default=128, help='kernel size of sinc layer')
    parser.add_argument('--trainable', dest='is_trainable', action='store_true', help='whether using trainable sinc layer')
    parser.add_argument('--pre_trained', type=str, default=None)
    parser.add_argument('--arch', type=str, help='the searched architecture')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--no-mask', dest='is_mask', action='store_false', help='whether use freq mask')
    parser.add_argument('--sr', type=int, default=16000, help='default sampling rate')
    parser.add_argument('--lr', type=float, default=5e-5, help='intial learning rate')
    parser.add_argument('--lr_min', type=float, default=2e-5, help='mininum learning rate')
    parser.add_argument('--weight_decay', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--comment', type=str, default='EXP', help='Comment to describe the saved mdoel')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
    parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')
    parser.add_argument('--rand', dest='is_rand', action='store_true', help='whether use rand start of input audio')

    # by default, use fixed first 4 seconds of the original audio file, not using randomly selected 4 seconds
    parser.set_defaults(is_rand=False)
    
    parser.set_defaults(is_mask=True)
    parser.set_defaults(is_trainable=False)
    

    args = parser.parse_args()
    args.comment = 'train-{}-{}'.format(args.comment, time.strftime("%Y%m%d-%H%M%S"))
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

    train_protocol = 'ASVspoof2019.LA.cm.train.trn.txt'
    dev_protocol = 'ASVspoof2019.LA.cm.dev.trl.txt'

    OUTPUT_CLASSES = 2
    
    # set random seed
    if args.seed:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    logging.info("args = %s", args)
    
    device = 'cuda'
    criterion = P2SGradLoss()
    criterion = criterion.cuda()
    
    # get the network architecture
    genotype = eval(args.arch)
    # initialise the model
    model = Network(args.init_channels, args.layers, args, OUTPUT_CLASSES, genotype)
    model = model.to(device)

    # to make sure never use pre trained front-end while setting it to trainable
    # i.e., never update pre trained front-end sinc layer 
    assert not (args.pre_trained and args.is_trainable), "Warning: when using pre-trained sinc layer, these parameter should be set to not trainable."
    # load pre trained front-end from the saved models, only parameters of the sinc layer is loaded
    if args.pre_trained:
        checkpoint = torch.load(args.pre_trained)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in ['sinc.low_hz_', 'sinc.band_hz_', 'sinc.conv.weight', 'sinc.conv.bias']}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logging.info("loaded pre-trained sinc layer")
        if args.sinc_scale == 'conv':
            for name, para in enumerate(model.parameters()):
                if name < 2:
                    para.requires_grad = False
            logging.info("freezed pre-trained conv layer")


    logging.info("param size = %fM", utils.count_parameters(model))

    train_dataset = ASVRawDataset(Path(args.data), 'train', train_protocol, args.is_rand)
    dev_dataset = ASVRawDataset(Path(args.data), 'dev', dev_protocol, is_rand=False)

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

    optimizer = torch.optim.Adam(
        model.parameters(),
        args.lr,
        weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(args.num_epochs), eta_min=args.lr_min)

    begin_epoch = 0
    best_acc = 85
    best_loss = 1e3
    writer_dict = {
        'writer': SummaryWriter(args.comment),
        'train_global_steps': begin_epoch * len(train_loader),
        'valid_global_steps': begin_epoch // args.valid_freq,
    }

    for epoch in range(args.num_epochs):
        lr = scheduler.get_last_lr()[0]
        logging.info('Epoch: %d lr: %e', epoch, lr)

        model.drop_path_prob = args.drop_path_prob * epoch / args.num_epochs
        
        train_acc, train_loss = train_from_scratch(args, train_loader, model, optimizer, criterion, epoch, writer_dict)
        logging.info('train_loss %f', train_loss)
        logging.info('train_acc %f', train_acc)
        # validation
        if epoch % args.valid_freq == 0:
            dev_acc, dev_loss = validate(dev_loader, model, criterion, epoch, writer_dict, validate_type='dev')
            logging.info('dev_loss %f', dev_loss)
            logging.info('dev_acc %f', dev_acc)
            if dev_acc > best_acc:
                print('*'*50)
                logging.info('found the best acc model')
                print('*'*50)
            best_acc = max(dev_acc, best_acc)

        torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))
        scheduler.step()
