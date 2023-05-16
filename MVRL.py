import os
import shutil
import sys
import yaml
import time
import signal
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
import csv
from datetime import datetime

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.nt_xent import NTXentLoss
from utils.weighted_nt_xent import WeightedNTXentLoss
from data_aug.dataset import MoleculeDatasetWrapper
from models.ginet_2D import GINet_2D
from models.ginet_3D import GINet_3D
from models.seq_block import SeqEncoder

dataset = "pre"

# here is the max_len para
smiles_max_len_dict = {
    "bace": 194,
    "bbbp": 234,
    "hiv": 495,
    "clintox": 254,
    "tox21": 267,
    "sider": 982,
    "muv": 84,
    "pre": 500,
}
max_len = smiles_max_len_dict.get(dataset, 500)

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False


def read_smiles(data_path):
    smiles_data = []
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            smiles = row[-1]
            smiles_data.append(smiles)
    return smiles_data


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('/home/yrl/muti-view-RL/config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))

        # shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))


class MVRL(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        
        dir_name = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join('runs', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)

        self.dataset = dataset
        self.nt_xent_criterion = NTXentLoss(self.device, **config['loss'])
        self.weighted_nt_xent_criterion = WeightedNTXentLoss(self.device, **config['loss'])

    def _get_device(self):
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def train(self):

        train_loader, valid_loader = self.dataset.get_data_loaders()

        model_2d = GINet_2D(**self.config["model"]).to(self.device)
        model_3d = GINet_3D(**self.config["model"]).to(self.device)
        model_seq = SeqEncoder(max_len, self.config["seq_model"]).to(self.device)
        model_2d = self._load_pre_trained_weights(model_2d, "2d")
        model_3d = self._load_pre_trained_weights(model_3d, "3d")
        model_seq = self._load_pre_trained_weights(model_seq, "seq")

        print(model_2d)
        print(model_3d)
        print(model_seq)
        optimizer_2d = torch.optim.Adam(
            model_2d.parameters(), self.config['optim']['init_lr'], 
            weight_decay=self.config['optim']['weight_decay']
        )
        print('Optimizer:', optimizer_2d)

        optimizer_3d = torch.optim.Adam(
            model_3d.parameters(), self.config['optim']['init_lr'], 
            weight_decay=self.config['optim']['weight_decay']
        )
        print('Optimizer:', optimizer_3d)

        optimizer_seq = torch.optim.Adam(
            model_3d.parameters(), self.config['optim']['init_lr'], 
            weight_decay=self.config['optim']['weight_decay']
        )

        scheduler_2d = CosineAnnealingLR(optimizer_2d, T_max=self.config['epochs']-9, eta_min=0, last_epoch=-1)
        scheduler_3d = CosineAnnealingLR(optimizer_3d, T_max=self.config['epochs']-9, eta_min=0, last_epoch=-1)
        scheduler_seq = CosineAnnealingLR(optimizer_3d, T_max=self.config['epochs']-9, eta_min=0, last_epoch=-1)

        if apex_support and self.config['fp16_precision']:
            model_2d, optimizer_2d = amp.initialize(model_2d, optimizer_2d,
                                              opt_level='O2',
                                              keep_batchnorm_fp32=True)
            model_3d, optimizer_3d = amp.initialize(model_3d, optimizer_3d,
                                              opt_level='O2',
                                              keep_batchnorm_fp32=True)    
            model_seq, optimizer_seq = amp.initialize(model_seq, optimizer_seq,
                                              opt_level='O2',
                                              keep_batchnorm_fp32=True)                               
        
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        torch.cuda.empty_cache() 

        for epoch_counter in range(self.config['epochs']):
            torch.cuda.empty_cache() 
            # with torch.no_grad():
            for bn, (Seq_feature, Seq_len, g_2d, g_3d, mols, frag_mols) in enumerate(train_loader):
                torch.cuda.empty_cache() 
                optimizer_2d.zero_grad()
                optimizer_3d.zero_grad()
                optimizer_seq.zero_grad()

                g_2d = g_2d.to(self.device, non_blocking=True)
                g_3d = g_3d.to(self.device, non_blocking=True)
                Seq_feature = Seq_feature.to(self.device, non_blocking=True)
                Seq_len = Seq_len.to(self.device, non_blocking=True)

                # get the representations and the projections
                __, z2d_global, z2d_sub = model_2d(g_2d)  # [N,C] [16, 256]
                __, z3d_global, z3d_sub = model_3d(g_3d)  # [N,C] 
                seq_global = model_seq(Seq_feature, Seq_len) # [16, 64]

                # normalize projection feature vectors
                # 分子对比损失计算
                # z2d_global = F.normalize(z2d_global, dim=1)
                # z3d_global = F.normalize(z3d_global, dim=1)
                # loss_global = self.weighted_nt_xent_criterion(z2d_global, z3d_global, mols)

                # normalize projection feature vectors
                # 官能团对比损失计算
                z2d_sub = F.normalize(z2d_sub, dim=1)
                z3d_sub = F.normalize(z3d_sub, dim=1)
                loss_sub = self.nt_xent_criterion(z2d_sub, z3d_sub)

                # 2D分子和SMILE序列特征对比损失计算
                z2d_global = F.normalize(z2d_global, dim=1)
                seq_global = F.normalize(seq_global, dim=1)
                loss_global = self.nt_xent_criterion(z2d_global, seq_global)

                #loss = loss_global + self.config['loss']['lambda_2'] * loss_sub
                loss = self.config['loss']['lambda_1']*loss_global + self.config['loss']['lambda_2'] * loss_sub

                # print("epoch_count: {}; bn: {}; loss_global: {}; loss_sub: {}; loss: {};".format(epoch_counter, bn, loss_global.item(), loss_sub.item(), loss.item()))

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('loss_global', loss_global, global_step=n_iter)
                    self.writer.add_scalar('loss_sub', loss_sub, global_step=n_iter)
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('cosine_lr_decay_2d', scheduler_2d.get_last_lr()[0], global_step=n_iter)
                    self.writer.add_scalar('cosine_lr_decay_3d', scheduler_3d.get_last_lr()[0], global_step=n_iter)
                    self.writer.add_scalar('cosine_lr_decay_seq', scheduler_seq.get_last_lr()[0], global_step=n_iter)
                    
                    print("epoch_count: {}; bn: {}; loss_global: {}; loss_sub: {}; loss: {};".format(epoch_counter, bn, loss_global.item(), loss_sub.item(), loss.item()))

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer_2d) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer_2d.step()
                optimizer_3d.step()
                optimizer_seq.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss_global, valid_loss_sub = self._validate(model_2d, model_3d, model_seq, valid_loader)
                valid_loss = valid_loss_global + 0.5 * valid_loss_sub 
                print("epoch_counter: {}; bn: {}; valid_loss_global: {}; valid_loss_sub: {}; valid_loss: {}; (validation)".format(epoch_counter, bn, valid_loss_global, valid_loss_sub, valid_loss))
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    print("Save the best model in  ", model_checkpoints_folder)
                    torch.save(model_2d.state_dict(), os.path.join(model_checkpoints_folder, 'model_2d.pth'))
                    torch.save(model_3d.state_dict(), os.path.join(model_checkpoints_folder, 'model_3d.pth'))
                    torch.save(model_seq.state_dict(), os.path.join(model_checkpoints_folder, 'model_seq.pth'))
            
                self.writer.add_scalar('valid_loss_global', valid_loss_global, global_step=valid_n_iter)
                self.writer.add_scalar('valid_loss_sub', valid_loss_sub, global_step=valid_n_iter)
                self.writer.add_scalar('valid_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1
            
            if (epoch_counter+1) % 5 == 0:
                torch.save(model_2d.state_dict(), os.path.join(model_checkpoints_folder, 'model_2d_{}.pth'.format(str(epoch_counter))))
                torch.save(model_3d.state_dict(), os.path.join(model_checkpoints_folder, 'model_3d_{}.pth'.format(str(epoch_counter))))
                torch.save(model_seq.state_dict(), os.path.join(model_checkpoints_folder, 'model_seq_{}.pth'.format(str(epoch_counter))))

            # warmup for the first 10 epochs
            if epoch_counter >= self.config['warmup'] - 1:
                scheduler_2d.step()
                scheduler_3d.step()
                scheduler_seq.step()

    def _load_pre_trained_weights(self, model, tag):
        try:
            checkpoints_folder = os.path.join(self.config['resume_from'], 'checkpoints')
            if(tag == "2d"):
                state_dict = torch.load(os.path.join(checkpoints_folder, 'model_2d.pth'))
                model.load_state_dict(state_dict)
                print("Loaded pre-trained model with success.")
            elif(tag == "3d"):
                state_dict = torch.load(os.path.join(checkpoints_folder, 'model_3d.pth'))
                model.load_state_dict(state_dict)
                print("Loaded pre-trained model with success.")
            elif(tag == "seq"):
                state_dict = torch.load(os.path.join(checkpoints_folder, 'model_seq.pth'))
                model.load_state_dict(state_dict)
                print("Loaded pre-trained model with success.")

        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model_2d, model_3d, model_seq, valid_loader):
        # validation steps
        with torch.no_grad():
            model_2d.eval()
            model_3d.eval()
            model_seq.eval()

            valid_loss_global, valid_loss_sub = 0.0, 0.0
            counter = 0

            for bn, (Seq_feature, Seq_len, g_2d, g_3d, mols, frag_mols) in enumerate(valid_loader):

                g_2d = g_2d.to(self.device, non_blocking=True)
                g_3d = g_3d.to(self.device, non_blocking=True)
                Seq_feature = Seq_feature.to(self.device, non_blocking=True)
                Seq_len = Seq_len.to(self.device, non_blocking=True)

                # get the representations and the projections
                __, z2d_global, z2d_sub = model_2d(g_2d)  # [N,C]
                __, z3d_global, z3d_sub = model_3d(g_3d)  # [N,C]
                seq_global = model_seq(Seq_feature, Seq_len) # [N, max_len]

                # normalize projection feature vectors
                # 官能团对比损失计算
                z2d_sub = F.normalize(z2d_sub, dim=1)
                z3d_sub = F.normalize(z3d_sub, dim=1)
                loss_sub = self.nt_xent_criterion(z2d_sub, z3d_sub)

                # 2D分子和SMILE序列特征对比损失计算
                z2d_global = F.normalize(z2d_global, dim=1)
                seq_global = F.normalize(seq_global, dim=1)
                loss_global = self.nt_xent_criterion(z2d_global, seq_global)

                valid_loss_global += loss_global.item()
                valid_loss_sub += loss_sub.item()

                counter += 1
        
            valid_loss_global /= counter
            valid_loss_sub /= counter

        model_2d.train()
        model_3d.train()
        model_seq.train()
        return valid_loss_global, valid_loss_sub


def main():
    # config = yaml.load(open("config.yaml", "r", encoding='utf-8'), Loader=yaml.FullLoader)
    config = yaml.load(open("/home/yrl/muti-view-RL/config.yaml", "r", encoding='utf-8'), Loader=yaml.FullLoader)

    print(config)
    dataset = MoleculeDatasetWrapper(max_len, config['batch_size'], **config['dataset'])
    mvrl = MVRL(dataset, config)
    mvrl.train()


if __name__ == "__main__":
    main()
