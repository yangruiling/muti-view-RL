import os
import shutil
import yaml
import torch
import pandas as pd
import numpy as np
from datetime import datetime

from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_auc_score

from data_aug.dataset_mix_test import MoleculeTestDatasetWrapper
from models.ginet_2D_finetune import GINet_2D 
from models.ginet_3D_finetune import GINet_3D
from models.seq_block_finetune import SeqEncoder




def _save_config_file(log_dir, config):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        with open(os.path.join(log_dir, 'config_finetune.yaml'), 'w') as config_file:
            yaml.dump(config, config_file)


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class FineTune_2d(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        self.dataset = dataset

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        dir_name = config['fine_tune_from'].split('/')[0] + '-' + \
            config['fine_tune_from'].split('/')[-1] + '-' + config['task_name']
        subdir_name = current_time + '-' + config['dataset']['target']
        self.log_dir = os.path.join('experiments', dir_name, subdir_name)

        model_yaml_dir = os.path.join(config['fine_tune_from'], 'ckpt')
        for fn in os.listdir(model_yaml_dir):
            if fn.endswith(".yaml"):
                model_yaml_fn = fn
                break
        model_yaml = os.path.join(model_yaml_dir, model_yaml_fn)
        model_config = yaml.load(open(model_yaml, "r"), Loader=yaml.FullLoader)
        self.model_config = model_config['model']
        self.model_config['dropout'] = self.config['model']['dropout']
        self.model_config['pool'] = self.config['model']['pool']

        if config['dataset']['task'] == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        elif config['dataset']['task'] == 'regression':
            if self.config["task_name"] in ['qm7', 'qm8']:
                # self.criterion = nn.L1Loss()
                self.criterion = nn.SmoothL1Loss()
            else:
                self.criterion = nn.MSELoss()

        # save config file
        _save_config_file(self.log_dir, self.config)

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def _step(self, model, data):
        pred = model(data)

        if self.config['dataset']['task'] == 'classification':
            loss = self.criterion(pred, data.y.view(-1))
        elif self.config['dataset']['task'] == 'regression':
            if self.normalizer:
                loss = self.criterion(pred, self.normalizer.norm(data.y))
            else:
                loss = self.criterion(pred, data.y)

        return loss

    def train(self):
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()

        self.normalizer = None
        if self.config["task_name"] in ['qm7']:
            labels = []
            for d in train_loader:
                labels.append(d.y)
            labels = torch.cat(labels)
            self.normalizer = Normalizer(labels)
            print("mean: ", self.normalizer.mean, "std: ", self.normalizer.std, "labels_shape: ", labels.shape)

        n_batches = len(train_loader)
        if n_batches < self.config['log_every_n_steps']:
            self.config['log_every_n_steps'] = n_batches
        self.model_config['num_layer'],
        model = GINet(self.config['dataset']['task'], self.model_config['num_layer'], self.model_config['emb_dim'], self.model_config['dropout'], self.model_config['pool']).to(self.device)
        model = self._load_pre_trained_weights(model)

        layer_list = []
        for name, param in model.named_parameters():
            if 'output_layers' in name:
                print(name)
                layer_list.append(name)
        writer = SummaryWriter(log_dir = save_dir)
        
        params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))

        if self.config['optim']['type'] == 'SGD':
            init_lr = self.config['optim']['lr'] * self.config['batch_size'] / 256
            optimizer = torch.optim.SGD(
                [   {'params': params, 'lr': init_lr}, 
                    {'params': base_params, 'lr': init_lr * self.config['optim']['base_ratio']}
                ],
                momentum=self.config['optim']['momentum'],
                weight_decay=self.config['optim']['weight_decay']
            )
        elif self.config['optim']['type'] == 'Adam':
            optimizer = torch.optim.Adam(
                [   {'params': params, 'lr': self.config['optim']['lr']}, 
                    {'params': base_params, 'lr': self.config['optim']['lr'] * self.config['optim']['base_ratio']}
                ],
                weight_decay=self.config['optim']['weight_decay']
            )
        else:
            raise ValueError('Not defined optimizer type!')

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_rmse = np.inf
        best_valid_mae = np.inf
        best_valid_roc_auc = 0

        for epoch_counter in range(self.config['epochs']):
            for bn, data in enumerate(train_loader):
                data = data.to(self.device)
                loss = self._step(model, data)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    print("epoch: ", epoch_counter, "bn: ", bn, "loss: ", loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                n_iter += 1
                writer.add_scalar("pre_trainloss", loss)

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                if self.config['dataset']['task'] == 'classification': 
                    valid_loss, valid_roc_auc = self._validate(model, valid_loader)
                    if valid_roc_auc > best_valid_roc_auc:
                        best_valid_roc_auc = valid_roc_auc
                        # save the model weights
                        torch.save(model.state_dict(), os.path.join(self.log_dir, 'model.pth')) 
                    writer.add_scalar("valid_loss", valid_loss) 
                    writer.add_scalar("valid_roc_auc", valid_roc_auc) 
                
                elif self.config['dataset']['task'] == 'regression': 
                    valid_loss, valid_rmse, valid_mae = self._validate(model, valid_loader)
                    if self.config["task_name"] in ['qm7', 'qm8'] and valid_mae < best_valid_mae:
                        best_valid_mae = valid_mae
                        # save the model weights
                        torch.save(model.state_dict(), os.path.join(self.log_dir, 'model.pth'))
                    elif valid_rmse < best_valid_rmse:
                        best_valid_rmse = valid_rmse
                        # save the model weights
                        torch.save(model.state_dict(), os.path.join(self.log_dir, 'model.pth'))
                    writer.add_scalar("valid_loss", valid_loss) 
                    writer.add_scalar("valid_rmse", valid_rmse) 
                    writer.add_scalar("valid_mae", valid_mae)
                
                valid_n_iter += 1
                    


        return self._test(model, test_loader)

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join(self.config['fine_tune_from'], 'ckpt')
            ckp_path = os.path.join(checkpoints_folder, 'model_2d.pth')
            state_dict = torch.load(ckp_path, map_location=self.device)
            model.load_my_state_dict(state_dict)
            print("Loaded pre-trained model {} with success.".format(ckp_path))
        
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):
        # test steps
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            num_data = 0
            for bn, data in enumerate(valid_loader):
                data = data.to(self.device)

                pred = model(data)
                loss = self._step(model, data)

                valid_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            valid_loss /= num_data
        
        model.train()

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            rmse = mean_squared_error(labels, predictions, squared=False)
            mae = mean_absolute_error(labels, predictions)
            print('Validation loss:', valid_loss, 'RMSE:', rmse, 'MAE:', mae)
            return valid_loss, rmse, mae

        elif self.config['dataset']['task'] == 'classification': 
            predictions = np.array(predictions)
            labels = np.array(labels)
            roc_auc = roc_auc_score(labels, predictions[:,1])
            print('Validation loss:', valid_loss, 'ROC AUC:', roc_auc)
            return valid_loss, roc_auc

    def _test(self, model, test_loader):
        model_path = os.path.join(self.log_dir, 'model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print("Loaded {} with success.".format(model_path))

        # test steps
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            test_loss = 0.0
            num_data = 0
            for bn, data in enumerate(test_loader):
                data = data.to(self.device)

                pred = model(data)
                loss = self._step(model, data)

                test_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

        test_loss /= num_data
        
        model.train()

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            rmse = mean_squared_error(labels, predictions, squared=False)
            mae = mean_absolute_error(labels, predictions)
            print('Test loss:', test_loss, 'RMSE:', rmse, 'MAE:', mae)
            return test_loss, rmse, mae

        elif self.config['dataset']['task'] == 'classification': 
            predictions = np.array(predictions)
            labels = np.array(labels)
            roc_auc = roc_auc_score(labels, predictions[:,1])
            print('Test loss:', test_loss, 'ROC AUC:', roc_auc)
            return test_loss, roc_auc

class FineTune_mix(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        self.dataset = dataset

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        dir_name = config['fine_tune_from'].split('/')[0] + '-' + \
            config['fine_tune_from'].split('/')[-1] + '-' + config['task_name']
        subdir_name = current_time + '-' + config['dataset']['target']
        self.log_dir = os.path.join('experiments', dir_name, subdir_name)

        # model_yaml_dir = os.path.join(config['fine_tune_from'], 'ckpt')
        model_yaml_dir = '/home/yrl/muti-view-RL/pretrained/1000/ckpt'
        for fn in os.listdir(model_yaml_dir):
            if fn.endswith(".yaml"):
                model_yaml_fn = fn
                break
        model_yaml = os.path.join(model_yaml_dir, model_yaml_fn)
        model_config = yaml.load(open(model_yaml, "r"), Loader=yaml.FullLoader)
        self.model_config = model_config['model']
        self.model_config['dropout'] = self.config['model']['dropout']
        self.model_config['pool'] = self.config['model']['pool']
        self.model_config['seq_model'] = model_config['seq_model']

        if config['dataset']['task'] == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        elif config['dataset']['task'] == 'regression':
            if self.config["task_name"] in ['qm7', 'qm8']:
                # self.criterion = nn.L1Loss()
                self.criterion = nn.SmoothL1Loss()
            else:
                self.criterion = nn.MSELoss()

        # save config file
        _save_config_file(self.log_dir, self.config)

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def _step(self, model, data, target):
        pred = model(data)

        if self.config['dataset']['task'] == 'classification':
            loss = self.criterion(pred, target.view(-1))
        elif self.config['dataset']['task'] == 'regression':
            # loss = self.criterion(pred, data.y)
            if self.normalizer:
                loss = self.criterion(pred, self.normalizer.norm(target.reshape(-1,1)))
            else:
                loss = self.criterion(pred, target.reshape(-1,1))

        return loss

    def train(self,hyperparameter):
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()

        self.normalizer = None
        if self.config["task_name"] in ['qm7']:
            labels = []
            for Seq_feature, Seq_len, g_2d, g_3d, mols, target in train_loader:
                labels.append(target)
            labels = torch.cat(labels)
            self.normalizer = Normalizer(labels)
            print("mean: ", self.normalizer.mean, "std: ", self.normalizer.std, "labels_shape: ", labels.shape)

        n_batches = len(train_loader)
        if n_batches < self.config['log_every_n_steps']:
            self.config['log_every_n_steps'] = n_batches
        self.model_config['num_layer'],
        model_2d = GINet_2D(self.config['dataset']['task'], self.model_config['num_layer'], self.model_config['emb_dim'], self.model_config['dropout'], self.model_config['pool']).to(self.device)
        model_2d = self._load_pre_trained_weights(model_2d,'2d')
        model_3d = GINet_3D(self.config['dataset']['task'], self.model_config['num_layer'], self.model_config['emb_dim'], self.model_config['dropout'], self.model_config['pool']).to(self.device)
        model_3d = self._load_pre_trained_weights(model_3d,'3d')
        seq_model = SeqEncoder(self.config['dataset']['task'], config['dataset']['max_len'] , self.model_config['seq_model']).to(self.device)
        seq_model = self._load_pre_trained_weights(seq_model,'seq')
        
        layer_list_2d = []
        for name, param in model_2d.named_parameters():
            if 'output_layers' in name:
                # print(name)
                layer_list_2d.append(name)
        
        layer_list_3d = []
        for name, param in model_3d.named_parameters():
            if 'output_layers' in name:
                # print(name)
                layer_list_3d.append(name)

        layer_list_seq = []
        for name, param in seq_model.named_parameters():
            if 'output_layers' in name:
                # print(name)
                layer_list_seq.append(name)

        writer = SummaryWriter(log_dir = save_dir)
        
        params_2d = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list_2d, model_2d.named_parameters()))))
        base_params_2d = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list_2d, model_2d.named_parameters()))))

        params_3d = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list_3d, model_3d.named_parameters()))))
        base_params_3d = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list_3d, model_3d.named_parameters()))))

        params_seq = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list_seq, seq_model.named_parameters()))))
        base_params_seq = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list_seq, seq_model.named_parameters()))))

        if self.config['optim']['type'] == 'SGD':
            init_lr = self.config['optim']['lr'] * self.config['batch_size'] / 256
            optimizer_2d = torch.optim.SGD(
                [   {'params': params_2d, 'lr': init_lr}, 
                    {'params': base_params_2d, 'lr': init_lr * self.config['optim']['base_ratio']}
                ],
                momentum=self.config['optim']['momentum'],
                weight_decay=self.config['optim']['weight_decay']
            )
            optimizer_3d = torch.optim.SGD(
                [   {'params': params_3d, 'lr': init_lr}, 
                    {'params': base_params_3d, 'lr': init_lr * self.config['optim']['base_ratio']}
                ],
                momentum=self.config['optim']['momentum'],
                weight_decay=self.config['optim']['weight_decay']
            )
            optimizer_seq = torch.optim.SGD(
                [   {'params': params_seq, 'lr': init_lr}, 
                    {'params': base_params_seq, 'lr': init_lr * self.config['optim']['base_ratio']}
                ],
                momentum=self.config['optim']['momentum'],
                weight_decay=self.config['optim']['weight_decay']
            )
        elif self.config['optim']['type'] == 'Adam':
            optimizer_2d = torch.optim.Adam(
                [   {'params': params_2d, 'lr': self.config['optim']['lr']}, 
                    {'params': base_params_2d, 'lr': self.config['optim']['lr'] * self.config['optim']['base_ratio']}
                ],
                weight_decay=self.config['optim']['weight_decay']
            )
            optimizer_3d = torch.optim.Adam(
                [   {'params': params_3d, 'lr': self.config['optim']['lr']}, 
                    {'params': base_params_3d, 'lr': self.config['optim']['lr'] * self.config['optim']['base_ratio']}
                ],
                weight_decay=self.config['optim']['weight_decay']
            )
            optimizer_seq = torch.optim.Adam(
                [   {'params': params_seq, 'lr': self.config['optim']['lr']}, 
                    {'params': base_params_seq, 'lr': self.config['optim']['lr'] * self.config['optim']['base_ratio']}
                ],
                weight_decay=self.config['optim']['weight_decay']
            )
        else:
            raise ValueError('Not defined optimizer type!')

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_rmse_2d = np.inf
        best_valid_rmse_3d = np.inf
        best_valid_rmse_seq = np.inf
        best_valid_mae_2d = np.inf
        best_valid_mae_3d = np.inf
        best_valid_mae_seq = np.inf
        best_valid_roc_auc_2d = 0
        best_valid_roc_auc_3d = 0
        best_valid_roc_auc_seq = 0

        for epoch_counter in range(self.config['epochs']):
            for bn, (Seq_feature, Seq_len, g_2d, g_3d, mols, target) in enumerate(train_loader):
                target = target.to(self.device)
                g_2d = g_2d.to(self.device)
                loss_2d = self._step(model_2d, g_2d, target)
                g_3d = g_3d.to(self.device)
                loss_3d = self._step(model_3d, g_3d, target)
                Seq_feature = Seq_feature.to(self.device)
                Seq_len = Seq_len.to(self.device)
                pred = seq_model(Seq_feature, Seq_len)
                
                
                if self.config['dataset']['task'] == 'classification':
                    loss_seq = self.criterion(pred, target.view(-1))
                elif self.config['dataset']['task'] == 'regression':
                    if self.normalizer:
                        loss_seq = self.criterion(pred, self.normalizer.norm(target.reshape(-1,1)))
                    else:
                        loss_seq = self.criterion(pred, target.reshape(-1,1))

                if n_iter % self.config['log_every_n_steps'] == 0:
                    print("epoch: ", epoch_counter, "bn: ", bn, "loss_2d: ", loss_2d.item(), "loss_3d: ", loss_3d.item(), "loss_seq: ", loss_seq.item())
                
                loss = hyperparameter[0] * loss_2d + hyperparameter[1] * loss_3d + hyperparameter[2] * loss_seq

                optimizer_2d.zero_grad()
                optimizer_3d.zero_grad()
                optimizer_seq.zero_grad()
                loss.backward()
                optimizer_2d.step()
                optimizer_3d.step()
                optimizer_seq.step()

                n_iter += 1
                writer.add_scalar("pre_trainloss2d", loss_2d)
                writer.add_scalar("pre_trainloss3d", loss_3d)
                writer.add_scalar("pre_trainlossseq", loss_seq)
                writer.add_scalar("pre_trainloss", loss)

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                if self.config['dataset']['task'] == 'classification': 
                    valid_loss, valid_roc_auc_2d, valid_roc_auc_3d, valid_roc_auc_seq = self._validate(model_2d, model_3d, seq_model, valid_loader, hyperparameter)
                    if valid_roc_auc_2d >= best_valid_roc_auc_2d:
                        best_valid_roc_auc_2d = valid_roc_auc_2d
                        # save the model weights
                        torch.save(model_2d.state_dict(), os.path.join(self.log_dir, 'model_2d.pth')) 
                    if valid_roc_auc_3d >= best_valid_roc_auc_3d:
                        best_valid_roc_auc_3d = valid_roc_auc_3d
                        # save the model weights
                        torch.save(model_3d.state_dict(), os.path.join(self.log_dir, 'model_3d.pth')) 
                    if valid_roc_auc_seq >= best_valid_roc_auc_seq:
                        best_valid_roc_auc_seq = valid_roc_auc_seq
                        # save the model weights
                        torch.save(seq_model.state_dict(), os.path.join(self.log_dir, 'model_seq.pth')) 
                    writer.add_scalar("valid_loss", valid_loss) 
                    writer.add_scalar("valid_roc_auc", valid_roc_auc_2d)
                    writer.add_scalar("valid_roc_auc", valid_roc_auc_3d)
                    writer.add_scalar("valid_roc_auc", valid_roc_auc_seq) 
                
                elif self.config['dataset']['task'] == 'regression': 
                    valid_loss, valid_rmse_2d, valid_mae_2d, valid_rmse_3d, valid_mae_3d, valid_rmse_seq, valid_mae_seq = self._validate(model_2d, model_3d, seq_model, valid_loader, hy)
                    if self.config["task_name"] in ['qm7', 'qm8'] and valid_mae_2d <= best_valid_mae_2d:
                        best_valid_mae_2d = valid_mae_2d
                        # save the model weights
                        torch.save(model_2d.state_dict(), os.path.join(self.log_dir, 'model_2d.pth'))
                    elif valid_rmse_2d <= best_valid_rmse_2d:
                        best_valid_rmse_2d = valid_rmse_2d
                        # save the model weights
                        torch.save(model_2d.state_dict(), os.path.join(self.log_dir, 'model_2d.pth'))
                    
                    if self.config["task_name"] in ['qm7', 'qm8'] and valid_mae_3d <= best_valid_mae_3d:
                        best_valid_mae_3d = valid_mae_3d
                        # save the model weights
                        torch.save(model_3d.state_dict(), os.path.join(self.log_dir, 'model_3d.pth'))
                    elif valid_rmse_3d <= best_valid_rmse_3d:
                        best_valid_rmse_3d = valid_rmse_3d
                        # save the model weights
                        torch.save(model_3d.state_dict(), os.path.join(self.log_dir, 'model_3d.pth'))

                    if self.config["task_name"] in ['qm7', 'qm8'] and valid_mae_seq <= best_valid_mae_seq:
                        best_valid_mae_seq = valid_mae_seq
                        # save the model weights
                        torch.save(seq_model.state_dict(), os.path.join(self.log_dir, 'model_seq.pth'))
                    elif valid_rmse_seq <= best_valid_rmse_seq:
                        best_valid_rmse_seq = valid_rmse_seq
                        # save the model weights
                        torch.save(seq_model.state_dict(), os.path.join(self.log_dir, 'model_seq.pth'))    
                        
                    writer.add_scalar("valid_loss", valid_loss) 
                    writer.add_scalar("valid_rmse", valid_rmse_2d) 
                    writer.add_scalar("valid_mae", valid_mae_2d)
                    writer.add_scalar("valid_rmse", valid_rmse_3d) 
                    writer.add_scalar("valid_mae", valid_mae_3d)
                    writer.add_scalar("valid_rmse", valid_rmse_seq) 
                    writer.add_scalar("valid_mae", valid_mae_seq)
                
                valid_n_iter += 1
        return self._test(model_2d, model_3d, seq_model, test_loader,hyperparameter)

    def abalation(self,hyperparameter):
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()

        self.normalizer = None
        if self.config["task_name"] in ['qm7']:
            labels = []
            for Seq_feature, Seq_len, g_2d, g_3d, mols, target in train_loader:
                labels.append(target)
            labels = torch.cat(labels)
            self.normalizer = Normalizer(labels)
            print("mean: ", self.normalizer.mean, "std: ", self.normalizer.std, "labels_shape: ", labels.shape)

        n_batches = len(train_loader)
        if n_batches < self.config['log_every_n_steps']:
            self.config['log_every_n_steps'] = n_batches
        self.model_config['num_layer'],
        model_2d = GINet_2D(self.config['dataset']['task'], self.model_config['num_layer'], self.model_config['emb_dim'], self.model_config['dropout'], self.model_config['pool']).to(self.device)
        model_2d = self._load_pre_trained_weights(model_2d,'2d')
        model_3d = GINet_3D(self.config['dataset']['task'], self.model_config['num_layer'], self.model_config['emb_dim'], self.model_config['dropout'], self.model_config['pool']).to(self.device)
        model_3d = self._load_pre_trained_weights(model_3d,'3d')
        seq_model = SeqEncoder(self.config['dataset']['task'], config['dataset']['max_len'] , self.model_config['seq_model']).to(self.device)
        seq_model = self._load_pre_trained_weights(seq_model,'seq')
        
        layer_list_2d = []
        for name, param in model_2d.named_parameters():
            if 'output_layers' in name:
                # print(name)
                layer_list_2d.append(name)
        
        layer_list_3d = []
        for name, param in model_3d.named_parameters():
            if 'output_layers' in name:
                # print(name)
                layer_list_3d.append(name)

        layer_list_seq = []
        for name, param in seq_model.named_parameters():
            if 'output_layers' in name:
                # print(name)
                layer_list_seq.append(name)
        return self._test(model_2d, model_3d, seq_model, test_loader,hyperparameter)


    def _load_pre_trained_weights(self, model, index = '2d'):
        try:
            checkpoints_folder = os.path.join(self.config['fine_tune_from'], 'ckpt')
            ckp_path = os.path.join(checkpoints_folder, 'model_'+ index + '.pth')
            state_dict = torch.load(ckp_path, map_location=self.device)
            model.load_my_state_dict(state_dict)
            print("Loaded pre-trained model {} with success.".format(ckp_path))
        
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model_2d, model_3d, model_seq, valid_loader, hyperparameter):
        # test steps
        predictions_2d = []
        predictions_3d = []
        predictions_seq = []
        labels = []
        with torch.no_grad():
            model_2d.eval()
            model_3d.eval()
            model_seq.eval()

            valid_loss = 0.0
            num_data = 0
            for bn, (Seq_feature, Seq_len, g_2d, g_3d, mols, target) in enumerate(valid_loader):
                target = target.to(self.device)
                g_2d = g_2d.to(self.device)
                g_3d = g_3d.to(self.device)
                Seq_feature = Seq_feature.to(self.device)
                Seq_len = Seq_len.to(self.device)

                pred_2d = model_2d(g_2d)
                loss_2d = self._step(model_2d, g_2d, target)
                pred_3d = model_3d(g_3d)
                loss_3d = self._step(model_3d, g_3d, target)
                pred_seq = model_seq(Seq_feature,Seq_len)
                if self.config['dataset']['task'] == 'classification':
                    loss_seq_v = self.criterion(pred_seq, target.view(-1))
                elif self.config['dataset']['task'] == 'regression':
                    if self.normalizer:
                        loss_seq_v = self.criterion(pred_seq, self.normalizer.norm(target.reshape(-1,1)))
                    else:
                        loss_seq_v = self.criterion(pred_seq, target.reshape(-1,1))
                
                loss = hyperparameter[0] * loss_2d + hyperparameter[1] * loss_3d + hyperparameter[2] * loss_seq_v

                valid_loss += loss.item() * target.size(0)
                num_data += target.size(0)

                if self.normalizer:
                    pred_2d = self.normalizer.denorm(pred_2d)
                    pred_3d = self.normalizer.denorm(pred_3d)
                    pred_seq = self.normalizer.denorm(pred_seq)

                if self.config['dataset']['task'] == 'classification':
                    pred_2d = F.softmax(pred_2d, dim=-1)
                    pred_3d = F.softmax(pred_3d, dim=-1)
                    pred_seq = F.softmax(pred_seq, dim=-1)

                if self.device == 'cpu':
                    predictions_2d.extend(pred_2d.detach().numpy())
                    predictions_3d.extend(pred_3d.detach().numpy())
                    predictions_seq.extend(pred_seq.detach().numpy())
                    labels.extend(target.flatten().numpy())
                else:
                    predictions_2d.extend(pred_2d.cpu().detach().numpy())
                    predictions_3d.extend(pred_3d.cpu().detach().numpy())
                    predictions_seq.extend(pred_seq.cpu().detach().numpy())
                    labels.extend(target.cpu().flatten().numpy())

            valid_loss /= num_data
        
        model_2d.train()
        model_3d.train()
        model_seq.train()

        if self.config['dataset']['task'] == 'regression':
            predictions_2d = np.array(predictions_2d)
            predictions_3d = np.array(predictions_3d)
            predictions_seq = np.array(predictions_seq)
            labels = np.array(labels)
            rmse_2d = mean_squared_error(labels, predictions_2d, squared=False)
            mae_2d = mean_absolute_error(labels, predictions_2d)
            rmse_3d = mean_squared_error(labels, predictions_3d, squared=False)
            mae_3d = mean_absolute_error(labels, predictions_3d)
            rmse_seq = mean_squared_error(labels, predictions_seq, squared=False)
            mae_seq = mean_absolute_error(labels, predictions_seq)

            print('Validation loss:', valid_loss, 'RMSE2d:', rmse_2d, 'MAE2d:', mae_2d,'RMSE3d:', rmse_3d, 'MAE3d:', mae_3d,'RMSEseq:', rmse_seq, 'MAEseq:', mae_seq)
            return valid_loss, rmse_2d, mae_2d, rmse_3d, mae_3d, rmse_seq, mae_seq

        elif self.config['dataset']['task'] == 'classification': 
            predictions_2d = np.array(predictions_2d)
            predictions_3d = np.array(predictions_3d)
            predictions_seq = np.array(predictions_seq)
            labels = np.array(labels)
            try:
                roc_auc_2d = roc_auc_score(labels, predictions_2d[:,1])
                roc_auc_3d = roc_auc_score(labels, predictions_3d[:,1])
                roc_auc_seq = roc_auc_score(labels, predictions_seq[:,1])
                print('Validation loss:', valid_loss, 'ROC AUC2d:', roc_auc_2d, 'ROC AUC3d:', roc_auc_3d, 'ROC AUCseq:', roc_auc_seq)
            except ValueError:
                print("ValueError")
                roc_auc_2d = -1
                roc_auc_3d = -1
                roc_auc_seq = -1
            return valid_loss, roc_auc_2d, roc_auc_3d, roc_auc_seq

    def _test(self, model_2d, model_3d, model_seq, test_loader, hyperparameter):
        
        # model_path = 'experiments/pretrained-3000-FreeSolv/Feb26_10-45-54-expt/model_2d.pth'
        model_path = os.path.join(self.log_dir, 'model_2d.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model_2d.load_state_dict(state_dict)
        print("Loaded {} with success.".format(model_path))
        # model_path = 'experiments/pretrained-3000-FreeSolv/Feb26_10-45-54-expt/model_3d.pth'
        model_path = os.path.join(self.log_dir, 'model_3d.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model_3d.load_state_dict(state_dict)
        print("Loaded {} with success.".format(model_path))
        # model_path = 'experiments/pretrained-3000-FreeSolv/Feb26_10-45-54-expt/model_seq.pth'
        model_path = os.path.join(self.log_dir, 'model_seq.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model_seq.load_state_dict(state_dict)
        print("Loaded {} with success.".format(model_path))

        # test steps
        predictions_2d = []
        predictions_3d = []
        predictions_seq = []
        labels = []
        with torch.no_grad():
            model_2d.eval()
            model_3d.eval()
            model_seq.eval()

            test_loss = 0.0
            num_data = 0
            for bn, (Seq_feature, Seq_len, g_2d, g_3d, mols, target) in enumerate(test_loader):
                target = target.to(self.device)
                g_2d = g_2d.to(self.device)
                g_3d = g_3d.to(self.device)
                Seq_feature = Seq_feature.to(self.device)
                Seq_len = Seq_len.to(self.device)

                pred_2d = model_2d(g_2d)
                loss_2d = self._step(model_2d, g_2d, target)
                pred_3d = model_3d(g_3d)
                loss_3d = self._step(model_3d, g_3d, target)
                pred_seq = model_seq(Seq_feature,Seq_len)
                if self.config['dataset']['task'] == 'classification':
                    loss_seq = self.criterion(pred_seq, target.view(-1))
                elif self.config['dataset']['task'] == 'regression':
                    if self.normalizer:
                        loss_seq = self.criterion(pred_seq, self.normalizer.norm(target.reshape(-1,1)))
                    else:
                        loss_seq = self.criterion(pred_seq, target.reshape(-1,1))
                
                loss = hyperparameter[0] * loss_2d + hyperparameter[1] * loss_3d + hyperparameter[2] * loss_seq

                test_loss += loss.item() * target.size(0)
                num_data += target.size(0)

                if self.normalizer:
                    pred_2d = self.normalizer.denorm(pred_2d)
                    pred_3d = self.normalizer.denorm(pred_3d)
                    pred_seq = self.normalizer.denorm(pred_seq)

                if self.config['dataset']['task'] == 'classification':
                    pred_2d = F.softmax(pred_2d, dim=-1)
                    pred_3d = F.softmax(pred_3d, dim=-1)
                    pred_seq = F.softmax(pred_seq, dim=-1)

                if self.device == 'cpu':
                    predictions_2d.extend(pred_2d.detach().numpy())
                    predictions_3d.extend(pred_2d.detach().numpy())
                    predictions_seq.extend(pred_2d.detach().numpy())
                    labels.extend(target.flatten().numpy())
                else:
                    predictions_2d.extend(pred_2d.cpu().detach().numpy())
                    predictions_3d.extend(pred_3d.cpu().detach().numpy())
                    predictions_seq.extend(pred_seq.cpu().detach().numpy())
                    labels.extend(target.cpu().flatten().numpy())

        test_loss /= num_data
        
        model_2d.train()
        model_3d.train()
        model_seq.train()

        if self.config['dataset']['task'] == 'regression':
            predictions_2d = np.array(predictions_2d)
            predictions_3d = np.array(predictions_3d)
            predictions_seq = np.array(predictions_2d)
            labels = np.array(labels)
            rmse_2d = mean_squared_error(labels, predictions_2d, squared=False)
            mae_2d = mean_absolute_error(labels, predictions_2d)
            rmse_3d = mean_squared_error(labels, predictions_3d, squared=False)
            mae_3d = mean_absolute_error(labels, predictions_3d)
            rmse_seq = mean_squared_error(labels, predictions_seq, squared=False)
            mae_seq = mean_absolute_error(labels, predictions_seq)

            print('Test loss:', test_loss, 'RMSE2d:', rmse_2d, 'MAE2d:', mae_2d,'RMSE3d:', rmse_3d, 'MAE3d:', mae_3d,'RMSEseq:', rmse_seq, 'MAEseq:', mae_seq)
            return test_loss, rmse_2d, mae_2d, rmse_3d, mae_3d, rmse_seq, mae_seq


        elif self.config['dataset']['task'] == 'classification': 
            predictions_2d = np.array(predictions_2d)
            predictions_3d = np.array(predictions_3d)
            predictions_seq = np.array(predictions_2d)
            labels = np.array(labels)
            try:
                roc_auc_2d = roc_auc_score(labels, predictions_2d[:,1])
                roc_auc_3d = roc_auc_score(labels, predictions_3d[:,1])
                roc_auc_seq = roc_auc_score(labels, predictions_seq[:,1])
                print('Test loss:', test_loss, 'ROC AUC2d:', roc_auc_2d, 'ROC AUC3d:', roc_auc_3d, 'ROC AUCseq:', roc_auc_seq)
            except ValueError:
                print("ValueError")
                roc_auc_2d = 1
                roc_auc_3d = 1
                roc_auc_seq = 1

            return test_loss, roc_auc_2d, roc_auc_3d, roc_auc_seq


def run(config,Hyperparameter):
    dataset = MoleculeTestDatasetWrapper(config['batch_size'], **config['dataset'])
    fine_tune = FineTune_mix(dataset, config)
    return fine_tune.train(Hyperparameter)
    # return fine_tune.abalation(Hyperparameter)

def get_config():
    # config = yaml.load(open("/home/yrl/muti-view-RL/config_finetune.yaml", "r", encoding='utf-8'), Loader=yaml.FullLoader)
    config = yaml.load(open("config_finetune.yaml", "r", encoding='utf-8'), Loader=yaml.FullLoader)

    if config['task_name'] == 'BBBP':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/bbbp/raw/BBBP.csv'
        config['dataset']['save_path'] = './data/bbbp/'
        config['dataset']['max_len'] = 234
        target_list = ["p_np"]

    elif config['task_name'] == 'Tox21':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/tox21/tox21.csv'
        config['dataset']['save_path'] = './data/tox21/'
        config['dataset']['max_len'] = 267
        target_list = [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", 
            "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ]

    elif config['task_name'] == 'ClinTox':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/clintox/clintox.csv'
        config['dataset']['save_path'] = './data/clintox/'
        config['dataset']['max_len'] = 254
        target_list = ['CT_TOX', 'FDA_APPROVED']

    elif config['task_name'] == 'HIV':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/hiv/HIV.csv'
        config['dataset']['save_path'] = './data/hiv/'
        config['dataset']['max_len'] = 495
        target_list = ["HIV_active"]

    elif config['task_name'] == 'BACE':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/bace/bace.csv'
        config['dataset']['save_path'] = './data/bace/'
        config['dataset']['max_len'] = 194
        target_list = ["Class"]

    elif config['task_name'] == 'SIDER':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/sider/sider.csv'
        config['dataset']['save_path'] = './data/sider/'
        config['dataset']['max_len'] = 982
        target_list = [
            "Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues", "Eye disorders", "Investigations", 
            "Musculoskeletal and connective tissue disorders", "Gastrointestinal disorders", "Social circumstances", 
            "Immune system disorders", "Reproductive system and breast disorders", 
            "Neoplasms benign, malignant and unspecified (incl cysts and polyps)", 
            "General disorders and administration site conditions", 
            "Endocrine disorders", "Surgical and medical procedures", "Vascular disorders", "Blood and lymphatic system disorders", 
            "Skin and subcutaneous tissue disorders", "Congenital, familial and genetic disorders", "Infections and infestations", 
            "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders", "Renal and urinary disorders", 
            "Pregnancy, puerperium and perinatal conditions", "Ear and labyrinth disorders", "Cardiac disorders", 
            "Nervous system disorders", "Injury, poisoning and procedural complications"
        ]
    
    elif config['task_name'] == 'MUV':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/muv/muv.csv'
        config['dataset']['save_path'] = './data/muv/'
        config['dataset']['max_len'] = 84
        target_list = [
            "MUV-466", "MUV-548", "MUV-600", "MUV-644", "MUV-652", "MUV-692", "MUV-712", "MUV-713", 
            "MUV-733", "MUV-737", "MUV-810", "MUV-832", "MUV-846", "MUV-852", "MUV-858", "MUV-859"
        ]

    elif config['task_name'] == 'FreeSolv':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = './data/freesolv/freesolv.csv'
        config['dataset']['save_path'] = './data/freesolv/'
        config['dataset']['max_len'] = 500
        target_list = ["expt"]
    
    elif config["task_name"] == 'ESOL':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = './data/esol/esol.csv'
        config['dataset']['save_path'] = './data/esol/'
        config['dataset']['max_len'] = 500
        target_list = ["measured log solubility in mols per litre"]

    elif config["task_name"] == 'Lipo':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = './data/lipophilicity/Lipophilicity.csv'
        config['dataset']['save_path'] = './data/lipophilicity/'
        config['dataset']['max_len'] = 500
        target_list = ["exp"]

    elif config["task_name"] == 'qm7':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = './data/qm7/qm7.csv'
        config['dataset']['save_path'] = './data/qm7/'
        config['dataset']['max_len'] = 500
        target_list = ["u0_atom"]

    elif config["task_name"] == 'qm8':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = './data/qm8/qm8.csv'
        config['dataset']['save_path'] = './data/qm8/'
        config['dataset']['max_len'] = 500
        target_list = [
            "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0", "f1-PBE0", "f2-PBE0", 
            "E1-CAM", "E2-CAM", "f1-CAM","f2-CAM"
        ]

    else:
        raise ValueError('Unspecified dataset!')

    print(config)
    return config, target_list


if __name__ == '__main__':
    

    config, target_list = get_config()

    os.makedirs('experiments', exist_ok=True)
    dir_name = config['fine_tune_from'].split('/')[0] + '-' + \
        config['fine_tune_from'].split('/')[-1] + '-' + config['task_name']
    save_dir = os.path.join('experiments', dir_name)

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    
    # a = torch.arange(0,1,0.1)
    # b = torch.arange(0,1,0.1)
    # Hyperparameter = []
    # for i in a:
    #     for j in b:
    #         if 1.0-(i+j) > 0: 
    #             Hyperparameter.append([i, j, 1.0-(i+j)])  
    # print(Hyperparameter)
    Hyperparameter = [[0.5,0.4,0.1]]
    
    if config['dataset']['task'] == 'classification':
        for hy in Hyperparameter:
            print("Hyperparameter: ", hy)
            save_list = []
            for target in target_list:
                config['dataset']['target'] = target
                roc_list = [target]
                test_loss, roc_auc_2d, roc_auc_3d, roc_auc_seq = run(config, hy)
                roc_list.append([roc_auc_2d, roc_auc_3d, roc_auc_seq])
                roc_list.append([hy])
                save_list.append(roc_list)
            df = pd.DataFrame(save_list)
            fn = '{}_ROC.csv'.format(config["task_name"])
            # fn = '{}_{}_ROC.csv'.format(config["task_name"], current_time)
            df.to_csv(os.path.join(save_dir, fn), index=False, mode='a', header=['label', 'ROC-AUC','Hyperparameter'])
    
    elif config['dataset']['task'] == 'regression':
        for hy in Hyperparameter:
            print("Hyperparameter: ", hy)
            save_rmse_list, save_mae_list = [], []
            for target in target_list:
                config['dataset']['target'] = target
                rmse_list, mae_list = [target], [target]
                test_loss, rmse_2d, mae_2d, rmse_3d, mae_3d, rmse_seq, mae_seq = run(config, hy)
                rmse_list.append([rmse_2d, rmse_3d, rmse_seq])
                rmse_list.append([hy])
                mae_list.append([mae_2d, mae_3d, mae_seq])
                mae_list.append([hy])
                
                save_rmse_list.append(rmse_list)
                save_mae_list.append(mae_list)

            df = pd.DataFrame(save_rmse_list)
            fn = '{}_RMSE.csv'.format(config["task_name"])
            df.to_csv(os.path.join(save_dir, fn), index=False, mode='a', header=['label', 'RMSE', 'Hyperparameter'])

            df = pd.DataFrame(save_mae_list)
            fn = '{}_MAE.csv'.format(config["task_name"])
            df.to_csv(os.path.join(save_dir, fn), index=False, mode='a', header=['label', 'MAE', 'Hyperparameter'])