from models import clustered_transformer
import torch
import torch.nn as nn
from torch import optim
from data_provider.data_loader import DataLoaderCreate
import os
import time
import numpy as np
from utils.tools import EarlyStopping, adjust_learning_rate

class Exp(object):
    def __init__(self, setting):
        self.setting = setting

        self.model_dict = {
            'clustered_transformer': clustered_transformer
        }

        model_params = {'seq_len': self.setting['seq_len'],
            'd_model': self.setting['d_model'],
            'pred_len': self.setting['pred_len'],
            'num_vars': len(self.setting['variables_list']),
            'e_layers': self.setting['e_layers'],
            'norm_flag': self.setting['normalization_flag']
        }

        self.model = self.model_dict[self.setting['model_type']].Model(model_params)

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr = self.setting['learning_rate'])
        return model_optim
    
    def _get_data(self, flag):
        data_set, data_loader = DataLoaderCreate(self.setting, flag)
        return data_set, data_loader

    def validation(self, val_data, val_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (input_arr, output_arr) in enumerate(val_loader):
                input_arr = input_arr.float()
                output_arr = output_arr.float()
                batch_size, num_var, s_len, num_lats, num_lons = input_arr.shape
                pred_output = self.model(input_arr) 
                
                prediction = pred_output.detach().cpu()
                ground_truth = output_arr.detach().cpu()

                loss = criterion(prediction, ground_truth)
                total_loss.append(loss) # not loss.item()??
            
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, exp_name):
        train_data, train_loader = self._get_data(flag = 'train')
        validation_data, validation_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        # for _, (input_arr, output_arr) in enumerate(train_loader):
        #     print(input_arr.shape, ":", output_arr.shape)
        #     break

        # for _, (input_arr, output_arr) in enumerate(test_loader):
        #     print(input_arr.shape, ":", output_arr.shape)
        #     break
        # for _, (input_arr, output_arr) in enumerate(vali_loader):
        #     print(input_arr.shape, ":", output_arr.shape)
        #     break

        # print(len(train_data), ":", len(test_data), ":", len(vali_data))    
        path = os.path.join(self.setting['checkpoints'], exp_name)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.setting['patience'], verbose = True)

        # early stopping and adjustable learning rate - implement later.
        model_optim = self._select_optimizer()
        loss_criterion = self._select_criterion()

        for epoch in range(self.setting['num_epochs']):
            iteration_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (input_arr, output_arr) in enumerate(train_loader):
                iteration_count += 1
                model_optim.zero_grad()
                input_arr = input_arr.float()
                output_arr = output_arr.float()
                batch_size, num_var, s_len, num_lats, num_lons = input_arr.shape
                pred_output = self.model(input_arr) 
                loss = loss_criterion(pred_output, output_arr)
                train_loss.append(loss.item())
            
                if (i + 1) % 10 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {1:.7f}".format(i+1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iteration_count
                    left_time = speed * ((self.setting['num_epochs'] - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iteration_count = 0
                    time_now = time.time()

                speed = (time.time() - time_now) / iteration_count
                loss.backward()
                model_optim.step()
                break

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))            
            train_loss = np.average(train_loss)
            validation_loss = self.validation(validation_data, validation_loader, loss_criterion)
            test_loss = self.validation(test_data, test_loader, loss_criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Val Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, validation_loss, test_loss))
            early_stopping(validation_loss, self.model, path)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.setting)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model



