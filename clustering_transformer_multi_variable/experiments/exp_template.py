from models import clustered_transformer
import torch
import torch.nn as nn
from torch import optim
from data_provider.data_loader import DataLoaderCreate
import os
import time
import numpy as np
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

class Exp(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model_dict = {
            'clustered_transformer': clustered_transformer
        }

        model_params = {'seq_len': self.args.seq_len,
            'd_model': self.args.d_model,
            'pred_len': self.args.pred_len,
            'e_layers': self.args.e_layers,
            'norm_flag': self.args.normalization_flag,
            'device': self.device,
            'num_vars': self.args.num_vars,
            'attention_masking': self.args.attention_masking,
            'time_enc': self.args.time_enc,
            'wavelet_transformation': self.args.wavelet_transformation   
        }

        self.model = self.model_dict[self.args.model_type].Model(model_params)
        self.model = self.model.to(self.device) #moving model to gpu.

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr = self.args.learning_rate)
        return model_optim
    
    def _get_data(self, flag):
        data_set, data_loader = DataLoaderCreate(self.args, flag)
        return data_set, data_loader

    def validation(self, val_data, val_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (input_arr, input_enc_arr, output_arr, output_enc_arr) in enumerate(val_loader):
                
                input_arr = input_arr.float().to(self.device)
                output_arr = output_arr.float().to(self.device)
                input_enc_arr = input_enc_arr.float().to(self.device)
                output_enc_arr = output_enc_arr.float().to(self.device)

                pred_output = self.model(input_arr, input_enc_arr) 
                
                pred_output = pred_output.detach().cpu()
                output_arr = output_arr.detach().cpu()

                loss = criterion(pred_output, output_arr)
                total_loss.append(loss) 

                input_arr = input_arr.cpu()
                input_enc_arr = input_enc_arr.cpu()
                output_enc_arr = output_enc_arr.cpu() 

                del input_arr, output_arr, input_enc_arr, output_enc_arr, pred_output

        
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, exp_name):

        # print('hello1')
        train_data, train_loader = self._get_data(flag = 'train')
        # print('hello2')
        validation_data, validation_loader = self._get_data(flag = 'val')
        # print('hello3')
        test_data, test_loader = self._get_data(flag = 'test')
        # print('hello4')
        
        # print(len(train_data), ":", len(validation_data), ":", len(test_data))
        # return 
        
        # for _, (input_arr, output_arr) in enumerate(train_loader):
        #     print(input_arr.shape, ":", output_arr.shape)
        #     break

        # for _, (input_arr, output_arr) in enumerate(test_loader):
        #     print(input_arr.shape, ":", output_arr.shape)
        #     break
 
        # for _, (input_arr, output_arr) in enumerate(vali_loader):
        #     print(input_arr.shape, ":", output_arr.shape)
        #     break

        path = os.path.join(self.args.checkpoints, exp_name )
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)

        early_stopping = EarlyStopping(self.args.patience, verbose = True)

        # early stopping and adjustable learning rate - implement later.
        model_optim = self._select_optimizer()
        loss_criterion = self._select_criterion()

        for epoch in range(self.args.num_epochs):
            iteration_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            # print('hello5')
            # stime = time.time()
            for i, (input_arr, input_enc_arr, output_arr, output_enc_arr) in enumerate(train_loader):
                # print('hello6')
                # shapes: (b,l,v,s),  (b,4,v,s), (b,l,v,p), (b,4,v,p)
                
                # etime = time.time()
                # print(i, 'data loading time = ', etime - stime)

                iteration_count += 1
                model_optim.zero_grad()
                
                input_arr = input_arr.float().to(self.device) #all float32
                output_arr = output_arr.float().to(self.device)
                input_enc_arr = input_enc_arr.float().to(self.device)
                output_enc_arr = output_enc_arr.float().to(self.device)

                # print(type(input_arr), type(input_enc_arr)) # <torch.Tensor>
                                
                # print(input_arr.shape, ":" , output_arr.shape) # (2, 20, 2304) #(2, 5, 2304)
                # (b, s, l) -> (b, p, l)

                pred_output = self.model(input_arr, input_enc_arr) 
            
                # pred_output = pred_output.detach().cpu().numpy()
                # output_arr = output_arr.detach().cpu().numpy()

                # print(pred_output.shape, ":", output_arr.shape)

                loss = loss_criterion(pred_output, output_arr)
                train_loss.append(loss.item())
                
                input_arr = input_arr.cpu()
                output_arr = output_arr.cpu()
                pred_output = pred_output.cpu()
                input_enc_arr = input_enc_arr.cpu()
                output_enc_arr = output_enc_arr.cpu() 
                
                del input_arr, output_arr, input_enc_arr, output_enc_arr, pred_output

                if (i + 1) % 10 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {1:.7f}".format(i+1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iteration_count
                    left_time = speed * ((self.args.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iteration_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()
            
                # stime = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))            
            train_loss = np.average(train_loss)
            validation_loss = self.validation(validation_data, validation_loader, loss_criterion)
            # test_loss = self.validation(test_data, test_loader, loss_criterion)

            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Val Loss: {3:.7f} Test Loss: {4:.7f}".format(
            #     epoch + 1, train_steps, train_loss, validation_loss, test_loss))
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Val Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, validation_loss))
            
            early_stopping(validation_loss, self.model, path)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
            adjust_learning_rate(model_optim, epoch + 1, self.setting)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, exp_name):
        test_data, test_loader = self._get_data(flag = 'test')
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + exp_name, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + exp_name + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (input_arr, input_enc_arr, output_arr, output_enc_arr) in enumerate(test_loader):
                
                input_arr = input_arr.float().to(self.device)
                output_arr = output_arr.float().to(self.device)
                input_enc_arr = input_enc_arr.float().to(self.device)
                output_enc_arr = output_enc_arr.float().to(self.device)

                pred_output = self.model(input_arr, input_enc_arr)

                pred_output = pred_output.detach().cpu().numpy()
                output_arr = output_arr.detach().cpu().numpy()
            
                preds.append(pred_output)
                trues.append(output_arr)
                
                input_arr = input_arr.cpu()
                input_enc_arr = input_enc_arr.cpu()
                output_enc_arr = output_enc_arr.cpu() 
                
                del input_arr, output_arr, input_enc_arr, output_enc_arr, pred_output
                
        preds = np.array(preds)
        trues = np.array(trues)

        # mae, mse, rmse, mape, mspe = metric(preds, trues)
        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))

        mae, mse, rmse = metric(preds, trues)
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse]))

        print(folder_path)
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return 

            