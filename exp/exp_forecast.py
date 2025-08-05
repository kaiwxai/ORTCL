import os
import time
import warnings
from tqdm import tqdm
import pickle
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import EarlyStopping, visual, LargeScheduler, attn_map

warnings.filterwarnings('ignore')

import logging



class MultiLayerFeatureExtractor:
    def __init__(self, model, num_layers=7):
        self.model = model
        self.num_layers = num_layers
        self.features = defaultdict(dict)
        self.handles = []

    def __enter__(self):
        for layer_idx in range(self.num_layers):
            decoder_layer = self.model.module.decoder.attn_layers[layer_idx]
            def make_closure(layer_id):
                def attn_hook(module, input, output):
                    self.features[f"layer{layer_id}"]["attn"] = output[0].detach().cpu()
                def proj_hook(module, input, output):
                    self.features[f"layer{layer_id}"]["proj"] = output.detach().cpu()
                return attn_hook, proj_hook

            attn_hook, proj_hook = make_closure(layer_idx)
            self.handles.extend([
                decoder_layer.attention.inner_attention.register_forward_hook(attn_hook),
                decoder_layer.attention.out_projection.register_forward_hook(proj_hook)
            ])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

class Exp_Forecast(Exp_Basic):

    def _build_model(self):
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = self.model_dict[self.args.model].Model(self.args)
            model = DDP(model.cuda(), device_ids=[self.args.local_rank], find_unused_parameters=True)
        else:
            self.args.device = self.device
            model = self.model_dict[self.args.model].Model(self.args)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.use_weight_decay:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                     weight_decay=self.args.weight_decay)
        else:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, epoch=0, flag='vali'):
        total_loss = []
        total_count = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()
                if self.args.output_attention:
                    # output used to calculate loss misaligned patch_len compared to input
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    # only use the forecast window to calculate loss
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if self.args.use_ims:
                    pred = outputs[:, -self.args.seq_len:, :]
                    true = batch_y
                    if flag == 'vali':
                        loss = criterion(pred, true)
                    elif flag == 'test':  # in this case, only pred_len is used to calculate loss
                        pred = pred[:, -self.args.pred_len:, :]
                        true = true[:, -self.args.pred_len:, :]
                        loss = criterion(pred, true)
                else:
                    loss = criterion(outputs[:, -self.args.pred_len:, :], batch_y[:, -self.args.pred_len:, :])

                loss = loss.detach().cpu()
                total_loss.append(loss)
                total_count.append(batch_x.shape[0])
                torch.cuda.empty_cache()

        if self.args.use_multi_gpu:
            total_loss = torch.tensor(np.average(total_loss, weights=total_count)).to(self.device)
            dist.barrier()
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item() / dist.get_world_size()
        else:
            total_loss = np.average(total_loss, weights=total_count)
        self.model.train()
        return total_loss

    def finetune(self, setting):
        # self.model.load_state_dict(torch.load('./checkpoints/etth1_checkpoint.pth'))
        finetune_data, finetune_loader = data_provider(self.args, flag='train')
        vali_data, vali_loader = data_provider(self.args, flag='val')
        test_data, test_loader = data_provider(self.args, flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(finetune_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))
        scheduler = LargeScheduler(self.args, model_optim)

        # 
        for epoch in range(self.args.finetune_epochs):
        # for epoch in range(1):
            iter_count = 0

            loss_val = torch.tensor(0., device="cuda")
            count = torch.tensor(0., device="cuda")

            self.model.train()
            epoch_time = time.time()

            print("Step number per epoch: ", len(finetune_loader))
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(finetune_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if self.args.use_ims:
                    # output used to calculate loss misaligned patch_len compared to input
                    loss = criterion(outputs[:, -self.args.seq_len:, :], batch_y)
                else:
                    # only use the forecast window to calculate loss
                    loss = criterion(outputs[:, -self.args.pred_len:, :], batch_y[:, -self.args.pred_len:, :])

                loss_val += loss
                count += 1

                if i % 50 == 0:
                    cost_time = time.time() - time_now
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f} | cost_time: {3:.0f} | memory: allocated {4:.0f}MB, reserved {5:.0f}MB, cached {6:.0f}MB "
                        .format(i, epoch + 1, loss.item(), cost_time,
                                torch.cuda.memory_allocated() / 1024 / 1024,
                                torch.cuda.memory_reserved() / 1024 / 1024,
                                torch.cuda.memory_cached() / 1024 / 1024))
                    time_now = time.time()

                loss.backward()
                model_optim.step()
                torch.cuda.empty_cache()
            print(iter_count)
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            if self.args.use_multi_gpu:
                dist.barrier()
                dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
            train_loss = loss_val.item() / count.item()

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            if self.args.train_test:
                test_loss = self.vali(test_data, test_loader, criterion, flag='test')
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss))


            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            scheduler.schedule_epoch(epoch)



        if self.args.win_sum == 1 and self.args.windows == 1:       
            best_model_path ='./checkpoints/' + self.args.data +'cross_domain_batch_' + str(self.args.windows) + '_checkpoint.pth'
        else:
            best_model_path ='./checkpoints/' + self.args.data +'_batch_' + str(self.args.windows) + '_checkpoint_full.pth'
        torch.save(self.model.state_dict(), best_model_path)
        
       
        save_dir = "./layer_features/" + self.args.data + "/"
        os.makedirs(save_dir, exist_ok=True)
        all_features = {f"layer{i}": {"attn": [], "proj": []} for i in range(7)}
        weights = {f"layer{i}": {"weight": []} for i in range(7)}
        for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(finetune_loader):
            B = batch_x.shape[0]
            sample_num = max(1, int(0.01 * B))  
            sample_indices = torch.randperm(B)[:sample_num]
            
            batch_x = batch_x[sample_indices].float().to(self.device)
            batch_y = batch_y[sample_indices].float().to(self.device)
            batch_x_mark = batch_x_mark[sample_indices].to(self.device)
            batch_y_mark = batch_y_mark[sample_indices].to(self.device)


            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            dec_inp = torch.cat([
                batch_y[:, :self.args.label_len, :],
                torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
            ], dim=1).float().to(self.device)

            with MultiLayerFeatureExtractor(self.model) as extractor:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                for layer_idx in range(7):
                    layer_key = f"layer{layer_idx}"
                    encoder_layer = self.model.module.backbone.decoder.attn_layers[layer_idx] 
                    if len(weights[layer_key]["weight"]) == 0:
                        weights[layer_key]["weight"].append(encoder_layer.attention.out_projection.weight.data.detach().cpu())
                    bias = encoder_layer.attention.out_projection.bias.data.detach().cpu()
                    all_features[layer_key]["attn"].append(extractor.features[layer_key]["attn"])
                    all_features[layer_key]["proj"].append(extractor.features[layer_key]["proj"]-bias)
                    
                    attn0 = torch.tensor(extractor.features[layer_key]["attn"])
                    B, L, H, D = attn0.shape
                    attn0 = attn0.view(B, L, -1)
                    proj0 = torch.tensor(extractor.features[layer_key]["proj"])
                    weight0 = torch.tensor(weights['layer0']['weight'][0])
                    proj_tmp = torch.matmul(attn0, weight0.T) + bias

            if (batch_idx+1) % 100 == 0:
                save_path = os.path.join(save_dir, f"features_batch_{batch_idx//100}.pkl")
                with open(save_path, "wb") as f:
                    pickle.dump(all_features, f)
                all_features = {f"layer{i}": {"attn": [], "proj": []} for i in range(9)}

        if any(len(v["attn"])>0 for v in all_features.values()):
            if self.args.win_sum == 1 and self.args.windows == 1:
                file_name = self.args.data + '_cross_domain_features_batch_final'+str(self.args.windows)+'.pkl'
            else: 
                file_name = self.args.data + '_features_batch_final'+str(self.args.windows)+'.pkl'
            save_path = os.path.join(save_dir, file_name)
            with open(save_path, "wb") as f:
                pickle.dump(all_features, f)
        
        save_dir = "./layer_features/"+self.args.data+"/"
        if self.args.win_sum == 1 and self.args.windows == 1:
            save_path = save_dir + self.args.data+ '_cross_domain_weight_batch_'+str(self.args.windows)+'.pkl'
        else:
            save_path = save_dir + self.args.data+ '_weight_batch_'+str(self.args.windows)+'.pkl'
        with open(save_path, "wb") as f:
            pickle.dump(weights, f)

        return self.model

    def test(self, setting, test=0):

        # model_path = './checkpoints/' + self.args.data+ '_batch_' + str(self.args.windows) + '_checkpoint_full.pth'

        # self.model.load_state_dict(torch.load(model_path))

        logging_file_name = "./log/" + self.args.data_path + "_train.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s: %(message)s",
            handlers=[
                logging.FileHandler(logging_file_name, encoding="utf-8"),
                logging.StreamHandler()  # 控制台输出
            ]
        )

        # print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))
        attns = []
        folder_path = './test_results/' + setting + '/' + self.args.data_path + '/' + f'{self.args.output_len}/'
        if not os.path.exists(folder_path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(folder_path)
        self.model.eval()
        if self.args.output_len_list is None:
            self.args.output_len_list = [self.args.output_len]

        preds_list = [[] for _ in range(len(self.args.output_len_list))]
        trues_list = [[] for _ in range(len(self.args.output_len_list))]
        self.args.output_len_list.sort()

        with torch.no_grad():
            for output_ptr in tqdm(range(len(self.args.output_len_list))):
                self.args.output_len = self.args.output_len_list[output_ptr]
                test_data, test_loader = data_provider(self.args, flag='test')
                # test_data, test_loader = data_provider(self.args, flag='test', windows=windows, win_sum=win_sum)
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    inference_steps = self.args.output_len // self.args.pred_len
                    dis = self.args.output_len - inference_steps * self.args.pred_len
                    if dis != 0:
                        inference_steps += 1
                    pred_y = []
                    for j in range(inference_steps):
                        if len(pred_y) != 0:
                            batch_x = torch.cat([batch_x[:, self.args.pred_len:, :], pred_y[-1]], dim=1)
                            tmp = batch_y_mark[:, j - 1:j, :]
                            batch_x_mark = torch.cat([batch_x_mark[:, 1:, :], tmp], dim=1)
                        if self.args.output_attention:
                            outputs, attns = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        pred_y.append(outputs[:, -self.args.pred_len:, :])         
                    pred_y = torch.cat(pred_y, dim=1)

                    if dis != 0:
                        pred_y = pred_y[:, :-self.args.pred_len+dis, :]

                    if self.args.use_ims:
                        batch_y = batch_y[:, self.args.label_len:self.args.label_len + self.args.output_len, :].to(
                            self.device)
                    else:
                        batch_y = batch_y[:, :self.args.output_len, :].to(self.device)
                    outputs = pred_y.detach().cpu()
                    batch_y = batch_y.detach().cpu()

                    if test_data.scale and self.args.inverse:
                        shape = outputs.shape
                        outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                    outputs = outputs[:, :, f_dim:]
                    batch_y = batch_y[:, :, f_dim:]

                    pred = outputs
                    true = batch_y

                    preds_list[output_ptr].append(pred)
                    trues_list[output_ptr].append(true)
                    if i % 10 == 0:
                        input = batch_x.detach().cpu().numpy()
                        gt = np.concatenate((input[0, -self.args.pred_len:, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, -self.args.pred_len:, -1], pred[0, :, -1]), axis=0)

                        if self.args.local_rank == 0:
                            # if self.args.output_attention:
                            attn = attns[0].cpu().numpy()[0, 0, :, :]
                            # attn_map(attn, os.path.join(folder_path, f'attn_{self.args.windows}_{i}_{self.args.local_rank}.pdf'))
                            # print('The pdf has been saved')

                            # print(gt.shape)

                            # visual(gt[96:], pd[96:], os.path.join(folder_path, f'{self.args.windows}_{i}_{self.args.local_rank}.pdf'))

                            data_dict = {
                                'attn': attn,
                                'gt': gt[96:],
                                'pd': pd[96:]
                            }

                            pkl_path = os.path.join(folder_path, f'data_{self.args.windows}_{i}_{self.args.local_rank}.pkl')
                            with open(pkl_path, 'wb') as f:
                                pickle.dump(data_dict, f)
                            print(f'Data dictionary saved to {pkl_path}')

        # logging.info(f"--------------------------------")
        if self.args.output_len_list is not None:
            for i in range(len(preds_list)):
                preds = preds_list[i]
                trues = trues_list[i]
                preds = torch.cat(preds, dim=0).numpy()
                trues = torch.cat(trues, dim=0).numpy()
            mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('------------------------mse:{}, mae:{}------------------------'.format(mse, mae))
        logging.info(f"output_len: {self.args.output_len_list[i]}")
        logging.info('mse:{}, mae:{}'.format(mse, mae))
        return

    def knowledge_retention_finetuing(self, setting): 
        model_path = './checkpoints/' + self.args.data+ '_batch_' + str(5) + '_checkpoint.pth'
        self.model.load_state_dict(torch.load(model_path))
        print(model_path)

    def knowledge_retention_editing(self, setting):    
        windows = self.args.windows + 1
        if windows == 2:
            model_path = './checkpoints/' + self.args.data+ '_batch_' + str(self.args.windows) + '_checkpoint.pth'
            self.model.load_state_dict(torch.load(model_path))
            print(model_path)
        file_path_before_feat = './layer_features/'+self.args.data+'/' + self.args.data+ '_features_batch_final'+str(self.args.windows)+'.pkl'
        file_path_before_weight = './layer_features/'+self.args.data+'/' + self.args.data+'_weight_batch_'+str(self.args.windows)+'.pkl'
        file_path_after_feat = './layer_features/'+self.args.data+'/' + self.args.data+'_features_batch_final'+str(windows)+'.pkl'
        with open(file_path_before_feat, 'rb') as file:
            feat_a = pickle.load(file)
        with open(file_path_before_weight, 'rb') as file:
            weight_a = pickle.load(file)
        with open(file_path_after_feat, 'rb') as file:
            feat_b = pickle.load(file)
        def reshape_and_transpose(tensor):
            return tensor.reshape(-1, tensor.shape[-1]).T
        K_a = torch.cat(feat_a['layer0']['attn'], dim=0).to(self.device)
        B, L, H, D = K_a.shape
        K_a = K_a.view(B, L, -1)
        V_a = torch.cat(feat_a['layer0']['proj'], dim=0).to(self.device)
        W = weight_a['layer0']['weight'][0].to(self.device)
        K_b = torch.cat(feat_b['layer0']['attn'], dim=0).to(self.device)
        B, L, H, D = K_b.shape
        K_b = K_b.view(B, L, -1)
        V_b = torch.cat(feat_b['layer0']['proj'], dim=0).to(self.device)
        K_a_mat = reshape_and_transpose(K_a)
        K_b_mat = reshape_and_transpose(K_b)
        V_a_mat = reshape_and_transpose(V_a)
        V_b_mat = reshape_and_transpose(V_b)
        U0, S0, V0h = torch.linalg.svd(W, full_matrices=False)
        rank = 256
        if rank is not None:
            S0 = S0[:rank]
            U0 = U0[:, :rank]
            V0h = V0h[:rank, :]
        V0 = V0h.T  
        P = V0 @ V0.T 
        K = torch.cat((K_a_mat, K_b_mat), dim=1)
        K_proj = P @ K
        _, _, Vh_k = torch.linalg.svd(K_proj.T, full_matrices=True)
        S = Vh_k.T  
        A = W @ S.T @ K_a_mat
        B = W @ S.T @ K_b_mat
        M = V_a_mat @ A.T + V_b_mat @ B.T
        U_M, _, Vh_M = torch.linalg.svd(M, full_matrices=True)
        R = U_M @ Vh_M
        W_star = R @ W @ S.T
        del K_proj, A, B, M
        encoder_layer = self.model.module.backbone.decoder.attn_layers[0]
        encoder_layer.attention.out_projection.weight = torch.nn.Parameter(W_star)
        return self.model

    def finedit(self, setting):    
        if self.args.windows == 2:
            model_path = './checkpoints/' + self.args.data+ '_batch_' + str(self.args.windows) + '_checkpoint_full.pth'
            # print(model_path)
            self.model.load_state_dict(torch.load(model_path))
        file_path_before_feat = './layer_features/'+self.args.data+'/' + self.args.data+ '_features_batch_final'+str(self.args.windows - 1)+'.pkl'
        file_path_before_weight = './layer_features/'+self.args.data+'/' + self.args.data+'_weight_batch_'+str(self.args.windows - 1)+'.pkl'
        file_path_after_feat = './layer_features/'+self.args.data+'/' + self.args.data+'_features_batch_final'+str(self.args.windows)+'.pkl'
        # file_path_after_weight = './layer_features/'+self.args.data+'/' + self.args.data+'_weight_batch_'+str(self.args.windows)+'.pkl'

        with open(file_path_before_feat, 'rb') as file:
            feat_a = pickle.load(file)

        with open(file_path_before_weight, 'rb') as file:
            weight_a = pickle.load(file)

        with open(file_path_after_feat, 'rb') as file:
            feat_b = pickle.load(file)

        # with open(file_path_after_weight, 'rb') as file:
            # weight_b = pickle.load(file)

        K_a = torch.cat(feat_a['layer0']['attn'], dim=0).to(self.device)
        B, L, H, D = K_a.shape
        K_a = K_a.view(B, L, -1)
        V_a = torch.cat(feat_a['layer0']['proj'], dim=0).to(self.device)
        W = weight_a['layer0']['weight'][0].to(self.device)
        # print(weight_a['layer0']['weight'][0].shape)
        # print(feat_a['layer0']['attn'][0].shape)
        # print(feat_a['layer0']['proj'][0].shape)




        K_b = torch.cat(feat_b['layer0']['attn'], dim=0).to(self.device)
        B, L, H, D = K_b.shape
        K_b = K_b.view(B, L, -1)
        V_b = torch.cat(feat_b['layer0']['proj'], dim=0).to(self.device)


        feat_dim = W.shape[0]

        def validate_orthogonal(matrix, name):
            """验证矩阵的正交性（支持GPU）"""
            identity = torch.eye(matrix.size(1), device=self.device)
            orth_error = torch.norm(matrix.T @ matrix - identity, p='fro')
            print(f"{name} Orthogonality Error: {orth_error.item():.6e}")



        def reshape_3d_to_2d(tensor):
            return tensor.permute(2, 0, 1).reshape(feat_dim, -1)

        K_a_2d = reshape_3d_to_2d(K_a)  # [1024, 2048*7=14336]
        K_b_2d = reshape_3d_to_2d(K_b)

        V_a_2d = reshape_3d_to_2d(V_a)
        V_b_2d = reshape_3d_to_2d(V_b)


        U0, S0, V0_T = torch.linalg.svd(W, full_matrices=False)
        print('------SVD-----------')
        V0 = V0_T.T  
        K_combined = torch.cat([K_a_2d, K_b_2d], dim=1) 
        K_proj = V0 @ (V0_T @ K_combined) 

        U_K, S_K, X_T = torch.linalg.svd(K_proj.T, full_matrices=False)
        print('------SVD-----------')
        S = X_T.T  
        validate_orthogonal(S, "S Matrix")

        A = W @ (S.T @ K_a_2d)  
        B = W @ (S.T @ K_b_2d)

        M = V_a_2d @ A.T + V_b_2d @ B.T + 1e-6*torch.eye(1024, device=self.device)
        U_M, S_M, V_M_T = torch.linalg.svd(M)
        print('------SVD-----------')
        R = U_M @ V_M_T
        validate_orthogonal(R, "R Matrix")

        W_star = R @ (W @ S.T) 

        del K_proj, A, B, M

        encoder_layer = self.model.module.backbone.decoder.attn_layers[0]
        encoder_layer.attention.out_projection.weight = torch.nn.Parameter(W_star)
        return self.model

    def cross_editing(self, setting):    
        model_path = './checkpoints/ETTh1cross_domain_batch_1_checkpoint.pth'
        # model_path = './checkpoints/ETTh1_to_ECL_to_weather_checkpoint.pth'

        self.model.load_state_dict(torch.load(model_path))
        print(model_path)

        file_path_before_feat = './layer_features/ETTh1/ETTh1_cross_domain_features_batch_final1.pkl'
        file_path_before_weight = './layer_features/ETTh1/ETTh1_cross_domain_weight_batch_1.pkl'
        file_path_after_feat = './layer_features/electricity/electricity_cross_domain_features_batch_final1.pkl'

        with open(file_path_before_feat, 'rb') as file:
            feat_a = pickle.load(file)
        with open(file_path_before_weight, 'rb') as file:
            weight_a = pickle.load(file)
        with open(file_path_after_feat, 'rb') as file:
            feat_b = pickle.load(file)

        
        K_a = torch.cat(feat_a['layer0']['attn'], dim=0).to(self.device)
        B, L, H, D = K_a.shape
        K_a = K_a.view(B, L, -1)
        V_a = torch.cat(feat_a['layer0']['proj'], dim=0).to(self.device)
        W = weight_a['layer0']['weight'][0].to(self.device)

        K_b = torch.cat(feat_b['layer0']['attn'], dim=0).to(self.device)
        B, L, H, D = K_b.shape
        K_b = K_b.view(B, L, -1)
        V_b = torch.cat(feat_b['layer0']['proj'], dim=0).to(self.device)

        feat_dim = W.shape[0]

        def reshape_3d_to_2d(tensor):
            return tensor.permute(2, 0, 1).reshape(feat_dim, -1)

        K_a_2d = reshape_3d_to_2d(K_a)  # [1024, 2048*7=14336]
        K_b_2d = reshape_3d_to_2d(K_b)
        V_a_2d = reshape_3d_to_2d(V_a)
        V_b_2d = reshape_3d_to_2d(V_b)

        U0, S0, V0_T = torch.linalg.svd(W, full_matrices=False)
        V0 = V0_T.T 

        K_combined = torch.cat([K_a_2d, K_b_2d], dim=1)  # [1024, 14336*2]
        K_proj = V0 @ (V0_T @ K_combined)  

        U_K, S_K, X_T = torch.linalg.svd(K_proj.T, full_matrices=False)
        S = X_T.T 

        A = W @ (S.T @ K_a_2d)
        B = W @ (S.T @ K_b_2d)

        M = V_a_2d @ A.T + V_b_2d @ B.T + 1e-6 * torch.eye(W.shape[0], device=self.device)
        U_M, S_M, V_M_T = torch.linalg.svd(M)
        target_rank = 256
        # if target_rank is not None:
        U_M = U_M[:, :target_rank]
        S_M = S_M[:target_rank]
        V_M_T = V_M_T[:target_rank, :]
        R = U_M @ V_M_T
        W_star = R @ (W @ S.T)

        # best_model_path ='./checkpoints/ETTh1_to_ECL_to_traffic_checkpoint.pth'
        best_model_path = './checkpoints/ETTh1_to_ECL_checkpoint.pth'
        torch.save(self.model.state_dict(), best_model_path)
        
        encoder_layer = self.model.module.backbone.decoder.attn_layers[0]
        encoder_layer.attention.out_projection.weight = torch.nn.Parameter(W_star)

        

        return self.model

    def finetune_all(self, setting):
        # self.model.load_state_dict(torch.load('./checkpoints/etth1_checkpoint.pth'))
        finetune_data, finetune_loader = data_provider(self.args, flag='train')
        vali_data, vali_loader = data_provider(self.args, flag='val')
        test_data, test_loader = data_provider(self.args, flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(finetune_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))
        scheduler = LargeScheduler(self.args, model_optim)

        # 
        for epoch in range(self.args.finetune_epochs):
        # for epoch in range(1):
            iter_count = 0

            loss_val = torch.tensor(0., device="cuda")
            count = torch.tensor(0., device="cuda")

            self.model.train()
            epoch_time = time.time()

            print("Step number per epoch: ", len(finetune_loader))
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(finetune_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if self.args.use_ims:
                    # output used to calculate loss misaligned patch_len compared to input
                    loss = criterion(outputs[:, -self.args.seq_len:, :], batch_y)
                else:
                    # only use the forecast window to calculate loss
                    loss = criterion(outputs[:, -self.args.pred_len:, :], batch_y[:, -self.args.pred_len:, :])

                loss_val += loss
                count += 1

                if i % 50 == 0:
                    cost_time = time.time() - time_now
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f} | cost_time: {3:.0f} | memory: allocated {4:.0f}MB, reserved {5:.0f}MB, cached {6:.0f}MB "
                        .format(i, epoch + 1, loss.item(), cost_time,
                                torch.cuda.memory_allocated() / 1024 / 1024,
                                torch.cuda.memory_reserved() / 1024 / 1024,
                                torch.cuda.memory_cached() / 1024 / 1024))
                    time_now = time.time()

                loss.backward()
                model_optim.step()
                torch.cuda.empty_cache()
            print(iter_count)
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            if self.args.use_multi_gpu:
                dist.barrier()
                dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
            train_loss = loss_val.item() / count.item()

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            if self.args.train_test:
                test_loss = self.vali(test_data, test_loader, criterion, flag='test')
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss))


            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            scheduler.schedule_epoch(epoch)



        if self.args.win_sum == 1 and self.args.windows == 1:       
            best_model_path ='./checkpoints/' + self.args.data +'cross_domain_batch_' + str(self.args.windows) + '_checkpoint.pth'
        else:
            best_model_path ='./checkpoints/' + self.args.data +'_batch_' + str(self.args.windows) + '_checkpoint_full.pth'
        torch.save(self.model.state_dict(), best_model_path)
        

        return self.model