import math
from itertools import cycle

import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from Loss.LearnMarineTripleLoss import ImprovedTripletLoss
from Loss.MarginEnhancementLoss import MarginEnLoss
from Loss.TripetLossPlus import TripletLossNoise
from Loss.TripletLossBase import TripletLoss
from Model.CNN import CNNModel_MultiDomainModel
from Model.LSTM import LSTM_MultiDomainModel
from Model.ResNet import ResNet_18_MultiDomainModel
from Model.Transformer import Transformer_MultiDomainModel
from tool.DataProcess import Read_Data_From_HUST, Read_Data_From_JNU, Read_Data_From_HIT, read_OrginCWRU
from tool.SampleBuild import TraditionalBuildTriadGroup, OutputMeanSample
from tool.addnoise import addNoiseBatch


def wasserstein_loss(source_features, target_features):
    # 
    loss = torch.mean(source_features - target_features) ** 2
    return loss


def calculate_accuracy(A, B, C):
    """
    A: (N, 1024) 类别标记向量，N为类别数
    B: (X, 1024) 样本向量，X为样本数
    C: (X) 样本的真实标签

    返回: 准确率 (float)
    """
    # 
    distances = torch.cdist(B, A, p=2)  # (X, N)

    # 
    predicted_labels = torch.argmin(distances, dim=1)  # (X)

    # 
    correct_predictions = (predicted_labels == C).sum().item()
    accuracy = correct_predictions / len(C)

    return accuracy


def MDRL_Train(model, cls_Loss,
               train_loader_D1, train_loader_D2, train_loader_D3, test_loader_D1, test_loader_D2, test_loader_D3,
               max_length,num_epochs,lambda_param=1.0, BestPath=f'BestResult/Pre_w.pth',
               device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 
    # 
    max_lr = 0.0001
    min_lr = 0.00001
    optimizer = optim.AdamW(model.parameters(), lr=max_lr)
    Loss_cc = cls_Loss
    model.to(device)
    for epoch in range(num_epochs):
        current_Train_loss = 9999.0
        print(f'\r Progress: {epoch + 1}/{num_epochs}', end='\n')
        # 
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(epoch / num_epochs * math.pi))
        # 
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 
        model.train()
        #  D1 
        train_losses = []
        ACC_train_1 = 0
        ACC_train_2 = 0
        ACC_train_3 = 0
        Train_loss = 0.0
        Domain_Loss = 0.0
        Cls_Loss = 0.0
        A_D1, A_D2, A_D3 = None, None, None
        A_D1_label, A_D2_label, A_D3_label = None, None, None
        for _ in range(max_length):
            inputs_D1, labels_D1 = next(train_loader_D1)  #  next  batch
            # 
            optimizer.zero_grad()
            features_1, outputs_1, margin_bounds_1 = model(inputs_D1, domain=1)
            # 
            gaussian_prior_1 = torch.randn_like(features_1).to(device)
            wasserstein_loss_1 = torch.mean(torch.abs(features_1 - gaussian_prior_1))
            A_D1, P_D1, N_D1, A_D1_label = TraditionalBuildTriadGroup(outputs_1, labels_D1, PK=1, NK=4)
            loss_1_Trp = Loss_cc(A_D1, P_D1, N_D1, A_D1_label, margin_bounds_1)
            # loss_1_Trp = Loss_cc(outputs_1, labels_D1)
            ACC_train_1 += calculate_accuracy(A_D1, outputs_1, labels_D1)

            #  D2 
            inputs_D2, labels_D2 = next(train_loader_D2)
            features_2, outputs_2, margin_bounds_2 = model(inputs_D2, domain=2)
            # 
            gaussian_prior_2 = torch.randn_like(features_2).to(device)
            wasserstein_loss_2 = torch.mean(torch.abs(features_2 - gaussian_prior_2))
            A_D2, P_D2, N_D2, A_D2_label = TraditionalBuildTriadGroup(outputs_2, labels_D2, PK=1, NK=4)
            loss_2_Trp = Loss_cc(A_D2, P_D2, N_D2, A_D2_label, margin_bounds_2)
            # loss_2_Trp = Loss_cc(outputs_2, labels_D2)
            ACC_train_2 += calculate_accuracy(A_D2, outputs_2, labels_D2)

            #  D3 
            inputs_D3, labels_D3 = next(train_loader_D3)
            features_3, outputs_3, margin_bounds_3 = model(inputs_D3, domain=3)
            # 
            gaussian_prior_3 = torch.randn_like(features_3).to(device)
            wasserstein_loss_3 = torch.mean(torch.abs(features_3 - gaussian_prior_3))
            A_D3, P_D3, N_D3, A_D3_label = TraditionalBuildTriadGroup(outputs_3, labels_D3, PK=1, NK=4)
            loss_3_Trp = Loss_cc(A_D3, P_D3, N_D3, A_D3_label, margin_bounds_3)
            # loss_3_Trp = Loss_cc(outputs_3, labels_D3)
            ACC_train_3 += calculate_accuracy(A_D3, outputs_3, labels_D3)

            # 
            wasserstein_loss = wasserstein_loss_1 + wasserstein_loss_2 + wasserstein_loss_3
            Domain_Loss += wasserstein_loss.item()

            # 
            exp_w_1 = torch.exp(-lambda_param * wasserstein_loss_1)
            exp_w_2 = torch.exp(-lambda_param * wasserstein_loss_2)
            exp_w_3 = torch.exp(-lambda_param * wasserstein_loss_3)
            # 
            total_exp = exp_w_1 + exp_w_2 + exp_w_3
            # 
            weight_1 = exp_w_1 / total_exp
            weight_2 = exp_w_2 / total_exp
            weight_3 = exp_w_3 / total_exp

            # cls_loss = loss_1_Trp + loss_2_Trp + loss_3_Trp
            cls_loss = weight_1 * loss_1_Trp + weight_2 * loss_2_Trp + weight_3 * loss_3_Trp
            Cls_Loss += cls_loss.item()
            Loss = cls_loss + wasserstein_loss
            Train_loss += Loss.item()

            Loss.backward()
            optimizer.step()

        Train_loss /= max_length
        Domain_Loss /= max_length
        Cls_Loss /= max_length
        train_losses.append(Train_loss)
        ACC_train_1 /= max_length
        ACC_train_2 /= max_length
        ACC_train_3 /= max_length

        # 
        model.eval()
        if Train_loss <= current_Train_loss:
            print("保存参数")
            current_Train_loss = Train_loss
            torch.save(model.feature_extractor.state_dict(), BestPath)

        val_loss_1, val_loss_2, val_loss_3 = 0.0, 0.0, 0.0
        val_W_loss_1, val_W_loss_2, val_W_loss_3 = 0.0, 0.0, 0.0
        ACC_Test_1 = 0
        ACC_Test_2 = 0
        ACC_Test_3 = 0

        with torch.no_grad():
            for inputs, labels in test_loader_D1:
                inputs, labels = inputs.to(device), labels.to(device)
                features, outputs, margin_bounds = model(inputs, domain=1)
                gaussian_prior = torch.randn_like(features).to(device)
                wasserstein_loss_1 = torch.mean(torch.abs(features - gaussian_prior))
                val_W_loss_1 += wasserstein_loss_1.item()

                ACC_Test_1 += calculate_accuracy(A_D1, outputs, labels)

            for inputs, labels in test_loader_D2:
                inputs, labels = inputs.to(device), labels.to(device)
                features, outputs, margin_bounds = model(inputs, domain=2)
                gaussian_prior = torch.randn_like(features).to(device)
                wasserstein_loss_2 = torch.mean(torch.abs(features - gaussian_prior))
                val_W_loss_2 += wasserstein_loss_2.item()
                ACC_Test_2 += calculate_accuracy(A_D2, outputs, labels)

            for inputs, labels in test_loader_D3:
                inputs, labels = inputs.to(device), labels.to(device)
                features, outputs, margin_bounds = model(inputs, domain=3)
                gaussian_prior = torch.randn_like(features).to(device)
                wasserstein_loss_3 = torch.mean(torch.abs(features - gaussian_prior))
                val_W_loss_3 += wasserstein_loss_3.item()
                ACC_Test_3 += calculate_accuracy(A_D3, outputs, labels)

            ACC_Test_1 /= len(test_loader_D1)
            ACC_Test_2 /= len(test_loader_D2)
            ACC_Test_3 /= len(test_loader_D3)
            val_W_loss_1 /= len(test_loader_D1)
            val_W_loss_2 /= len(test_loader_D2)
            val_W_loss_3 /= len(test_loader_D3)
            Test_W_loss = val_W_loss_1 + val_W_loss_2 + val_W_loss_3

            print(f'Epoch [{epoch + 1}/{num_epochs}]')
            print(
                f'Training Loss: {Train_loss:.4f}, Training Accuracy: {ACC_train_1:.4f}-{ACC_train_2:.4f}-{ACC_train_3:.4f}')
            print(f'Train Domain Loss: {Domain_Loss:.4f}, Train Cls Loss: {Cls_Loss:.4f}')
            print(f'Margin_1:{margin_bounds_1}')
            print(f' Test Accuracy: {ACC_Test_1:.4f}-{ACC_Test_2:.4f}-{ACC_Test_3:.4f}')
            print(f'Test Domain Loss: {Test_W_loss:.4f}')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer_MultiDomainModel(num_classes_1=4, num_classes_2=10, num_classes_3=3)
    # model = CNNModel_MultiDomainModel(num_classes_1=4, num_classes_2=10, num_classes_3=3)
    snr = -10
    # Train_D1_X, Train_D1_Y, _, _, Test_D1_X, Test_D1_Y = Read_Data_From_HUST(filepath='data/HUST',
    #                                                                         SampleLength=1024,
    #                                                                         SampleNum=400,
    #                                                                         Rate=[0.8, 0.1, 0.1],
    #                                                                         stander=False
    #                                                                         )

    Train_D1_X, Train_D1_Y, _, _, Test_D1_X, Test_D1_Y = Read_Data_From_JNU(filepath='data/JNU',
                                                                            SampleLength=1024,
                                                                            SampleNum=400,
                                                                            Rate=[0.8, 0.1, 0.1],)
    # Train_D1_X = addNoiseBatch(Train_D1_X, snr)
    # Train_D1_X= addNoiseBatch(Train_D1_X, snr)

    # 11

    # Train_D2_X, Train_D2_Y, _, _, Test_D2_X, Test_D2_Y = Read_Data_From_JNU(filepath='data/JNU',
    #                                                                         SampleLength=1024,
    #                                                                         SampleNum=400,
    #                                                                         Rate=[0.8, 0.1, 0.1],)
    # Train_D2_X = addNoiseBatch(Train_D2_X, snr)
    # Train_D2_X= addNoiseBatch(Train_D2_X, snr)

    Train_D2_X, Train_D2_Y, _, _, Test_D2_X, Test_D2_Y = read_OrginCWRU(filepath='data/CWRU', SampleNum=200, SampleLength=1024, normal=False, Rate=[0.8, 0.1, 0.1])

    # 4

    Train_D3_X, Train_D3_Y, _, _, Test_D3_X, Test_D3_Y = Read_Data_From_HIT(filepath='data/HIT',
                                                                            SampleLength=1024,
                                                                            SampleNum=400,
                                                                            Rate=[0.8, 0.1, 0.1],
                                                                            )

    # Train_D3_X = addNoiseBatch(Train_D3_X, snr)
    # Train_D3_X= addNoiseBatch(Train_D3_X, snr)

    Train_D1_X = torch.tensor(Train_D1_X, dtype=torch.float32).to(device)

    Train_D1_Y = torch.tensor(Train_D1_Y, dtype=torch.int64).to(device)
    Test_D1_X = torch.tensor(Test_D1_X, dtype=torch.float32).to(device)
    Test_D1_Y = torch.tensor(Test_D1_Y, dtype=torch.int64).to(device)
    mean = Train_D1_X.mean()
    std = Train_D1_X.std()
    # Train_D1_X = (Train_D1_X - mean) / std
    # Test_D1_X = (Test_D1_X - mean) / std

    Train_D2_X = torch.tensor(Train_D2_X, dtype=torch.float32).to(device)
    Train_D2_Y = torch.tensor(Train_D2_Y, dtype=torch.int64).to(device)
    Test_D2_X = torch.tensor(Test_D2_X, dtype=torch.float32).to(device)
    Test_D2_Y = torch.tensor(Test_D2_Y, dtype=torch.int64).to(device)

    mean = Train_D2_X.mean()
    std = Train_D2_X.std()
    # Train_D2_X = (Train_D2_X - mean) / std
    # Test_D2_X = (Test_D2_X - mean) / std

    Train_D3_X = torch.tensor(Train_D3_X, dtype=torch.float32).to(device)
    Train_D3_Y = torch.tensor(Train_D3_Y, dtype=torch.int64).to(device)
    Test_D3_X = torch.tensor(Test_D3_X, dtype=torch.float32).to(device)
    Test_D3_Y = torch.tensor(Test_D3_Y, dtype=torch.int64).to(device)
    mean = Train_D3_X.mean()
    std = Train_D3_X.std()
    # Train_D3_X = (Train_D3_X - mean) / std
    # Test_D3_X = (Test_D3_X - mean) / std

    print(Train_D1_X.shape)
    print(Train_D2_X.shape)
    print(Train_D3_X.shape)

    #  DataLoader
    train_loader_D1 = DataLoader(TensorDataset(Train_D1_X, Train_D1_Y), batch_size=128, shuffle=True)
    train_loader_D2 = DataLoader(TensorDataset(Train_D2_X, Train_D2_Y), batch_size=128, shuffle=True)
    train_loader_D3 = DataLoader(TensorDataset(Train_D3_X, Train_D3_Y), batch_size=128, shuffle=True)
    test_loader_D1 = DataLoader(TensorDataset(Test_D1_X, Test_D1_Y), batch_size=64, shuffle=False)
    test_loader_D2 = DataLoader(TensorDataset(Test_D2_X, Test_D2_Y), batch_size=64, shuffle=False)
    test_loader_D3 = DataLoader(TensorDataset(Test_D3_X, Test_D3_Y), batch_size=64, shuffle=False)

    max_length = max(len(train_loader_D1), len(train_loader_D2), len(train_loader_D3))
    train_loader_D1 = cycle(train_loader_D1)
    train_loader_D2 = cycle(train_loader_D2)
    train_loader_D3 = cycle(train_loader_D3)


    # cls_Loss = TripletLossNoise()
    # cls_Loss = ImprovedTripletLoss(alpha=0.4)
    # 
    # cls_Loss = TripletLoss(margin=0.8)
    # 
    cls_Loss = MarginEnLoss(alpha=0.5, r=1)

    MDRL_Train(model=model, cls_Loss=cls_Loss,
               train_loader_D1=train_loader_D1, train_loader_D2=train_loader_D2,
               train_loader_D3=train_loader_D3, test_loader_D1=test_loader_D1,
               test_loader_D2=test_loader_D2, test_loader_D3=test_loader_D3,
               max_length=max_length, num_epochs=200, lambda_param=1.0, BestPath=f'BestResult/Pre_w_Trans_HUST.pth',
               device=None)
