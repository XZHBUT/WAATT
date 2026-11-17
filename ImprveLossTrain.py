import math
import time

import torch
from thop import profile
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from Loss.LearnMarineTripleLoss import ImprovedTripletLoss
from Loss.MarginEnhancementLoss import MarginEnLoss
from Loss.TripletLossBase import TripletLoss
from Model.CNN import CNNModel
from Model.LSTM import LSTM_Classifier
from Model.ResNet import ResNet_18_Ordinary, ResNet_18_TL
from Model.Transformer import Transformer_Classifier
from tool.Build_Task import Build_Task_from_CWRU_All, Build_Task_from_JNU_All, Build_Task_from_SEU_All, \
    Build_Task_from_HUST_SelectWay
from tool.DataProcess import read_OrginCWRU
from tool.SampleBuild import TraditionalBuildTriadGroup, OutputMeanSample
from tool.addnoise import addNoiseBatch, addLaplaceNoiseBatch


# 
def calculate_accuracy(A, A_label, B, C):
    """
    A: (N, 1024) 类别标记向量，N为类别数
    B: (X, 1024) 样本向量，X为样本数
    C: (X) 样本的真实标签

    返回: 准确率 (float)
    """
    # A.unsqueeze(0) -> (1, N, 1024), B.unsqueeze(1) -> (X, 1, 1024)
    # 
    distances = torch.norm(B.unsqueeze(1) - A.unsqueeze(0), dim=2)  # (X, N)

    # 
    predicted_labels_index = torch.argmin(distances, dim=1)  # (X)
    predicted_labels = torch.tensor([A_label[i] for i in predicted_labels_index]).to(C.device)
    # 
    correct_predictions = (predicted_labels == C).sum().item()
    accuracy = correct_predictions / len(C)

    return accuracy


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    # Train_X, Train_Y, _, _, Test_X, Test_Y = read_OrginCWRU(filepath='data/CWRU',
    #                                                         SampleLength=1024,
    #                                                         SampleNum=100,
    #                                                         normal=False,
    #                                                         Rate=[0.8, 0.1, 0.1]
    #                                                         )

    # Train_X, Train_Y, Test_X, Test_Y = Build_Task_from_CWRU_All(
    #     filepath='data/CWRU', Task=1,
    #     SampleLength=1024,  Shot=20, QueryShot=20)

    Train_X, Train_Y, Test_X, Test_Y = Build_Task_from_HUST_SelectWay(filepath='data/HUST', Task=1, Way=11,
        SampleLength=1024,  Shot=100, QueryShot=20)

    # Train_X, Train_Y, Test_X, Test_Y = Build_Task_from_SEU_All(filepath='data/SEU', Task=1,
    #     SampleLength=1024,  Shot=50, QueryShot=100)



    snr = -8
    best_model_filepath = f'BestResult/Test/Trans_hust_{snr}.pth'
    Train_X = addNoiseBatch(Train_X[0], snr)
    # Valid_X = addNoiseBatch(Valid_X, snr)
    Test_X = addNoiseBatch(Test_X[0], snr)

    # Train_X = addLaplaceNoiseBatch(Train_X[0].detach().cpu().numpy(), snr)
    # Test_X = addLaplaceNoiseBatch(Test_X[0].detach().cpu().numpy(), snr)

    # NumPyPyTorch，dtype=torch.int64
    Train_X = torch.tensor(Train_X, dtype=torch.float32)
    Train_Y = torch.tensor(Train_Y[0], dtype=torch.int64)
    Test_X = torch.tensor(Test_X, dtype=torch.float32)
    Test_Y = torch.tensor(Test_Y[0], dtype=torch.int64)

    mean = Train_X.mean()
    std = Train_X.std()
    # Train_X = (Train_X - mean) / std
    # Test_X = (Test_X - mean) / std

    # TensorDataset
    train_dataset = TensorDataset(Train_X, Train_Y)

    test_dataset = TensorDataset(Test_X, Test_Y)

    # DataLoader，batch_size
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet_18_TL(out_f=64, num_classes=11, initial_margin=5.0)
    # model = LSTM_Classifier(out_f=128, num_classes=64, initial_margin=5.0)
    # model = Transformer_Classifier(out_f=128, num_classes=11, initial_margin=5.0)
    # model = CNNModel(out_f=64, num_classes=11, initial_margin=5.0)
    model.feature_extractor.load_state_dict(torch.load("BestResult/Pre_w_ResNet_HUST.pth"))
    # model.feature_extractor.load_state_dict(torch.load("BestResult/Pre_w_Transformer.pth"))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")
    model.to(device)
    num_epochs, batch_size = 200, 64
    # 
    max_lr = 0.0001
    min_lr = 0.00001

    # 
    criterion = MarginEnLoss(alpha=0.2, r=2)

    optimizer = optim.Adam(model.parameters(), lr=max_lr)
    # 
    tensor_list = []

    start_time = time.time()
    Curr_val_accuracy = 0
    for epoch in range(num_epochs):
        print(f'\r Progress: {epoch + 1}/{num_epochs}', end='\n')
        # 
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(epoch / num_epochs * math.pi))
        # 
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # 
        model.train()
        train_loss = 0.0
        # 
        correct_train = 0

        a = 0
        for inputs, labels in train_loader:
            a = a + 1
            inputs = inputs.to(device)  # GPU
            labels = labels.to(device)  # GPU
            optimizer.zero_grad()
            feature_out, outputs, margin_bounds = model(inputs)
            Anchor_Group, Positive_Group, Negative_Group, Anchor_Group_label = TraditionalBuildTriadGroup(outputs,
                                                                                                          labels,
                                                                                                          PK=1, NK=4)
            loss = criterion(Anchor_Group, Positive_Group, Negative_Group, Anchor_Group_label, margin_bounds)
            # loss = criterion(outputs, labels)
            gaussian_prior = torch.randn_like(inputs).to(device)
            wasserstein_loss = torch.mean(torch.abs(inputs - gaussian_prior))
            # Loss = loss + wasserstein_loss
            Loss =loss
            Loss.backward()
            optimizer.step()  # 
            # 
            # 

            train_loss += loss.item()
            Anchor, Anchor_Label = OutputMeanSample(outputs, labels)
            correct_train += (calculate_accuracy(Anchor, Anchor_Label, outputs, labels) * len(inputs))
            print('batch:' + str(a) + '----' +
                  'loss:' + str(loss.item()) + '----' +
                  'correct_train:' + str(
                calculate_accuracy(Anchor, Anchor_Label, outputs, labels)))

        train_loss /= len(train_loader)
        train_accuracy = correct_train / len(train_dataset)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # 
        model.eval()
        val_loss = 0.0
        correct_val = 0

        with torch.no_grad():
            _, output_all_X, margin_bounds = model(Train_X.to(device))
            Anchor_Group, Anchor_Group_label = OutputMeanSample(output_all_X, Train_Y)
            # ，，，。
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                feature_out, outputs, _ = model(inputs)
                _, Positive_Group, Negative_Group, _ = TraditionalBuildTriadGroup(outputs, labels, PK=1, NK=4)
                loss = criterion(Anchor_Group, Positive_Group, Negative_Group, Anchor_Group_label, margin_bounds)
                # loss = criterion(outputs, labels)
                val_loss += loss.item()
                # Anchor, Anchor_Label = OutputMeanSample(outputs, labels)
                correct_val += (calculate_accuracy(Anchor_Group, Anchor_Group_label, outputs, labels) * len(inputs))

        if epoch % 10 == 0:
            tensor_list.append(margin_bounds)
            print(margin_bounds)


        val_loss /= len(test_loader)
        val_accuracy = correct_val / len(test_dataset)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        if val_accuracy >= Curr_val_accuracy:
            Curr_val_accuracy = val_accuracy
            # 、Anchor_GroupAnchor_Group_label
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'Anchor_Group': Anchor_Group,
                'Anchor_Group_label': Anchor_Group_label
            }
            torch.save(checkpoint, best_model_filepath)  # 
            print("保持参数")
        # print(margin_bounds)

        # with open(train_loss_file, 'a') as f_train_loss:
        #     f_train_loss.write(f'{epoch + 1}\t{train_loss}\n')
        #
        # with open(val_loss_file, 'a') as f_val_loss:
        #     f_val_loss.write(f'{epoch + 1}\t{val_loss}\n')
        #


    end_time = time.time()
    # （）
    execution_time = end_time - start_time
    execution_time_minutes = execution_time / 60
    # 
    print(f"执行时间：{execution_time_minutes:.2f} min")

    # 
    with torch.no_grad():
        feature_out, outputs, margin_bounds = model(Test_X.to(device))
    # （）
    _, predicted = torch.max(outputs, 1)

    # 
    correct = (predicted == Test_Y.to(device)).sum().item()
    total = Test_Y.size(0)
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

    #  numpy 
    predicted_np = predicted.cpu().numpy()
    labels_np = Test_Y.cpu().numpy()

    # #  Precision  Recall
    # precision = precision_score(labels_np, predicted_np, average='weighted')
    # recall = recall_score(labels_np, predicted_np, average='weighted')

    #  Precision  Recall
    # print(f'Precision: {precision * 100:.2f}')
    # print(f'Recall: {recall * 100:.2f}')
    # print(f"Total Parameters: {total_params / 1e6:.3f}M")
    # print(f"Total FLOPs: {macs / 10 ** 6 :.3f} MFLOPs")  # FLOPsMFLOPs

    # tensortxt，
    with open("tensor_output.txt", "w") as file:
        for tensor in tensor_list:
            tensor_str = ', '.join(map(str, tensor.detach().cpu().numpy()))  #  detach() 
            file.write(tensor_str + "\n")  # tensor
