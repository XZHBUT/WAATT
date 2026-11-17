import math
import time

import torch
from thop import profile
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from Model.CNN import CNNModel_Ordinary
from Model.LSTM import LSTM_Ordinary
from Model.ResNet import ResNet_18_Ordinary
from Model.Transformer import Transformer_Ordinary
from tool.Build_Task import Build_Task_from_CWRU_All, Build_Task_from_HUST_SelectWay, Build_Task_from_JNU_All, \
    Build_Task_from_SEU_All
from tool.DataProcess import read_OrginCWRU
from tool.addnoise import addNoiseBatch, addLaplaceNoiseBatch


# 
def compute_accuracy(output, target):
    _, predicted = torch.max(output, 1)
    correct = (predicted == target).sum().item()
    accuracy = correct / target.size(0)
    return accuracy

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train_X, Train_Y, Test_X, Test_Y = Build_Task_from_CWRU_All(
    #     filepath='data/CWRU', Task=1,
    #     SampleLength=1024,  Shot=50, QueryShot=100)

    Train_X, Train_Y, Test_X, Test_Y = Build_Task_from_HUST_SelectWay(filepath='data/HUST', Task=1, Way=11,
        SampleLength=1024,  Shot=50, QueryShot=100)

    # Train_X, Train_Y, Test_X, Test_Y = Build_Task_from_SEU_All(filepath='data/SEU', Task=1,
    #     SampleLength=1024,  Shot=50, QueryShot=100)

    snr = -10
    Train_X = addNoiseBatch(Train_X[0], snr)
    # # Valid_X = addNoiseBatch(Valid_X, snr)
    Test_X = addNoiseBatch(Test_X[0], snr)
    best_model_filepath = f'BestResult/Test/ResNet_hust_ooo_{snr}.pth'
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
    # model = LSTM_Ordinary(num_classes=11)
    model =ResNet_18_Ordinary(num_classes=11)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")
    model.to(device)
    num_epochs, batch_size = 200, 64
    # 
    max_lr = 0.001
    min_lr = 0.0003

    # 
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=max_lr)

    start_time = time.time()
    curr_acc = 0
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
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()  # 
            # 
            # 
            train_loss += loss.item()
            correct_train += (compute_accuracy(outputs, labels) * len(inputs))
            print('batch:' + str(a) + '----' +
                  'loss:' + str(loss.item()) + '----' +
                  'correct_train:' + str(compute_accuracy(outputs, labels)))

        train_loss /= len(train_loader)
        train_accuracy = correct_train / len(train_dataset)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        # 
        model.eval()
        val_loss = 0.0
        correct_val = 0

        with torch.no_grad():
            # ，，，。
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                correct_val += (compute_accuracy(outputs, labels) * len(inputs))

        val_loss /= len(test_loader)
        val_accuracy = correct_val / len(test_dataset)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        # with open(train_loss_file, 'a') as f_train_loss:
        #     f_train_loss.write(f'{epoch + 1}\t{train_loss}\n')
        #
        # with open(val_loss_file, 'a') as f_val_loss:
        #     f_val_loss.write(f'{epoch + 1}\t{val_loss}\n')
        #
        if curr_acc < val_accuracy:
            curr_acc = val_accuracy
            torch.save(model.state_dict(),
                   best_model_filepath)  #  'best_model.pth'
            print("保存")

    end_time = time.time()
    # （）
    execution_time = end_time - start_time
    execution_time_minutes = execution_time / 60
    # 
    print(f"执行时间：{execution_time_minutes:.2f} min")

    # 
    with torch.no_grad():
        outputs = model(Test_X.to(device))
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



