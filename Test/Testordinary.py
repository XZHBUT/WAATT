import torch

from Draw.Draw_Tsen import visualize_anchors_with_boundaries_tsne, visualize_tsne
from Model.CNN import CNNModel, CNNModel_Ordinary
from Model.LSTM import LSTM_Classifier, LSTM_Ordinary
from Model.ResNet import ResNet_18_TL, ResNet_18_Ordinary
from Model.Transformer import Transformer_Ordinary
from tool.Build_Task import Build_Task_from_CWRU_All
from tool.addnoise import addNoiseBatch
from torch.utils.data import TensorDataset, DataLoader

def compute_accuracy(output, target):
    _, predicted = torch.max(output, 1)
    correct = (predicted == target).sum().item()
    accuracy = correct / target.size(0)
    return accuracy

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Train_X, Train_Y, Test_X, Test_Y = Build_Task_from_CWRU_All(
        filepath='../data/CWRU', Task=1,
        SampleLength=1024,  Shot=100, QueryShot=100)

    snr = 8
    Train_X = addNoiseBatch(Train_X[0], snr)
    # Valid_X = addNoiseBatch(Valid_X, snr)
    Test_X = addNoiseBatch(Test_X[0], snr)

    # NumPyPyTorch，dtype=torch.int64
    Train_X = torch.tensor(Train_X, dtype=torch.float32)
    Train_Y = torch.tensor(Train_Y[0], dtype=torch.int64)
    Test_X = torch.tensor(Test_X, dtype=torch.float32)
    Test_Y = torch.tensor(Test_Y[0], dtype=torch.int64)

    # TensorDataset
    train_dataset = TensorDataset(Train_X, Train_Y)

    test_dataset = TensorDataset(Test_X, Test_Y)

    # DataLoader，batch_size
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    model = Transformer_Ordinary(num_classes=10)
    # model = LSTM_Classifier(out_f=64, num_classes=10, initial_margin=4.0)
    model.to(device)
    # 
    model.load_state_dict(torch.load(f"../BestResult/Test/Trans_ooo_8.pth"))
    # Anchor_GroupAnchor_Group_label
    model.eval()

    #  Test_X  Test_Y ，Test_Y （）
    dataset = TensorDataset(Test_X)  # ，
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 
    outputs = []

    # ， no_grad 
    with torch.no_grad():
        for batch in data_loader:
            batch_input = batch[0].to(device)
            batch_output = model(batch_input)
            outputs.append(batch_output)

    # 


    output = torch.cat(outputs, dim=0)


    # output = model(Train_X.to(device))
    acc = compute_accuracy(output, Test_Y.to(device))
    print(acc)
    visualize_tsne(output, Test_Y.to(device), select_shot=None)



