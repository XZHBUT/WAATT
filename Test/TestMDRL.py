import torch

from Draw.Confusion import plot_confusion_matrix_HUST
from Draw.Draw_Tsen import visualize_anchors_with_boundaries_tsne, visualize_anchors_hust_tsne
from Model.CNN import CNNModel
from Model.LSTM import LSTM_Classifier
from Model.ResNet import ResNet_18_TL
from Model.Transformer import Transformer_Classifier, plot_confusion_matrix
from tool.Build_Task import Build_Task_from_CWRU_All, Build_Task_from_HUST_SelectWay
from tool.addnoise import addNoiseBatch
from torch.utils.data import TensorDataset, DataLoader

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

    # Train_X, Train_Y, Test_X, Test_Y = Build_Task_from_CWRU_All(
    #     filepath='../data/CWRU', Task=1,
    #     SampleLength=1024,  Shot=100, QueryShot=100)

    Train_X, Train_Y, Test_X, Test_Y = Build_Task_from_HUST_SelectWay(filepath='../data/HUST', Task=1, Way=11,
        SampleLength=1024,  Shot=20, QueryShot=100)

    snr = 8
    Train_X = addNoiseBatch(Train_X[0], snr)
    # Valid_X = addNoiseBatch(Valid_X, snr)
    Test_X = addNoiseBatch(Test_X[0], snr)

    # NumPyPyTorch，dtype=torch.int64
    Train_X = torch.tensor(Train_X, dtype=torch.float32)
    Train_Y = torch.tensor(Train_Y[0], dtype=torch.int64)
    Test_X = torch.tensor(Test_X, dtype=torch.float32)
    Test_Y = torch.tensor(Test_Y[0], dtype=torch.int64)
    # Test_X = Train_X
    # Test_Y = Train_Y
    # TensorDataset
    train_dataset = TensorDataset(Train_X, Train_Y)

    test_dataset = TensorDataset(Test_X, Test_Y)

    # DataLoader，batch_size
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # model = CNNModel(out_f=64, num_classes=11, initial_margin=5.0)
    # model = LSTM_Classifier(out_f=128, num_classes=64, initial_margin=5.0)
    model = Transformer_Classifier(out_f=128, num_classes=11, initial_margin=5.0)
    model.to(device)
    checkpoint = torch.load(f"../BestResult/Test/Trans_hust_8.pth")
    # 
    model.load_state_dict(checkpoint['model_state_dict'])
    # Anchor_GroupAnchor_Group_label
    Anchor_Group = checkpoint['Anchor_Group']
    Anchor_Group_label = checkpoint['Anchor_Group_label']
    model.eval()
    output = model(Test_X.to(device))

    #  Test_X  Test_Y ，Test_Y （）
    dataset = TensorDataset(Test_X)  # ，
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False)

    # 
    outputs = []

    # ， no_grad 
    with torch.no_grad():
        for batch in data_loader:
            batch_input = batch[0].to(device)
            batch_output = model(batch_input)
            outputs.append(batch_output[1])
            # batch_input.detach().cpu().numpy()

    # 
    output = torch.cat(outputs, dim=0)

    # acc = calculate_accuracy(Anchor_Group.to(device), Anchor_Group_label, output, Test_Y.to(device))
    # print(acc)

    # if acc >= 0.0:
    distances = torch.norm(output.unsqueeze(1) - Anchor_Group.unsqueeze(0), dim=2)  # (X, N)
    predicted_labels_index = torch.argmin(distances, dim=1)  # (X)
    query_prediction = torch.tensor([Anchor_Group_label[i] for i in predicted_labels_index]).to(Test_Y.device)
    plot_confusion_matrix_HUST(y_true=Test_Y, y_pred=query_prediction)
    visualize_anchors_hust_tsne(Anchor_Group.to(device), None, output, Anchor_Group_label, Test_Y.to(device))


