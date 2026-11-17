import numpy as np
import os

import torch

from tool.DataProcess import read_OrginCWRU

from tool.addnoise import addNoise, addMixedNoise, addNoiseBatch




# np.random.seed(42)  # 

def addNoiseBatch_Random(x, y, min_snr, max_snr, N, ramdom_snr=None, isMix=False):
    noisy_samples = []
    labels = np.full(len(x) * N, y[0])
    if ramdom_snr == None:
        span_snr = N * 10
    for i in range(len(x)):
        list_Random_snr = np.random.choice(np.linspace(min_snr, max_snr, num=span_snr), N, replace=False)
        # print(list_Random_snr)
        for j in range(N):
            # Randomly select an SNR value from the specified range
            snr = list_Random_snr[j]
            # Generate noisy sample
            if isMix==True:
                noisy_sample = addMixedNoise(x[i], snr)
            else:
                noisy_sample = addNoise(x[i], snr)
            noisy_samples.append(noisy_sample)

            # Store the SNR value as the label
            # labels.append(snr)

    # Stack the noisy samples
    noisy_samples = np.stack(noisy_samples, axis=0)

    return noisy_samples, labels



def BuildNoiseSample(X_orig_Sample, labels_orig_Sample, min_snr, max_snr, n_Noise, num_class, isMix=False):
    # X_orig_Sample (400, 1024)
    # X_noisy_samples (4000, 1024)
    X_NoiseSample = []
    Y_NoiseSample = []
    for i_class in range(0, num_class):
        i_class_index = (labels_orig_Sample == i_class)
        i_X_sample = X_orig_Sample[i_class_index]
        i_Y_sample = labels_orig_Sample[i_class_index]
        # print(i_Y_sample.shape)
        # print(i_X_sample.shape)
        i_noisy_samples, i_labels = addNoiseBatch_Random(x=i_X_sample, y=i_Y_sample, min_snr=min_snr, max_snr=max_snr,
                                                         N=n_Noise, isMix=isMix)
        # print(i_noisy_samples.shape)
        # print(i_labels.shape)
        # print(i_labels)
        X_NoiseSample.append(i_noisy_samples)
        Y_NoiseSample.append(i_labels)

    # Stack the noisy samples
    X_noisy_samples = np.concatenate(X_NoiseSample, axis=0)
    Y_noisy_samples = np.concatenate(Y_NoiseSample, axis=0)
    # print(X_noisy_samples.shape)
    # print(Y_noisy_samples.shape)
    # print(Y_noisy_samples)
    return X_noisy_samples, Y_noisy_samples


def TraditionalBuildNoiseSample(X_orig_Sample, labels_orig_Sample,Snr, num_class):
    # X_orig_Sample (400, 1024)
    X_NoiseSample = []
    Y_NoiseSample = []
    for i_class in range(0, num_class):
        i_class_index = (labels_orig_Sample == i_class)
        i_X_sample = X_orig_Sample[i_class_index]
        i_Y_sample = labels_orig_Sample[i_class_index]
        i_noisy_samples = addNoiseBatch(i_X_sample, Snr)
        i_labels = i_Y_sample
        X_NoiseSample.append(i_noisy_samples)
        Y_NoiseSample.append(i_labels)
    # Stack the noisy samples
    X_noisy_samples = np.concatenate(X_NoiseSample, axis=0)
    Y_noisy_samples = np.concatenate(Y_NoiseSample, axis=0)
    # print(X_noisy_samples.shape)
    # print(Y_noisy_samples.shape)
    # print(Y_noisy_samples)
    return X_noisy_samples, Y_noisy_samples

def OutputMeanSample(Orig_Sample, Orig_labels):
    # Orig_Sample (80, 1024)
    # Orig_labels (80)
    # 
    B, L = Orig_Sample.shape
    #  torch.unique 
    unique_labels = torch.unique(Orig_labels)
    # 
    num_classes = unique_labels.size(0)
    #  unique_labels  Python 
    unique_labels_list = unique_labels.tolist()
    # 
    class_means = torch.zeros((num_classes, L)).to(Orig_Sample.device)

    for index, i_class in enumerate(unique_labels_list):
        # ，
        class_indices = (Orig_labels == i_class)
        # 
        class_samples = Orig_Sample[class_indices]  # (8, 1024)
        # 
        if class_samples.size(0) > 0:
            class_means[index] = class_samples.mean(dim=0)
    # 
    class_labels = unique_labels_list
    # print(class_means.shape) torch.Size([10, 1024])
    return class_means, class_labels

def BuildTriadGroup(Orig_Sample, Orig_labels, Noise_Sample, Noise_labels, num_classes, PK=None, NK=None, IndexP=4, InedxN=32):
    # Orig_Sample (80, 1024)
    # Orig_labels (80)
    # Noise_Sample (800, 1024)
    # Noise_labels (800)
    # PK = NK = 10

    # 
    B, L = Orig_Sample.shape
    if PK == None:
        PK = 10
    if NK == None:
        NK = PK
    # print('PK',PK)
    # print('NK', NK)
    #  torch.unique 
    unique_labels = torch.unique(Orig_labels)
    # 
    num_classes = unique_labels.size(0)
    #  unique_labels  Python 
    unique_labels_list = unique_labels.tolist()
    # 
    class_means = torch.zeros((num_classes, L)).to(Noise_Sample.device)
    # ,
    positive_group = []
    negative_group = []
    for index, i_class in enumerate(unique_labels_list):
        # ，
        class_indices = (Orig_labels == i_class)
        # 
        class_samples = Orig_Sample[class_indices]  # (8, 1024)

        # 
        A_Psample_dis = torch.norm(class_samples - class_samples.mean(dim=0), dim=1)
        # 
        half_num_samples_PS = class_samples.size(0) // IndexP
        APSample_nearest_indices = torch.topk(A_Psample_dis, half_num_samples_PS, largest=True).indices
        positive_group.append(class_samples[APSample_nearest_indices])

        # positive_group.append(class_samples)

        # 
        if class_samples.size(0) > 0:
            class_means[index] = class_samples.mean(dim=0)

        # 
        other_class_indices = (Orig_labels != i_class)
        # 
        other_class_samples = Orig_Sample[other_class_indices]  # (8, 1024)
        # # 
        A_othersamples_dis = torch.norm(other_class_samples - class_samples.mean(dim=0), dim=1)
        # 
        half_num_samples = other_class_samples.size(0) // InedxN
        Aothersamples_nearest_indices = torch.topk(A_othersamples_dis, half_num_samples, largest=False).indices
        negative_group.append(other_class_samples[Aothersamples_nearest_indices])

        # negative_group.append(other_class_samples)

    # 
    class_labels = torch.arange(num_classes)

    # print(class_means.shape) torch.Size([10, 1024])

    Anchor_Group = class_means
    Anchor_Group_label = unique_labels_list
    # negative_group = []
    for index, i_class in enumerate(unique_labels_list):
        i_anchor = Anchor_Group[index]

        # 
        class_Noise_indices = (Noise_labels == i_class)
        # 
        class_Noise_samples = Noise_Sample[class_Noise_indices]
        # print(class_Noise_samples.shape) (80, 1024) 80=8x10
        # 
        A_Pnosie_dis = torch.norm(class_Noise_samples - i_anchor, dim=1)
        #  K 

        # print(A_Pnosie_dis.size(0))
        if A_Pnosie_dis.size(0) < PK:
            AP_nearest_indices = torch.topk(A_Pnosie_dis, A_Pnosie_dis.size(0), largest=True).indices
        else:
            AP_nearest_indices = torch.topk(A_Pnosie_dis, PK, largest=True).indices
        #  K 
        Pnosie_k = class_Noise_samples[AP_nearest_indices]
        # print(Pnosie_k.shape) torch.Size([K, 1024])
        # 
        positive_group[index] = torch.cat((positive_group[index], Pnosie_k), dim=0)
        # positive_group.append(Pnosie_k)

        # 
        other_sample_mean = torch.cat((Anchor_Group[:index], Anchor_Group[index + 1:]), dim=0)
        # # print(other_sample_mean_list) (9, 1024)
        negative_group[index] = torch.cat((negative_group[index], other_sample_mean), dim=0)
        class_other_Noise_indices = (Noise_labels != i_class)
        # 
        class_other_Noise_samples = Noise_Sample[class_other_Noise_indices]
        # print(class_other_Noise_samples.shape) torch.Size([720, 1024]) 720=800-80
        # 
        A_Nnosie_dis = torch.norm(class_other_Noise_samples - i_anchor, dim=1)
        #  K 
        if A_Pnosie_dis.size(0) < NK:
            AN_nearest_indices = torch.topk(A_Nnosie_dis, A_Pnosie_dis.size(0), largest=False).indices
        else:
            AN_nearest_indices = torch.topk(A_Nnosie_dis, NK, largest=False).indices
        #  K 
        Nnosie_k = class_other_Noise_samples[AN_nearest_indices]
        # print(Nnosie_k.shape) torch.Size([NK, 1024])
        negative_group[index] = torch.cat((negative_group[index], Nnosie_k), dim=0)

    Positive_Group = positive_group # ，index
    Negative_Group = negative_group

    return Anchor_Group, Positive_Group, Negative_Group, Anchor_Group_label

def TraditionalBuildTriadGroup(Orig_Sample, Orig_labels, PK=None, NK=None):
    # 
    # 
    B, L = Orig_Sample.shape
    if PK == None:
        PK = 2
    if NK == None:
        NK = PK
    #  torch.unique 
    unique_labels = torch.unique(Orig_labels)
    # 
    num_classes = unique_labels.size(0)
    #  unique_labels  Python 
    unique_labels_list = unique_labels.tolist()
    # 
    class_means = torch.zeros((num_classes, L)).to(Orig_Sample.device)
    # ,
    positive_group = []

    for index, i_class in enumerate(unique_labels_list):
        # ，
        class_indices = (Orig_labels == i_class)
        # 
        class_samples = Orig_Sample[class_indices]  # (8, 1024)
        # positive_group.append(class_samples)
        # 
        if class_samples.size(0) > 0:
            class_means[index] = class_samples.mean(dim=0)
    # 
    class_labels = torch.arange(num_classes)
    Anchor_Group = class_means
    Anchor_Group_label = unique_labels_list
    negative_group = []
    for index, i_class in enumerate(unique_labels_list):
        i_anchor = Anchor_Group[index]
        # 
        class_Noise_indices = (Orig_labels == i_class)
        # 
        class_samples = Orig_Sample[class_Noise_indices]
        # print(class_Noise_samples.shape) (80, 1024) 80=8x10
        # 
        A_Pnosie_dis = torch.norm(class_samples - i_anchor, dim=1)
        #  K 
        if A_Pnosie_dis.size(0) < PK:
            AP_nearest_indices = torch.topk(A_Pnosie_dis, A_Pnosie_dis.size(0), largest=True).indices
        else:
            AP_nearest_indices = torch.topk(A_Pnosie_dis, PK, largest=True).indices
        #  K 
        Pnosie_k = class_samples[AP_nearest_indices]
        # print(Pnosie_k.shape) torch.Size([K, 1024])
        # 
        # positive_group[index] = torch.cat((positive_group[index], Pnosie_k), dim=0)
        positive_group.append(Pnosie_k)


        # 
        # other_sample_mean = torch.cat((Anchor_Group[:index], Anchor_Group[index + 1:]), dim=0)
        # print(other_sample_mean_list) (9, 1024)
        # negative_group.append(other_sample_mean)
        class_other_Noise_indices = (Orig_labels != i_class)
        # 
        class_other_samples = Orig_Sample[class_other_Noise_indices]
        # print(class_other_Noise_samples.shape) torch.Size([720, 1024]) 720=800-80
        # 
        A_Nnosie_dis = torch.norm(class_other_samples - i_anchor, dim=1)
        #  K 
        if A_Nnosie_dis.size(0) < NK:
            AN_nearest_indices = torch.topk(A_Nnosie_dis, A_Nnosie_dis.size(0), largest=False).indices
        else:
            AN_nearest_indices = torch.topk(A_Nnosie_dis, NK, largest=False).indices
        #  K 
        Nnosie_k = class_other_samples[AN_nearest_indices]
        # print(Nnosie_k.shape) torch.Size([NK, 1024])
        # negative_group[index] = torch.cat((negative_group[index], Nnosie_k), dim=0)
        negative_group.append(Nnosie_k)

    Positive_Group = positive_group # ，index
    Negative_Group = negative_group

    return Anchor_Group, Positive_Group, Negative_Group, Anchor_Group_label




if __name__ == '__main__':
    Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y = read_OrginCWRU(filepath='../data/CWRU',
                                                                        SampleLength=1024,
                                                                        SampleNum=40,
                                                                        normal=False,
                                                                        Rate=[0.5, 0.3, 0.2]
                                                                        )

    # i_class_index = (Train_Y == 1)
    # i_X_sample = Train_X[i_class_index]
    # i_Y_sample = Train_Y[i_class_index]
    # i_noisy_samples, i_labels = addNoiseBatch_Random(x=i_X_sample, y=i_Y_sample,min_snr=-10, max_snr=10, N=10)
    #
    # i_noisy_samples = torch.tensor(np.concatenate((i_noisy_samples, i_X_sample), axis=0), dtype=torch.float32)
    # i_labels = torch.tensor(np.concatenate((i_labels, np.full(i_Y_sample.shape, 0)),axis=0), dtype=torch.float32)
    # start_tsne(i_noisy_samples, i_labels,f'_bef.png')

    # addNoiseBatch_Random(Train_X, Train_Y, min_snr=-4, max_snr=4, N=10)
    # X_noisy_samples, Y_noisy_samples = BuildNoiseSample(Train_X, Train_Y, min_snr=-4, max_snr=4, n_Noise=10, num_class=10)

    # i_noisy_samples = torch.tensor(np.concatenate((X_noisy_samples, Train_X), axis=0), dtype=torch.float32)
    # i_labels = torch.tensor(np.concatenate((Y_noisy_samples, Train_Y), axis=0), dtype=torch.float32)
    # start_tsne(i_noisy_samples, i_labels,f'_bef.png')

    X_noisy_samples, Y_noisy_samples = BuildNoiseSample(Test_X, Test_Y, min_snr=-4, max_snr=4, n_Noise=10,
                                                        num_class=10)
    Train_X = torch.tensor(Train_X, dtype=torch.float32)
    Train_Y = torch.tensor(Train_Y, dtype=torch.int64)
    Valid_X = torch.tensor(Valid_X, dtype=torch.float32)
    Valid_Y = torch.tensor(Valid_Y, dtype=torch.int64)
    Test_X = torch.tensor(Test_X, dtype=torch.float32)
    Test_Y = torch.tensor(Test_Y, dtype=torch.int64)

    print(Train_X.shape)
    print(Train_Y.shape)

    X_noisy_samples = torch.tensor(X_noisy_samples, dtype=torch.float32)
    Y_noisy_samples = torch.tensor(Y_noisy_samples, dtype=torch.int64)

    # K
    indices = torch.randperm(Test_X.size(0))[:20]  # K

    # 
    X_random = Test_X[indices]
    Y_random = Test_Y[indices]

    BuildTriadGroup(X_random, Y_random, X_noisy_samples, Y_noisy_samples, num_classes=10, NK=100)
    a,b,c,d = TraditionalBuildTriadGroup(Train_X,Train_Y)
    print(a.shape)

    # num_class = 10
    #
    # for i_class in range(0, num_class):
    #     i_class_index = (Train_Y == i_class)
    #     i_sample = Train_X[i_class_index]
    #
    #     i_other_class_index = Train_Y != i_class
    #     i_other_sample = Train_X[Train_Y != i_other_class_index]
    #     print(i_sample.shape)
    #     print(i_other_sample.shape)
