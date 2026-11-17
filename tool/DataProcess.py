import pywt  # pywt
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import torch
import xlrd
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg

from scipy.io import loadmat
# scikit-learnpreprocessing，，
from sklearn import preprocessing
# 
import pandas as pd
# scikit-learnStratifiedShuffleSplit，
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset



def read_OneOH(filepath, ClassNameList):
    '''
    # 获取一种负载文件下的的数据

    :return: dict{} key：类名， Velue：该的全部电波
    '''

    # .mat,，filenames
    filenames = os.listdir(filepath)

    # MATLAB。filenames，
    # ，， 'DE' 。
    def capture(original_path, ClassNameList):
        files = {}
        for i in filenames:

            ClassName = ''
            for one in ClassNameList:
                if one in i:
                    ClassName = one
                    break
            # 
            file_path = os.path.join(filepath, i)
            file = loadmat(file_path)
            file_keys = file.keys()
            for key in file_keys:
                if 'DE' in key:
                    files[ClassName] = file[key].ravel()
        return files

    return capture(filepath, ClassNameList)


def SliceDate(data, slice_rate, SampleNumpy, SampleLength):
    '''
    :param data: dict (key:类别名， value： 完整的数据)
    :param slice_rate: 训练集：验证集：测试集
    :param SampleNumpy: 一个类别所拥有的 训练集+验证集+测试集的数量 ps：例如总的样本是 N 训练集0.7 一个负载下的类是：（N/负载数/类别数） * 0.7
    :return: 然后将这些数据存储在Train_Samples和Test_Samples字典中，其中键是文件名，值是切分后的数据
    '''
    keys = data.keys()
    step = int(SampleLength / 2)
    Train_Samples = {}
    Vaild_Samples = {}
    Test_Samples = {}
    # 
    for i in keys:

        slice_data = data[i]  # （）， slice_data 
        samp_train = int(SampleNumpy * slice_rate[0])  # 
        samp_vaild = int(SampleNumpy * slice_rate[1])  # 
        samp_test = int(SampleNumpy * slice_rate[2])  # 
        Train_sample = []
        Vaild_sample = []
        Test_sample = []  #  Train_sample  Test_sample ，。
        # ，，samp_train
        for j in range(samp_train):
            sample = slice_data[j * step: j * step + SampleLength]
            Train_sample.append(sample)

        # ，
        for h in range(samp_vaild):
            sample = slice_data[
                     samp_train * step + SampleLength + h * step: samp_train * step + SampleLength + h * step + SampleLength]
            Vaild_sample.append(sample)

        local = samp_train + samp_vaild
        # 
        for h in range(samp_test):
            sample = slice_data[
                     local * step + SampleLength + h * step: local * step + SampleLength + h * step + SampleLength]
            Test_sample.append(sample)

        Train_Samples[i] = Train_sample
        Vaild_Samples[i] = Vaild_sample
        Test_Samples[i] = Test_sample  #  Train_Samples  Test_Samples ， i 

    return Train_Samples, Vaild_Samples, Test_Samples


# ，，
# 0，1，2，3.。。。，label
# ，，
def add_labels(data_Dict, ClassNameList):
    '''

    :param data_Dict: 一个负载下的字典
    :param ClassNameList:  类名
    :return:  两个列表
    '''
    X = []
    Y = []
    label = 0
    for i in ClassNameList:
        x = data_Dict[i]
        X += x
        lenx = len(x)
        Y += [label] * lenx
        label += 1
    return X, Y

    #  0，1
    #  Train_X, Test_X


def scalar_stand(Train_X, Vaild_X, Test_X):
    data_all = np.vstack((Train_X, Vaild_X, Test_X))  # ， data_all
    Standard = preprocessing.StandardScaler()
    Train_X = Standard.fit_transform(Train_X)
    Vaild_X = Standard.transform(Vaild_X)
    Test_X = Standard.transform(Test_X)
    '''
    使用训练集的均值和标准差来对训练集和测试集的数据进行标准化的原因如下：

    避免信息泄漏（Data Leakage）：在机器学习中，我们通常将数据分为训练集和测试集，以评估模型的性能。如果在标准化时使用了整个数据集（包括测试集）的均值和标准差，那么在某种程度上会将测试集的信息引入到训练过程中，这可能导致过于乐观的性能评估。因此，应该仅使用训练集的统计信息，以避免信息泄漏。

    模拟真实世界：在实际应用中，我们通常只能在训练阶段获取有限数量的数据用于模型训练，而测试数据是未来模型将要处理的新数据。因此，标准化过程应该模拟实际情况，即模型在训练期间只能访问训练数据的统计信息，然后将训练数据的分布应用于测试数据。

    保持一致性：使用训练集的统计信息对训练集和测试集进行标准化可以确保两者之间的一致性。这意味着模型在训练和测试期间看到的数据分布是相同的，从而使得模型更容易泛化到未见过的数据。
    '''
    return Train_X, Vaild_X, Test_X


def read_OrginCWRU(filepath, SampleNum=200, SampleLength=1024, normal=False, Rate=[0.7, 0.2, 0.1]):
    ClassName = ['B007', 'B014', 'B021', 'IR007', 'IR014', 'IR021', 'OR007', 'OR014', 'OR021', 'normal']

    # 
    folder_names = [f for f in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, f))]

    Train_X, Train_Y = [], []
    Valid_X, Valid_Y = [], []
    Test_X, Test_Y = [], []
    # 
    for OneOH_File in folder_names:
        OneOH_Dict = read_OneOH(filepath + '/' + OneOH_File, ClassName)
        # OneOH_Dict: ， key：， value： （）

        Train_Samples_OneOH, Vaild_Samples_OneOH, Test_Samples_OneOH = SliceDate(OneOH_Dict, Rate, int(SampleNum / 4),
                                                                                 SampleLength)
        # Train_Samples_OneOH：  key：， value： ，num x 1024
        Train_X_OneOH, Train_Y_OneOH = add_labels(Train_Samples_OneOH, ClassName)
        Valid_X_OneOH, Valid_Y_OneOH = add_labels(Vaild_Samples_OneOH, ClassName)
        Test_X_OneOH, Test_Y_OneOH = add_labels(Test_Samples_OneOH, ClassName)

        Train_X += Train_X_OneOH
        Train_Y += Train_Y_OneOH
        Valid_X += Valid_X_OneOH
        Valid_Y += Valid_Y_OneOH
        Test_X += Test_X_OneOH
        Test_Y += Test_Y_OneOH

    # / .
    if normal:
        Train_X, Valid_X, Test_X = scalar_stand(Train_X, Valid_X, Test_X)

    Train_X = np.asarray(Train_X)
    Valid_X = np.asarray(Valid_X)
    Test_X = np.asarray(Test_X)

    Test_Y = np.asarray(Test_Y, dtype=np.int32)
    Valid_Y = np.asarray(Valid_Y, dtype=np.int32)
    Train_Y = np.asarray(Train_Y, dtype=np.int32)

    return Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y

def SliceDate_Shot(data, Train_shot=3,Test_shot=10, SampleLength=1024):

    keys = data.keys()
    step = int(SampleLength / 2)
    Train_Samples = {}
    Test_Samples = {}
    # 
    for i in keys:
        slice_data = data[i]  # （）， slice_data 
        samp_train = Train_shot  # 
        samp_test = Test_shot  # 
        Train_sample = []
        Test_sample = []  #  Train_sample  Test_sample ，。
        # ，，samp_train
        for j in range(samp_train):
            sample = slice_data[j * step: j * step + SampleLength]
            Train_sample.append(sample)

        local = samp_train
        # 
        for h in range(samp_test):
            sample = slice_data[
                     local * step + SampleLength + h * step: local * step + SampleLength + h * step + SampleLength]
            Test_sample.append(sample)

        Train_Samples[i] = Train_sample
        Test_Samples[i] = Test_sample  #  Train_Samples  Test_Samples ， i 

    return Train_Samples, Test_Samples


def read_OrginCWRU_fromOneOH_fewshot(filepath, Train_shot=3,Test_shot=10, SampleLength=1024, normal=False, Select_OH='0HP', Select_shot=None):
    ClassName = ['B007', 'B014', 'B021', 'IR007', 'IR014', 'IR021', 'OR007', 'OR014', 'OR021', 'normal']
    if Select_shot == None:
        Select_shot = [0, 1, 2,3,4,5,6,7,8,9]
        Select_ClassName_shot = [ClassName[i] for i in Select_shot]
    else:
        Select_ClassName_shot = [ClassName[i] for i in Select_shot]

    # 
    folder_names = [f for f in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, f))]
    OneOH_Dict = read_OneOH(filepath + '/' + Select_OH, ClassName)

    Train_Samples_OneOH,  Test_Samples_OneOH = SliceDate_Shot(OneOH_Dict, Train_shot=Train_shot,Test_shot=Test_shot,SampleLength=SampleLength)
    Select_Train_Samples_OneOH = {key: Train_Samples_OneOH[key] for key in Select_ClassName_shot}
    Select_Test_Samples_OneOH = {key: Test_Samples_OneOH[key] for key in Select_ClassName_shot}

    Train_X_OneOH, Train_Y_OneOH = add_labels(Select_Train_Samples_OneOH, Select_ClassName_shot)
    Test_X_OneOH, Test_Y_OneOH = add_labels(Select_Test_Samples_OneOH, Select_ClassName_shot)


    # / .

    Train_X = np.asarray(Train_X_OneOH)
    Test_X = np.asarray(Test_X_OneOH)

    Test_Y = np.asarray(Test_Y_OneOH, dtype=np.int32)
    Train_Y = np.asarray(Train_Y_OneOH, dtype=np.int32)
    return Train_X, Train_Y,  Test_X, Test_Y




##################################### 

def extract_data_from_JNU_folder(folder_path):
    '''
    返回是一个字典，key是文件名不带后缀，value是的从第17行第一列数据的数组
    '''
    # 
    result_dict = {}

    # 
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            # 
            file_path = os.path.join(folder_path, filename)

            #  CSV ，，
            data = pd.read_csv(file_path, skiprows=16, usecols=[0])

            # 
            file_name_without_extension = os.path.splitext(filename)[0]

            # 
            result_dict[file_name_without_extension] = data.values.flatten()

    return result_dict


def extract_data(data_dict, Classname, SampleNumpy, SampleLength):
    result_dict = {}

    for classname in Classname:
        combined_data = []

        for filename, data in data_dict.items():
            if classname in filename:
                # 
                step = int(SampleLength / 2)
                sample_onefile = []
                for j in range(SampleNumpy):
                    sample = data[j * step: j * step + SampleLength]
                    sample_onefile.append(sample)

                combined_data.extend(sample_onefile)

        result_dict[classname] = combined_data

    return result_dict


def split_data_JNU(data_dict, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1):
    # 
    categories = list(data_dict.keys())

    # 
    Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y = [], [], [], [], [], []

    # 
    label_mapping = {category: i for i, category in enumerate(categories)}

    for category, label in label_mapping.items():
        samples = data_dict[category]
        num_samples = len(samples)
        num_train = int(num_samples * train_ratio)
        num_valid = int(num_samples * valid_ratio)
        num_test = int(num_samples * test_ratio)

        # 
        train_samples = samples[:num_train]
        valid_samples = samples[num_train:num_train + num_valid]
        test_samples = samples[num_train + num_valid:num_train + num_valid + num_test]

        # 
        Train_X.extend(train_samples)
        Train_Y.extend([label] * len(train_samples))
        Valid_X.extend(valid_samples)
        Valid_Y.extend([label] * len(valid_samples))
        Test_X.extend(test_samples)
        Test_Y.extend([label] * len(test_samples))

    return Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y


def Read_Data_From_JNU(filepath, SampleNum=500, SampleLength=1024, normal=False, Rate=None):
    # csv， key： value：
    if Rate is None:
        Rate = [0.7, 0.2, 0.1]
    data_dict_everyfile = extract_data_from_JNU_folder(filepath)

    #    
    classname = ['ib', 'n', 'ob', 'tb']
    data_dict_everyclass = extract_data(data_dict_everyfile, classname, SampleNum // 3, SampleLength)

    # 
    Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y = split_data_JNU(data_dict_everyclass, Rate[0], Rate[1], Rate[2])

    # / .
    if normal:
        Train_X, Valid_X, Test_X = scalar_stand(Train_X, Valid_X, Test_X)

    Train_X = np.asarray(Train_X)
    Valid_X = np.asarray(Valid_X)
    Test_X = np.asarray(Test_X)

    Test_Y = np.asarray(Test_Y, dtype=np.int32)
    Valid_Y = np.asarray(Valid_Y, dtype=np.int32)
    Train_Y = np.asarray(Train_Y, dtype=np.int32)

    return Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y


def read_OrginJNU(filepath, SampleNum=500, SampleLength=1024, normal=True, Rate=None):
    # csv， key： value：
    if Rate is None:
        Rate = [0.7, 0.2, 0.1]
    data_dict_everyfile = extract_data_from_JNU_folder(filepath)

    #    
    classname = ['ib', 'n', 'ob', 'tb']
    data_dict_everyclass = extract_data(data_dict_everyfile, classname, SampleNum, SampleLength)

    # 
    Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y = split_data_JNU(data_dict_everyclass, Rate[0], Rate[1], Rate[2])

    # / .
    if normal:
        Train_X, Valid_X, Test_X = scalar_stand(Train_X, Valid_X, Test_X)

    Train_X = np.asarray(Train_X)
    Valid_X = np.asarray(Valid_X)
    Test_X = np.asarray(Test_X)

    Test_Y = np.asarray(Test_Y, dtype=np.int32)
    Valid_Y = np.asarray(Valid_Y, dtype=np.int32)
    Train_Y = np.asarray(Train_Y, dtype=np.int32)

    return Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y



######################################## 

def min_max_normalize(data):
    """
    对每个通道进行最大最小归一化。

    参数:
    - data: 一个二维NumPy数组，形状为(L, 8)，代表8个通道的序列。

    返回:
    - 归一化后的数据，形状同样为(L, 8)。
    """
    # 

    data = data.transpose()
    normalized_data = np.zeros(data.shape)

    # 
    for i in range(data.shape[0]):  # 
        channel_min = data[i, :].min()  # 
        channel_max = data[i, :].max()  # 

        # 
        normalized_data[i, :] = (data[i, :] - channel_min) / (channel_max - channel_min)

    normalized_data = normalized_data.transpose()

    return normalized_data

import numpy as np

def standardize(data):
    """
    对每个通道进行标准化（Z-score 标准化）。

    参数:
    - data: 一个二维NumPy数组，形状为(L, 8)，代表8个通道的序列。

    返回:
    - 标准化后的数据，形状同样为(L, 8)。
    """
    # 
    data = data.transpose()
    standardized_data = np.zeros(data.shape)

    # 
    for i in range(data.shape[0]):  # 
        channel_mean = data[i, :].mean()  # 
        channel_std = data[i, :].std()    # 

        # 
        standardized_data[i, :] = (data[i, :] - channel_mean) / channel_std

    # 
    standardized_data = standardized_data.transpose()

    return standardized_data


def Read_csv_Data(file_path, row_num=102400, lienum=8, isTime=True):
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        data_started = False
        data_rows = []
        row_count = 0

        for row in reader:
            # 
            if row and row[0].strip().lower() == 'data':
                data_started = True
                continue

            # 
            if data_started and row:
                # ，\t
                if len(row) == 1 and '\t' in row[0]:
                    # ，\t
                    row = row[0].split('\t')

                # 8，0.0
                if isTime:
                    processed_row = [float(value) if value else 0.0 for value in row[:lienum]]
                else:
                    processed_row = [float(value) if value else 0.0 for value in row[1:lienum]]
                data_rows.append(processed_row)
                row_count += 1  # 

            # 10,000，
            if row_count == row_num:
                break

        # # [8, L]
        # data_rows_transposed = list(zip(*data_rows))
        return data_rows


def Biuld_Sample(Data_list, SampleNumpy, SampleLength, step=None):
    Sample_List = []
    if step == None:
        step = int(SampleLength / 2)

    for j in range(SampleNumpy):
        sample = Data_list[j * step: j * step + SampleLength]
        Sample_List.append(list(map(list, zip(*sample))))

    # print(Sample_List[0][-1])
    return Sample_List


def Read_Data_From_SEU(filepath, SampleNum=200, SampleLength=1024, Rate=[0.7, 0.2, 0.1], stander=False):
    # 4
    print("Creating Data Sets.....SEU")
    if Rate is None:
        Rate = [0.7, 0.2, 0.1]
    SampleNum_onescv = int(SampleNum / 2)
    Data_class = {}
    # 
    # Bearing_class_name = ['ball', 'comb', 'health', 'inner', 'outer']
    # Bearing_dilepath = filepath + '/' + 'bearingset'
    # # 
    # Bearing_file_names = [f for f in os.listdir(Bearing_dilepath)]
    # # print(Bearing_file_names)
    #
    # for Bear_fault_i in Bearing_class_name:
    #     Data_class[Bear_fault_i] = []
    #     for Bearing_csv_i in Bearing_file_names:
    #         if Bear_fault_i in Bearing_csv_i:
    #             # 
    #             file_path_csv = os.path.join(Bearing_dilepath, Bearing_csv_i)
    #
    #             Data_csv_rows = Read_csv_Data(file_path_csv, row_num=int(
    #                 SampleLength + ((SampleNum_onescv - 1) * 0.5 * SampleLength)),
    #                                           lienum=8, isTime=True)  # 
    #             Data_csv_rows = np.array(Data_csv_rows)
    #             if stander:
    #                 Data_csv_rows = standardize(Data_csv_rows)
    #             Bearing_csv_i_Sample = Biuld_Sample(Data_csv_rows, SampleNum_onescv, SampleLength)  # 
    #             Data_class[Bear_fault_i].extend(Bearing_csv_i_Sample)  # 

    # 
    Gear_class_name = ['Chipped', 'Miss', 'Root', 'Surface']
    Gear_dilepath = filepath + '/' + 'gearset'
    # 
    Gear_file_names = [f for f in os.listdir(Gear_dilepath)]
    # print(Gear_file_names)
    for Gear_fault_i in Gear_class_name:
        Data_class[Gear_fault_i] = []
        for Gear_csv_i in Gear_file_names:
            if Gear_fault_i in Gear_csv_i:
                # 
                file_path_csv = os.path.join(Gear_dilepath, Gear_csv_i)
                Data_csv_rows = Read_csv_Data(file_path_csv, row_num=int(
                    SampleLength + ((SampleNum_onescv - 1) * 0.5 * SampleLength)),
                                              lienum=8, isTime=True)  # 
                Data_csv_rows = np.array(Data_csv_rows)
                if stander:
                    Data_csv_rows = standardize(Data_csv_rows)
                Gear_csv_i_Sample = Biuld_Sample(Data_csv_rows, int(SampleNum / 2), SampleLength)  # 
                Data_class[Gear_fault_i].extend(Gear_csv_i_Sample)  # 

    Train_X = []
    Train_Y = []
    Valid_X = []
    Valid_Y = []
    Test_X = []
    Test_Y = []
    OneClassTrainNum = int(SampleNum * Rate[0])
    OneClassVaildNum = int(SampleNum * Rate[1])
    OneClassTestNum = int(SampleNum * Rate[2])
    for index, (key, value) in enumerate(Data_class.items()):
        Train_X += Data_class[key][0: OneClassTrainNum]
        Train_Y += [index] * OneClassTrainNum

        Valid_X += Data_class[key][OneClassTrainNum: OneClassTrainNum + OneClassVaildNum]
        Valid_Y += [index] * OneClassVaildNum

        Test_X += Data_class[key][
                  OneClassTrainNum + OneClassVaildNum: OneClassTrainNum + OneClassVaildNum + OneClassTestNum]
        Test_Y += [index] * OneClassTestNum

    Train_X = np.asarray(Train_X)
    Valid_X = np.asarray(Valid_X)
    Test_X = np.asarray(Test_X)

    Test_Y = np.asarray(Test_Y, dtype=np.int32)
    Valid_Y = np.asarray(Valid_Y, dtype=np.int32)
    Train_Y = np.asarray(Train_Y, dtype=np.int32)


    return Train_X[:, 0, :], Train_Y, Valid_X[:, 0, :], Valid_Y, Test_X[:, 0, :], Test_Y

########################################## HUST
def Read_xls_data(file_path, row_num=102400, isTime=True):
    # Excel
    workbook = xlrd.open_workbook(file_path)
    # 
    sheet = workbook.sheet_by_index(0)

    data_started = False
    data_rows = []
    row_count = 0

    # 
    for i in range(sheet.nrows):
        row = sheet.row_values(i)

        # 
        if row and str(row[0]).strip().lower() == 'data':
            data_started = True
            continue

        # 
        if data_started and row:
            # ，\t
            if len(row) == 1 and '\t' in str(row[0]):
                # ，\t
                row = str(row[0]).split('\t')

            # 8，0.0
            if isTime:
                processed_row = [float(value) if value else 0.0 for value in row[:5]]
            else:
                processed_row = [float(value) if value else 0.0 for value in row[1:5]]
            data_rows.append(processed_row)
            row_count += 1  # 

            # ，
            if row_count == row_num:
                break

    return data_rows


def Read_txt_from(file_path, num_rows=102400, lienum=5, isTime=True):
    data_started = False
    data_rows = []
    with open(file_path, 'r') as file:
        for line in file:
            # 
            if line.strip() == "Time (seconds) and Data Channels":
                data_started = True
                continue

            # 
            if data_started:
                # 
                row = line.strip().split()
                if row:  # 
                    # （Pythonend_col）
                    if isTime:
                        selected_row = row[:lienum]
                    else:
                        selected_row = row[1:lienum]
                    data_rows.append([float(x) for x in selected_row])  # float
                    if len(data_rows) == num_rows:  # 
                        break
    return data_rows


def Read_Data_From_HUST(filepath, SampleNum=200, SampleLength=1024, Rate=None, stander=True, lienum=5, isTime=True):
    print("Creating Data Sets.....HUST")
    if Rate is None:
        Rate = [0.7, 0.2, 0.1]

    Data_class = {}
    # 
    Bearing_class_name = ['0.5X_B', '0.5X_C', '0.5X_I', '0.5X_O', 'B', 'C', 'I', 'O', 'H']
    Bearing_dilepath = filepath + '/' + 'bearing'
    # 
    Bearing_file_names = [f for f in os.listdir(Bearing_dilepath)]
    Bearing_file_type = [0 for f in os.listdir(Bearing_dilepath)]
    # print(Bearing_file_names)
    # print(Bearing_file_type)
    SampleNum_onescv = int(SampleNum / 1)
    for Bear_fault_i in Bearing_class_name:
        Data_class[Bear_fault_i] = []
        for Bearing_csv_i in Bearing_file_names:
            if Bear_fault_i in Bearing_csv_i and Bearing_file_type[Bearing_file_names.index(Bearing_csv_i)] == 0:
                Bearing_file_type[Bearing_file_names.index(Bearing_csv_i)] = 1

                file_path_csv = os.path.join(Bearing_dilepath, Bearing_csv_i)

                Data_csv_rows = Read_csv_Data(file_path_csv, row_num=int(
                    SampleLength + ((SampleNum_onescv - 1) * 0.5 * SampleLength)),
                                              lienum=lienum, isTime=True)  # 
                Data_csv_rows = np.array(Data_csv_rows)
                if stander:
                    Data_csv_rows = standardize(Data_csv_rows)
                Bearing_csv_i_Sample = Biuld_Sample(Data_csv_rows, SampleNum_onescv, SampleLength)  # 
                Data_class[Bear_fault_i].extend(Bearing_csv_i_Sample)  # 

    # 
    Gear_class_name = ['B_', 'M_']
    Gear_dilepath = filepath + '/' + 'gearbox'
    # 
    Gear_file_names = [f for f in os.listdir(Gear_dilepath)]

    for Gear_fault_i in Gear_class_name:
        Data_class[Gear_fault_i] = []
        for Gear_csv_i in Gear_file_names:
            if Gear_fault_i in Gear_csv_i:
                # print(Gear_csv_i)
                # 
                file_path_csv = os.path.join(Gear_dilepath, Gear_csv_i)
                Data_csv_rows = Read_txt_from(file_path_csv, num_rows=int(
                    SampleLength + ((SampleNum_onescv - 1) * 0.5 * SampleLength)),
                                              lienum=lienum, isTime=True)  # 
                Data_csv_rows = np.array(Data_csv_rows)
                if stander:
                    Data_csv_rows = standardize(Data_csv_rows)
                Gear_csv_i_Sample = Biuld_Sample(Data_csv_rows, int(SampleNum / 1), SampleLength)  # 
                Data_class[Gear_fault_i].extend(Gear_csv_i_Sample)  # 

    Train_X = []
    Train_Y = []
    Valid_X = []
    Valid_Y = []
    Test_X = []
    Test_Y = []

    OneClassTrainNum = int(SampleNum * Rate[0])
    OneClassVaildNum = int(SampleNum * Rate[1])
    OneClassTestNum = int(SampleNum * Rate[2])
    for index, (key, value) in enumerate(Data_class.items()):
        Train_X += Data_class[key][0: OneClassTrainNum]
        Train_Y += [index] * OneClassTrainNum

        Valid_X += Data_class[key][OneClassTrainNum: OneClassTrainNum + OneClassVaildNum]
        Valid_Y += [index] * OneClassVaildNum

        Test_X += Data_class[key][
                  OneClassTrainNum + OneClassVaildNum: OneClassTrainNum + OneClassVaildNum + OneClassTestNum]
        Test_Y += [index] * OneClassTestNum

    Train_X = np.asarray(Train_X)
    Valid_X = np.asarray(Valid_X)
    Test_X = np.asarray(Test_X)

    Test_Y = np.asarray(Test_Y, dtype=np.int32)
    Valid_Y = np.asarray(Valid_Y, dtype=np.int32)
    Train_Y = np.asarray(Train_Y, dtype=np.int32)


    return Train_X[:, 2, :], Train_Y, Valid_X[:, 2, :], Valid_Y, Test_X[:, 2, :], Test_Y


###################################################### HIT

def Read_npy_from(file_path, one_Sample_Length=1204):
    # 
    np.random.seed(42)
    data_all = np.load(file_path, encoding="latin1")  # 
    # 6
    data = data_all[:, :6, :one_Sample_Length]
    label = data_all[0, 7, 0]
    shuffled_data = data[np.random.permutation(int(len(data)))]
    # data (N, )
    return shuffled_data, label


def Read_Data_From_HIT(filepath, SampleNum=200, SampleLength=1024, Rate=None):
    print("Creating Data Sets.....HIT")
    if Rate is None:
        Rate = [0.7, 0.2, 0.1]

    Data_class = {}
    # 
    file_names = [f for f in os.listdir(filepath)]
    # print(file_names)
    for file_names_i in file_names:
        file_path_csv = os.path.join(filepath, file_names_i)
        data, label = Read_npy_from(file_path_csv,one_Sample_Length=SampleLength)
        if str(label) in Data_class:

            Data_class[str(label)] = np.concatenate((Data_class[str(label)], data), axis=0)
        else:
            Data_class[str(label)] = data


    Train_X = []
    Train_Y = []
    Valid_X = []
    Valid_Y = []
    Test_X = []
    Test_Y = []

    OneClassTrainNum = int(SampleNum * Rate[0])
    OneClassVaildNum = int(SampleNum * Rate[1])
    OneClassTestNum = int(SampleNum * Rate[2])

    for index, (key, value) in enumerate(Data_class.items()):
        Train_X += Data_class[key][0: OneClassTrainNum].tolist()
        Train_Y += [index] * OneClassTrainNum

        Valid_X += Data_class[key][OneClassTrainNum: OneClassTrainNum + OneClassVaildNum].tolist()
        Valid_Y += [index] * OneClassVaildNum

        Test_X += Data_class[key][
                  OneClassTrainNum + OneClassVaildNum: OneClassTrainNum + OneClassVaildNum + OneClassTestNum].tolist()
        Test_Y += [index] * OneClassTestNum

    print("Created Successfully; Number of channels:6;Class:{};SampleNum:{};SampleLength:{}".format(
        len(Data_class), SampleNum, SampleLength))

    Train_X = np.asarray(Train_X)
    Valid_X = np.asarray(Valid_X)
    Test_X = np.asarray(Test_X)

    Test_Y = np.asarray(Test_Y, dtype=np.int32)
    Valid_Y = np.asarray(Valid_Y, dtype=np.int32)
    Train_Y = np.asarray(Train_Y, dtype=np.int32)

    return Train_X[:, 5, :], Train_Y, Valid_X[:, 5, :], Valid_Y, Test_X[:, 5, :], Test_Y





if __name__ == '__main__':

    read_OrginCWRU_fromOneOH_fewshot(filepath='../data/CWRU',Select_shot=[0,1,2])

    # Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y = read_OrginCWRU(filepath='../data/CWRU',
    #                                                                     SampleLength=1024,
    #                                                                     SampleNum=200,
    #                                                                     normal=False,
    #                                                                     Rate=[0.7, 0.2, 0.1]
    #                                                                     )
    # #
    # # Train_X = addNoiseBatch(Train_X, -4)
    # # Valid_X = addNoiseBatch(Valid_X, -4)
    # # Test_X = addNoiseBatch(Test_X, -4)
    # print(Train_X.shape)
    # print(Train_Y.shape)
    # print(Valid_X.shape)
    # print(Valid_Y.shape)
    # print(Test_X.shape)
    # print(Test_Y.shape)
    # import matplotlib.pyplot as plt
    # #
    # # #  Train_X 
    # signal_to_plot = Train_X[5, :]
    #
    # # 
    # color = 'b'  # ，
    #
    # # 
    # plt.figure(figsize=(10, 6))
    #
    # # 
    # # plt.plot(signal_to_plot, color=color)
    # # plt.title('Signal Plot')
    # # plt.xlabel('Sample Index')
    # # plt.ylabel('Amplitude')
    # # plt.show()
    #
    #
    #
    # Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y = read_OrginJNU(filepath='../data/JNU',
    #                                                                     SampleLength=1024,
    #                                                                     SampleNum=200,
    #                                                                     normal=False,
    #                                                                     Rate=[0.7, 0.2, 0.1]
    #                                                                     )
    #
    # # Train_X = addPinkNoiseBatch(Train_X, -4)
    # # Valid_X = addPinkNoiseBatch(Valid_X, -4)
    # # Test_X = addPinkNoiseBatch(Test_X, -4)
    # print(Train_X.shape)
    # print(Train_Y.shape)
    # print(Valid_X.shape)
    # print(Valid_Y.shape)
    # print(Test_X.shape)
    # print(Test_Y.shape)
    #
    #
    # folder_path = "../data/SEU"
    #
    # Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y = Read_Data_From_SEU(folder_path, SampleNum=200, SampleLength=1024, Rate=[0.7, 0.2, 0.1], isMaxMin=False)
    #
    # # NumPyPyTorch，dtype=torch.int64
    # Train_X = torch.tensor(Train_X, dtype=torch.float32)
    # Train_Y = torch.tensor(Train_Y, dtype=torch.int64)
    # Valid_X = torch.tensor(Valid_X, dtype=torch.float32)
    # Valid_Y = torch.tensor(Valid_Y, dtype=torch.int64)
    # Test_X = torch.tensor(Test_X, dtype=torch.float32)
    # Test_Y = torch.tensor(Test_Y, dtype=torch.int64)
    #
    # print("SEU",Train_X.shape)
    #
    Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y = Read_Data_From_HUST('../data/HUST', SampleNum=200, SampleLength=1024, Rate=None,  lienum=5, isTime=True)

    Train_X = torch.tensor(Train_X, dtype=torch.float32)
    Train_Y = torch.tensor(Train_Y, dtype=torch.int64)
    Valid_X = torch.tensor(Valid_X, dtype=torch.float32)
    Valid_Y = torch.tensor(Valid_Y, dtype=torch.int64)
    Test_X = torch.tensor(Test_X, dtype=torch.float32)
    Test_Y = torch.tensor(Test_Y, dtype=torch.int64)
    print(Train_X.shape)
    #
    # Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y = Read_Data_From_HIT('../data/HIT', SampleNum=200, SampleLength=1024, Rate=None)
    #
    # Train_X = torch.tensor(Train_X, dtype=torch.float32)
    # Train_Y = torch.tensor(Train_Y, dtype=torch.int64)
    # Valid_X = torch.tensor(Valid_X, dtype=torch.float32)
    # Valid_Y = torch.tensor(Valid_Y, dtype=torch.int64)
    # Test_X = torch.tensor(Test_X, dtype=torch.float32)
    # Test_Y = torch.tensor(Test_Y, dtype=torch.int64)
    # print(Train_X.shape)