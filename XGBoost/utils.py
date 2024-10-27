import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import pickle

# 定义颜色码
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
ENDC = "\033[0m"  # 重置颜色

#
def encode_string_columns(df, feature_columns, encoders=None, default_value=-1):
    '''
    用于处理string类型字段的标签编码，并打印编码对应关系
    :param df:
    :param feature_columns:
    :param encoders:
    :param default_value:
    :return:
    '''
    if encoders is None:
        encoders = {}

    for col in feature_columns:
        if df[col].dtype == 'object':  # 检查字段是否为字符串类型
            if col not in encoders:  # 如果没有现成的编码器，则新建
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                encoders[col] = le
            else:  # 使用已有的编码器
                le = encoders[col]
                # 将未见过的标签编码为默认值
                transformed_values = []
                for value in df[col]:
                    if value in le.classes_:
                        transformed_values.append(le.transform([value])[0])
                    else:
                        transformed_values.append(default_value)
                df[col] = transformed_values

            # 打印编码对应关系
            print(f'{BLUE}Encoding for column{ENDC} "{col}":')
            for class_value, encoded_value in zip(le.classes_, range(len(le.classes_))):
                print(f'  {class_value} -> {encoded_value}')
            print(f'  Unseen values -> {default_value}')

    return df, encoders


def encode_string_columns_onehot(df, feature_columns, encoders=None):
    '''用于处理 string 类型字段的独热编码，并打印编码对应关系
    这里应该明确一下这个函数的功能：
    我们要把非数值型的数据转化成one-hot编码，也就是升维
    例如service被转化为service_tcp、service_udp、service_icmp这三个维度
    升维会导致什么后果呢？维度提升，那么就不能用原来的feature_columns了，而且input_size也会发生相应的变化
    '''
    if encoders is None:
        encoders = {}

    # 逐列检查和处理
    for col in feature_columns:
        print("检查列："+str(col))
        if df[col].dtype == 'object':  # 检查字段是否为字符串类型

            # 打印转化前的类别信息
            unique_values = df[col].unique()
            print(f'{col} 对应 {len(unique_values)} 个属性，分别为: {", ".join(unique_values)}')

            # 如果没有现成的编码器，则生成新的独热编码
            if col not in encoders:
                # 对该列执行独热编码
                dummies = pd.get_dummies(df[col], prefix=col)
                print("对该列执行独热编码:\n"+str(dummies))

                # 确保新增的特征项为0和1
                dummies = dummies.astype(int)

                # 删除原列，并将独热编码的列合并到原数据集中
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                print("删除原列后:\n" + str(df))

                # 将编码信息存储到 encoders 中，以便未来使用
                encoders[col] = dummies.columns.tolist()
                print("将编码信息存储到 encoders 中，以便未来使用: "+str(encoders))
            else:  # 测试集处理
                # 使用已有的编码器
                # 创建一个全零的 DataFrame
                dummies = pd.DataFrame(0, index=df.index, columns=encoders[col])

                # 将测试集中存在的类别填充为 1
                new_dummies = pd.get_dummies(df[col], prefix=col)
                for dummy_col in new_dummies.columns:
                    if dummy_col in dummies.columns:
                        dummies[dummy_col] = new_dummies[dummy_col].astype(int)

                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

            '''# 打印编码对应关系
            print(f'One-hot encoding for column "{col}":')
            for category in encoders[col]:
                print(f'  {category.split("_")[-1]} -> {category}')'''

    # 更新 feature_columns
    print(f"{YELLOW}原特征项:{ENDC} "+str(feature_columns))
    print("原特征项个数: "+str(len(feature_columns)))
    updated_feature_columns = df.columns.tolist()
    print(f"{GREEN}更新特征项为:{ENDC} "+str(updated_feature_columns))
    print("更新特征项个数为: " + str(len(updated_feature_columns)))

    input_size=len(updated_feature_columns)#也就是实际输入的维度


    return df, encoders, input_size,updated_feature_columns


def encode_labels(df, label_column, label_encoder=None,default_value=-1):
    '''
    用于处理标签列的编码并打印编码对应关系
    :param df:
    :param label_column:
    :param label_encoder:
    :param default_value:
    :return:
    '''
    if label_encoder is None:#如果没有现成的标签编码器，则新建
        label_encoder = LabelEncoder()
        df[label_column] = label_encoder.fit_transform(df[label_column])
    else:
        transformed_labels = []
        for label in df[label_column]:
            if label in label_encoder.classes_:
                transformed_labels.append(label_encoder.transform([label])[0])
            else:
                transformed_labels.append(default_value)  # 默认值处理
        df[label_column] = transformed_labels

    # 打印编码对应关系
    print(f'{YELLOW}Encoding for label column{ENDC} "{label_column}":')
    for class_value, encoded_value in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
        print(f'  {class_value} -> {encoded_value}')
    print(f'  Unseen values -> {default_value}')
    return df, label_encoder






def load_data(file_path, feature_columns, label_column, encoders=None, label_encoder=None):
    '''
    加载数据
    将CSV数据转化为张量

    :param file_path:数据集的路径
    :param feature_columns:选择使用数据集的哪些列作为分类特征feature
    :param label_column:选择数据集的哪些列作为标签label
    :param batch_size:
    :param encoders:输入的特征列的编码器，默认是啥都没有，自己创建一个，但测试时需要使用训练时创建的编码器
    :param label_encoder:

    :return dataloader:
    :return num_classes:标签类别的数量，例如恶意流量就是个二分类，本质上就俩标签
    :return real_input_size:真正的input_size，因为one_hot编码升维了，所以输入模型的特征向量维度比选择的数据集特征维度要高
    :return encoders:返回的特征列的编码器，如果需要保存编码器的话
    :return label_encoder:标签编码器，如果需要保存编码器的话
    '''
    df = pd.read_csv(file_path)
    # 只保留特征列和标签列
    df = df[feature_columns + [label_column]]

    # 对字符串类型的列进行标签编码，并打印编码对应关系
    #df, encoders = encode_string_columns(df, feature_columns, encoders)#不用这个了，采用独热编码搞
    df, encoders,real_input_size,updated_feature_columns= encode_string_columns_onehot(df, feature_columns, encoders)

    # 对标签列进行编码
    df, label_encoder = encode_labels(df, label_column, label_encoder)


    # 提取特征和标签
    X = df[updated_feature_columns].values
    y = df[label_column].values

    print("提取特征和标签：\n" + str(X)+"\n"+str(y))

    # 转换为张量
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # 创建数据集和数据加载器
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, shuffle=True)

    return dataloader, len(label_encoder.classes_), real_input_size,encoders, label_encoder

# 保存编码器（包括特征列的编码器和标签编码器）
def save_encoders(encoders, label_encoder, path='encoders.pkl'):
    with open(path, 'wb') as f:
        pickle.dump({'encoders': encoders, 'label_encoder': label_encoder}, f)

# 加载编码器
def load_encoders(path='encoders.pkl'):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['encoders'], data['label_encoder']

# 保存模型
def save_model(model, path='dnn_model.pth'):
    torch.save(model.state_dict(), path)


# 加载模型
def load_model(model, path='dnn_model.pth'):
    model.load_state_dict(torch.load(path))
    return model
