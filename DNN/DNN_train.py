import torch
import torch.optim as optim
import torch.nn as nn
from DNN_model import DNNModel
from utils import load_data, save_model,save_encoders
from config import feature_columns,label_column,input_size,DatasetForTraining

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 训练函数
def train_model(train_loader, input_size, num_classes, epochs=50, lr=0.001):
    model = DNNModel(input_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}')

    save_model(model)


if __name__ == '__main__':
    '''feature_columns = ['dur', 'proto', 'service', 'state','spkts',
                       'dpkts', 'sbytes', 'dbytes','rate','sttl',
                       'dttl', 'sload', 'dload','sloss','dloss',
                       'sinpkt', 'dinpkt', 'sjit','djit','swin',
                       'stcpb', 'dtcpb', 'dwin','tcprtt','synack',
                       'ackdat', 'smean', 'dmean','trans_depth','response_body_len',
                       'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm','ct_src_dport_ltm','ct_dst_sport_ltm',
                       'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd','ct_flw_http_mthd','ct_src_ltm',
                       'ct_srv_dst', 'is_sm_ips_ports']  # 特征列名
    label_column = 'label'  # 标签列名'''
    batch_size = 8192
    epochs = 300
    lr=0.001

    # 加载数据
    train_loader, num_classes,real_input_size,encoders, label_encoder= load_data(DatasetForTraining, feature_columns, label_column, batch_size)
    save_encoders(encoders, label_encoder, 'encoders.pkl')#保存编码器，也就是数据集里string值与int编码的对应关系
    print("数据加载完成，准备开始训练...")

    train_model(train_loader, real_input_size, num_classes=num_classes, epochs=epochs,lr=lr)
