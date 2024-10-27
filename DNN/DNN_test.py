import torch
from DNN_model import DNNModel
from utils import load_data, load_model,load_encoders
from config import feature_columns,label_column,input_size,DatasetForTesting

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 测试函数
def test_model(test_loader, input_size, num_classes):
    model = DNNModel(input_size, num_classes).to(device)
    model = load_model(model)
    model.eval()

    correct = 0
    total = 0

    true_positive = [0] * num_classes  # 真正例
    false_positive = [0] * num_classes  # 假正例
    false_negative = [0] * num_classes  # 假负例

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 更新TP、FP、FN
            for i in range(len(labels)):
                if predicted[i] == labels[i]:  # 预测正确
                    true_positive[labels[i]] += 1
                else:  # 预测错误
                    false_positive[predicted[i]] += 1
                    false_negative[labels[i]] += 1

            '''
            print("预测结果："+str(predicted))
            print("实际结果：" + str(labels))
            print("************************************")'''

        # 计算精确率和召回率
        precision = [0] * num_classes
        recall = [0] * num_classes

        for i in range(num_classes):
            if true_positive[i] + false_positive[i] > 0:
                precision[i] = true_positive[i] / (true_positive[i] + false_positive[i])
            if true_positive[i] + false_negative[i] > 0:
                recall[i] = true_positive[i] / (true_positive[i] + false_negative[i])

            print(f'Class {i} - Precision: {precision[i]:.2f}, Recall: {recall[i]:.2f}')

    print(f'Accuracy: {100 * correct / total:.2f}%')

if __name__ == '__main__':
    '''feature_columns = ['dur', 'proto', 'service', 'state', 'spkts',
                       'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl',
                       'dttl', 'sload', 'dload', 'sloss', 'dloss',
                       'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin',
                       'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack',
                       'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len',
                       'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
                       'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm',
                       'ct_srv_dst', 'is_sm_ips_ports']  # 特征列名
    label_column = 'label'  # 标签列名'''
    batch_size = 64

    encoders, label_encoder=load_encoders()#加载保存的编码器

    # 加载测试数据
    test_loader, num_classes ,real_input_size, _ , _ = load_data(DatasetForTesting, feature_columns, label_column, batch_size,encoders, label_encoder)
    print("数据已加载完成，准备测试...")
    # 假设输入的特征数是42
    test_model(test_loader, real_input_size, num_classes=num_classes)
