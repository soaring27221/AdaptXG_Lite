import xgboost as xgb
from utils import load_data, load_model
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    feature_columns = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8']  # 特征列名
    label_column = 'attack_cat'  # 标签列名

    # 加载测试数据
    X_test, y_test = load_data('UNSW_NB15-test.csv', feature_columns, label_column)

    # 加载模型
    model = load_model()

    # 进行预测
    y_pred = model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
