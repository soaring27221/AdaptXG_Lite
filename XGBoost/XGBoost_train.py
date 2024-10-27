import xgboost as xgb
from XGBoost_model import create_xgboost_model
from utils import load_data, save_model
from config import feature_columns,label_column,input_size,DatasetForTraining


def train_model(train_loader, num_classes, epochs=50, lr=0.001):
    model = create_xgboost_model()
    model.fit(X_train, y_train)
    return model


if __name__ == '__main__':

    # 加载数据
    train_loader, num_classes,encoders, label_encoder = load_data(DatasetForTraining, feature_columns, label_column)

    # 训练模型
    model = train_model(X_train, y_train)

    # 保存模型
    save_model(model)

    print("Model training complete and saved.")