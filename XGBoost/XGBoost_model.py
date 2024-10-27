import xgboost as xgb
from sklearn.model_selection import GridSearchCV

class XGBoostModel:
    def __init__(self, params=None):
        if params is None:
            # 设置默认参数
            self.params = {
                'objective': 'multi:softmax',
                'num_class': 8,  # 根据实际类别数调整
                'learning_rate': 0.1,
                'max_depth': 6,
                'n_estimators': 100,
                'verbosity': 1,
                'seed': 42
            }
        else:
            self.params = params
        self.model = xgb.XGBClassifier(**self.params)

    def train(self, X, y):
        """训练模型"""
        self.model.fit(X, y)
        return self

    def save_model(self, path='xgboost_model.json'):
        """保存模型"""
        self.model.save_model(path)

    def load_model(self, path='xgboost_model.json'):
        """加载模型"""
        self.model.load_model(path)

    def predict(self, X):
        """进行预测"""
        return self.model.predict(X)

    def evaluate(self, X, y):
        """评估模型性能"""
        y_pred = self.predict(X)
        accuracy = (y_pred == y).mean()
        return accuracy

    def grid_search(self, X, y, param_grid, cv=3):
        """进行超参数网格搜索"""
        grid_search = GridSearchCV(self.model, param_grid, cv=cv, scoring='accuracy', verbose=1)
        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_
        return grid_search.best_params_

    def get_params(self):
        """获取当前模型的参数"""
        return self.model.get_params()
