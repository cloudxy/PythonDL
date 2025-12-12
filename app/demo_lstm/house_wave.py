import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

class XgBoostClass:

    def house_wave(self):
        # ---------------------- 1. 生成模拟表格数据 ----------------------
        X, y = make_regression(
            n_samples=1000, n_features=5, n_informative=3,
            noise=10, random_state=42
        )

        feature_names = [f'feature_{i + 1}' for i in range(5)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y

        # ---------------------- 2. 数据预处理 ----------------------
        X = df[feature_names]
        y = df['target']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 转为XGBoost原生的DMatrix格式
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train, feature_names=feature_names)
        dtest = xgb.DMatrix(X_test_scaled, label=y_test, feature_names=feature_names)

        # ---------------------- 3. 构建XGBoost模型（原生参数） ----------------------
        params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.1,
            'max_depth': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'rmse',
            'random_state': 42
        }

        # 定义验证集
        watchlist = [(dtrain, 'train'), (dtest, 'test')]

        # ---------------------- 4. 训练模型 ----------------------
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=watchlist,
            early_stopping_rounds=10,
            verbose_eval=10
        )

        # ---------------------- 5. 模型预测与评估 ----------------------
        y_train_pred = model.predict(dtrain)
        y_test_pred = model.predict(dtest)

        # 计算评估指标
        def evaluate(y_true, y_pred, set_name):
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            print(f"\n{set_name} 评估结果：")
            print(f"MAE（平均绝对误差）: {mae:.2f}")
            print(f"RMSE（均方根误差）: {rmse:.2f}")
            print(f"R²（决定系数）: {r2:.4f}")

        evaluate(y_train, y_train_pred, "训练集")
        evaluate(y_test, y_test_pred, "测试集")

        # ---------------------- 6. 结果可视化（添加中文字体配置） ----------------------
        # 【关键修改】设置Matplotlib的中文字体，使用你系统可用的字体
        plt.rcParams['font.sans-serif'] = ['Songti SC']  # 宋体（你系统可用的中文字体）

        # 替换为 Arial Unicode MS
        #plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        # 替换为 PingFang HK
        #plt.rcParams['font.sans-serif'] = ['PingFang HK']
        # 替换为 楷体（Kai）
        #plt.rcParams['font.sans-serif'] = ['Kai']


        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

        plt.figure(figsize=(12, 5))

        # 1. 真实值 vs 预测值散点图
        plt.subplot(1, 2, 1)
        plt.scatter(y_test, y_test_pred, alpha=0.6)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title('XGBoost 真实值 vs 预测值')
        plt.grid(True, alpha=0.3)

        # 2. 特征重要性
        plt.subplot(1, 2, 2)
        xgb.plot_importance(model, ax=plt.gca(), importance_type='weight')
        plt.xlabel('特征重要性得分')
        plt.ylabel('特征名称')
        plt.title('XGBoost 特征重要性排名')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    xgboost = XgBoostClass()
    xgboost.house_wave()