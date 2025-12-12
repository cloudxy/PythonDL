import numpy as np
import matplotlib.pyplot as plt

# 解决Mac系统Matplotlib中文显示和字体缺失警告问题
plt.rcParams["font.family"] = "Arial Unicode MS"  # 使用Mac系统兼容的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# 解决matplotlib在无GUI环境下的显示问题（可选但建议添加）
# 也可根据环境换成'agg'（仅保存不显示）、'Qt5Agg'等
plt.switch_backend('TkAgg')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input  # 新增：导入Input层
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


class LstmClass:

    def sin_wave(self):
        # ---------------------- 1. 生成模拟时序数据 ----------------------
        # 生成1000个时间点的正弦波数据（加噪声模拟真实场景）
        time_steps = np.arange(1000)
        data = np.sin(time_steps * 0.02) + np.random.randn(1000) * 0.1  # 正弦波+高斯噪声
        data = data.reshape(-1, 1)  # 转为 (样本数, 特征数) 格式

        # ---------------------- 2. 数据预处理 ----------------------
        # 1. 归一化（LSTM对数据范围敏感，通常归一到[0,1]或[-1,1]）
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)

        # 2. 构造时序输入输出对：用前 look_back 个值预测下1个值
        look_back = 10  # 时间窗口大小（可调整）
        X, y = [], []
        for i in range(look_back, len(data_scaled)):
            X.append(data_scaled[i-look_back:i, 0])  # 输入：前look_back个时间步
            y.append(data_scaled[i, 0])              # 输出：第i个时间步（预测目标）

        # 3. 转换为数组并调整维度：LSTM输入要求 (样本数, 时间步, 特征数)
        X = np.array(X)
        y = np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # 最终维度：(n_samples, look_back, 1)

        # 4. 划分训练集和测试集（后20%为测试集）
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # ---------------------- 3. 构建LSTM模型（关键修改部分） ----------------------
        model = Sequential()
        # 新增：使用Input层定义输入形状，替代LSTM层中的input_shape参数
        model.add(Input(shape=(look_back, 1)))  # 输入形状：(时间步, 特征数)
        # LSTM层：32个隐藏单元，不再需要input_shape参数
        model.add(LSTM(units=32))
        # 输出层：1个神经元（回归任务，预测连续值）
        model.add(Dense(units=1))

        # 编译模型：优化器用Adam，损失函数用MSE（回归任务常用）
        model.compile(optimizer='adam', loss='mean_squared_error')

        # ---------------------- 4. 训练模型 ----------------------
        history = model.fit(
            X_train, y_train,
            epochs=20,          # 训练轮数（可调整）
            batch_size=32,      # 批次大小（可调整）
            validation_data=(X_test, y_test)  # 验证集（测试集）
        )

        # ---------------------- 5. 模型预测与反归一化 ----------------------
        # 预测（训练集+测试集）
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # 反归一化：将预测值转换回原始数据范围
        y_train_pred = scaler.inverse_transform(y_train_pred)
        y_test_pred = scaler.inverse_transform(y_test_pred)
        y_train_true = scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test_true = scaler.inverse_transform(y_test.reshape(-1, 1))

        # ---------------------- 6. 模型评估 ----------------------
        # 计算MSE和RMSE
        train_rmse = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
        print(f"训练集RMSE: {train_rmse:.4f}")
        print(f"测试集RMSE: {test_rmse:.4f}")

        # ---------------------- 7. 结果可视化 ----------------------
        plt.figure(figsize=(12, 6))

        # 绘制训练过程的损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='训练损失')
        plt.plot(history.history['val_loss'], label='测试损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.title('训练/测试损失曲线')

        # 绘制真实值与预测值对比
        plt.subplot(1, 2, 2)
        # 拼接训练集真实值、测试集真实值、测试集预测值（保持时间顺序）
        train_plot = np.empty_like(data)
        train_plot[:look_back] = np.nan  # 前look_back个值无预测
        train_plot[look_back:train_size+look_back] = y_train_pred
        test_plot = np.empty_like(data)
        test_plot[:train_size+look_back] = np.nan
        test_plot[train_size+look_back:] = y_test_pred

        plt.plot(data, label='原始数据')
        plt.plot(train_plot, label='训练集预测')
        plt.plot(test_plot, label='测试集预测')
        plt.xlabel('时间步')
        plt.ylabel('数值')
        plt.legend()
        plt.title('LSTM时序预测结果')

        plt.tight_layout()
        plt.show()

# 修复主程序入口：使用Python标准的入口语句，实例化类并调用方法
if __name__ == "__main__":
    # 1. 实例化LstmClass类
    lstm_instance = LstmClass()
    # 2. 调用sin_wave方法执行逻辑
    lstm_instance.sin_wave()