import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import mse_loss

class CNN1DRegressor(nn.Module):
    def __init__(self, input_length=3):
        super(CNN1DRegressor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        # 假设输入长度为3，经过池化层后长度减半为1.5，取整为1
        self.fc1 = nn.Linear(64 * (input_length // 2), 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_cnn_model(X_train, y_train, num_epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1DRegressor(input_length=X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32).unsqueeze(1).to(device)
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).to(device)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    
    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1, 1))
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    return model

def evaluate_cnn1d_model(model, X_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32).unsqueeze(1).to(device)
    y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        predictions = model(X_test_tensor).view(-1)
        # 计算MRE
        mre = torch.mean(torch.abs((y_test_tensor - predictions) / y_test_tensor)).item()
        # 计算MSE
        mse = mse_loss(predictions, y_test_tensor).item()
    
    return mse, mre

def predict_with_loaded_model(model_path, input_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1DRegressor(input_length=3).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(input_tensor).item()
    
    return prediction

# 示例用法
if __name__ == "__main__":
    from data_loader import load_data_3utilization, split_data

    # 加载数据
    file_path = 'dataset/ieee-mean-active.csv'
    X, y = load_data_3utilization(file_path)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 训练模型
    cnn_model = train_cnn_model(X_train, y_train, num_epochs=50)

    # 评估模型
    mse, mre = evaluate_cnn1d_model(cnn_model, X_test, y_test)
    print(f"CNN1D Model MSE: {mse}, MRE: {mre}")

    # 保存模型
    torch.save(cnn_model.state_dict(), 'cnn1d_model.pth')