import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('worldcities.csv')
data['target'] = data.apply(
    lambda row: 1 if row['country'] == 'Russia' else 0, axis=1)

# Подготовка данных
X = data[['lat', 'lng']].values
y = data['target'].values

# Нормализация данных
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

class DeepNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DeepNN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

        # Задаём функцию активации
        self.activation = nn.Tanh()
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_activation(self.output_layer(x))
        return x

# Инициализация сети
input_size = 2
hidden_sizes = [20, 10]
output_size = 1
model = DeepNN(input_size, hidden_sizes, output_size)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Обучение модели
epochs = 6000
train_losses = []
p_scores = []

for epoch in range(epochs):
    model.train()
    
    # Прямой проход
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Обратное распространение
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_tensor)
        y_pred_test_binary = (y_pred_test > 0.5).float()
        
        # Вычисление p
        p_test_1 = torch.mean((y_pred_test_binary[y_test_tensor == 1] == 1).float()).item()
        p_test_0 = torch.mean((y_pred_test_binary[y_test_tensor == 0] == 0).float()).item()
        p_score = 0.5 * (p_test_1 + p_test_0)
        p_scores.append(p_score)
        
    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}, p: {p_score:.4f}")

# Вывод финального результата
print(f"Final p score: {p_scores[-1]:.4f}")

# График изменения метрики p
plt.figure(figsize=(10, 6))
plt.plot(p_scores, label='p score')
plt.axhline(0.97, color='red', linestyle='--', label='Target p > 0.97')
plt.xlabel('Epoch')
plt.ylabel('p score')
plt.title('p score during training')
plt.legend()
plt.show()

# График предсказания классов
with torch.no_grad():
    y_pred_full = model(torch.tensor(X, dtype=torch.float32))
    y_pred_class = (y_pred_full > 0.5).float().numpy()
plt.scatter(X[:, 1], X[:, 0], c=y_pred_class[:, 0], cmap='coolwarm', alpha=0.6)
plt.ylabel('Latitude') 
plt.xlabel('Longitude')
plt.title('Predicted Classes')
plt.show()

# Вычисление количества умножений
# Подсчёт количества умножений для сети с несколькими скрытыми слоями
multiplications_count = input_size * hidden_sizes[0]  # Умножения на первом слое
for i in range(1, len(hidden_sizes)):
    multiplications_count += hidden_sizes[i-1] * hidden_sizes[i]  # Умножения между скрытыми слоями
multiplications_count += hidden_sizes[-1] * output_size  # Умножения на выходном слое

print(f"Количество умножений: {multiplications_count}")


# Дальше я попробую уменьшить число умножений
# Инициализация сети
input_size = 2
hidden_sizes = [10, 10]
output_size = 1
model = DeepNN(input_size, hidden_sizes, output_size)

# Оптимизатор и функция потерь
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Обучение модели
epochs = 6000
train_losses = []
p_scores = []

for epoch in range(epochs):
    model.train()
    
    # Прямой проход
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Обратное распространение
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    
    # Оценка на тесте
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_tensor)
        y_pred_test_binary = (y_pred_test > 0.5).float()
        
        # Вычисление p
        p_test_1 = torch.mean((y_pred_test_binary[y_test_tensor == 1] == 1).float()).item()
        p_test_0 = torch.mean((y_pred_test_binary[y_test_tensor == 0] == 0).float()).item()
        p_score = 0.5 * (p_test_1 + p_test_0)
        p_scores.append(p_score)
        
    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}, p: {p_score:.4f}")

# Вывод финального результата
print(f"Final p score: {p_scores[-1]:.4f}")

# График изменения метрики p
plt.figure(figsize=(10, 6))
plt.plot(p_scores, label='p score')
plt.axhline(0.97, color='red', linestyle='--', label='Target p > 0.97')
plt.xlabel('Epoch')
plt.ylabel('p score')
plt.title('p score during training')
plt.legend()
plt.show()

# График предсказания классов (пример)
with torch.no_grad():
    y_pred_full = model(torch.tensor(X, dtype=torch.float32))
    y_pred_class = (y_pred_full > 0.5).float().numpy()
plt.scatter(X[:, 1], X[:, 0], c=y_pred_class[:, 0], cmap='coolwarm', alpha=0.6)
plt.ylabel('Latitude')  # Меняем метки осей
plt.xlabel('Longitude')
plt.title('Predicted Classes')
plt.show()

# Вычисление количества умножений
# Подсчёт количества умножений для сети с несколькими скрытыми слоями
multiplications_count = input_size * hidden_sizes[0]  # Умножения на первом слое
for i in range(1, len(hidden_sizes)):
    multiplications_count += hidden_sizes[i-1] * hidden_sizes[i]  # Умножения между скрытыми слоями
multiplications_count += hidden_sizes[-1] * output_size  # Умножения на выходном слое

print(f"Количество умножений: {multiplications_count}")

# Мне удалось добиться уменьшения количества умножений с 250 до 130
