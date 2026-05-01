import torch
import torch.nn as nn
import torch.optim as optim

# 1. Data Setup
X = torch.tensor([[1.0]], requires_grad=True) 
Y = torch.tensor([[0.0]])

# 2. Structure: Linear(1,1) -> ReLU -> Linear(1,1) -> Sigmoid
model = nn.Sequential(
    nn.Linear(1, 1),
    nn.ReLU(),
    nn.Linear(1, 1),
    nn.Sigmoid()
)

# 3. Loss and Optimizer
criterion = nn.BCELoss() # Binary Cross Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.1) # Stochastic Gradient Descent

# 4. Training Loop
epochs = 100
print(f"Starting training for {epochs} epochs...\n")

for epoch in range(epochs):
    # a. Reset gradients from the previous step to prevent accumulation
    optimizer.zero_grad()
    
    # b. Forward Pass: Compute prediction (ŷ)
    y_hat = model(X)
    
    # c. Compute Loss (L)
    loss = criterion(y_hat, Y)
    
    # d. Backward Pass: Autograd computes gradients using the Chain Rule
    loss.backward()
    
    # e. Optimization: Update weights using the formula: w = w - lr * grad
    optimizer.step()
