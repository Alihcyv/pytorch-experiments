X = torch.tensor([1.0], requires_grad=True)
Y = torch.tensor([0.0])

model = nn.Sequential(
    nn.Linear(1, 1),
    nn.ReLU(),
    nn.Linear(1, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

optimizer.zero_grad()
y_hat = model(X)
loss = criterion(y_hat, Y)
loss.backward()
optimizer.step()
print(loss.item())
