from pickletools import optimize
import torch
import torch.utils.data
from tqdm import tqdm
from src.metric import log_rmse

def train(
    train_features, 
    train_labels, 
    test_features, 
    test_labels, 
    model,
    criterion,
    learning_rate,
    weight_decay,
    epochs,
    batch_size
    ):
        train_losses = []
        test_losses = []

        train = torch.utils.data.TensorDataset(train_features, train_labels)
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr = learning_rate,
                                     weight_decay=weight_decay)
        
        for _ in tqdm(range(1, epochs+1)):
            for X,y in train_loader:
                l = criterion(model(X),y)

                optimizer.zero_grad()
                l.backward()
                optimizer.step()
            
            train_losses.append(log_rmse(model(train_features), train_labels))
            if test_labels is not None:
                test_losses.append(log_rmse(model(test_features), test_labels))

        if not test_losses:
            return train_losses[-1], None    

        return train_losses[-1], test_losses[-1]
        
