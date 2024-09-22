import torch


class Train():
    def __init__(self, model, optimizer, epochs, dataloader, criterion, test_obj, device='cpu', print_cost=True, use_features=False, use_weights=False):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device
        self.print_cost = print_cost
        self.test = test_obj
        self.use_features = use_features
        self.use_weights = use_weights

    def train(self):
        self.model.to(self.device)
        for epoch in range(self.epochs):
            self.model.train()  # Set model to training mode
            total_loss = 0
            for batch in self.dataloader:
                batch = [x.to(self.device) for x in batch]

                if self.use_features:
                    users, items, labels, weights, features,_,_,_ = batch
                    features = features.float()
                    outputs = self.model(features)
                elif self.use_weights:
                    users, items, labels, weights,_,_,_ = batch
                    weights = weights.float()
                    outputs = self.model(users, items, weights)
                else:
                    users, items, labels,_,_,_,_ = batch
                    users, items = users.long(), items.long()
                    outputs = self.model(users, items)

                labels = labels.float()
                outputs = outputs.view(-1)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)

            # Testing phase
            self.model.eval()  # Set model to evaluation mode
            total_test_loss = 0
            with torch.no_grad():
                for batch in self.test:
                    batch = [x.to(self.device) for x in batch]

                    if self.use_features:
                        users, items, labels, weights, features, _,_,_ = batch
                        features = features.float()
                        outputs = self.model(features)
                    elif self.use_weights:
                        users, items, labels, weights, _,_,_ = batch
                        weights = weights.float()
                        outputs = self.model(users, items, weights)
                    else:
                        users, items, labels, _, _ ,_,_= batch
                        users, items = users.long(), items.long()
                        outputs = self.model(users, items)

                    labels = labels.float()
                    outputs = outputs.view(-1)
                    loss = self.criterion(outputs, labels)
                    total_test_loss += loss.item()

            avg_test_loss = total_test_loss / len(self.test)

            if self.print_cost:
                print(f'Epoch {epoch + 1}: Train Loss: {avg_loss:.4f}, Test Loss: {avg_test_loss:.4f}')

        if self.print_cost:
            print('Learning finished')