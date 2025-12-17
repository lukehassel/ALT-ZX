# Hyperparameters
from ZXNet.main import ZXNet


LR = 0.001
EPOCHS = 100
UNCERTAINTY_THRESH = -5.0 # Empirical value, see Fig 4b for range
JACCARD_THRESH = 0.8
    
if __name__ == "__main__":

    model = ZXNet(num_node_features=6) # Based on 6 features extracted above
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        
        for batch_g1, batch_g2, labels in train_loader:
            optimizer.zero_grad()
            
            # Forward Pass
            logits = model(batch_g1, batch_g2)
            
            # Custom Jaccard Loss
            loss = model.zxnet_loss(logits, labels, batch_g1, batch_g2, 
                            uncertainty_threshold=UNCERTAINTY_THRESH,
                            jaccard_threshold=JACCARD_THRESH)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch}: Loss {total_loss}")