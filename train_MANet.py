import datasets
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import segmentation_models_pytorch as smp
import metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

OCTA500_set_combined = datasets.OCTA5003M_Dataset('OCTA500_3mm', target_size=(320, 320))

OCTA500_TRAIN_SIZE = 160 # suggested from literature to have 160, 20, 20 train, test, validation split
OCTA500_TEST_SIZE = 20
OCTA500_VAL_SIZE = 20

train_octa500, test_octa500, val_octa500 = data.random_split(
    OCTA500_set_combined, [OCTA500_TRAIN_SIZE, OCTA500_TEST_SIZE, OCTA500_VAL_SIZE], 
    generator=torch.Generator().manual_seed(42))

train_loader = data.DataLoader(train_octa500, batch_size=8, shuffle=True)
val_loader = data.DataLoader(val_octa500, batch_size=8, shuffle=False)
test_loader = data.DataLoader(test_octa500, batch_size=8, shuffle=False)

manet = smp.MAnet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
).to(device)          

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(manet.parameters(), lr=1e-4, weight_decay=1e-5)

NUM_EPOCHS = 100

for epoch in range(NUM_EPOCHS ):
    train_loss = 0.0
    
    # training loop
    manet.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        # Clear gradients from the previous step
        optimizer.zero_grad() 
        
        outputs = manet(inputs) # (B, 1, H, W)
        
        # Calculate loss
        loss = criterion(outputs.squeeze(1), targets.squeeze(1).float())
        # Backward pass: compute gradient of the loss with respect to parameters
        loss.backward()
        # Update weights based on gradients
        optimizer.step() 

        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Validation
    manet.eval()
    val_loss = 0.0
    val_dice = 0.0
    val_jaccard = 0.0
    val_balanced_acc = 0.0
    val_auc = 0.0

    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = manet(inputs)
            loss = criterion(outputs.squeeze(1), targets.squeeze(1).float())
            val_loss += loss.item()

            # Dice score metric
            probs  = torch.sigmoid(outputs).squeeze(1)
            preds = (probs > 0.5).long()
            targets_squeezed = targets.squeeze(1).long()

            val_dice += metrics.dice_score(preds, targets_squeezed)
            val_jaccard += metrics.jaccard_index(preds, targets_squeezed)
            val_balanced_acc += metrics.balanced_acc(preds, targets_squeezed)
            val_auc += metrics.auc(probs, targets_squeezed)

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_jaccard /= len(val_loader)
        val_balanced_acc /= len(val_loader)
        val_auc /= len(val_loader)

    print(f""" 
    Epoch: {epoch + 1} of {NUM_EPOCHS}
    Train loss: {train_loss}
    Validation loss: {val_loss}
    Dice score: {val_dice}
    Jaccard: {val_jaccard}
    Balanced Accuracy: {val_balanced_acc}
    AUC: {val_auc}
    """)

# test loop
test_loss = 0.0
test_dice = 0.0
test_jaccard= 0.0
test_balanced_acc = 0.0
test_auc = 0.0

manet.eval()
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = manet(inputs)
        loss = criterion(outputs.squeeze(1), targets.squeeze(1).float())
        test_loss += loss.item()

        probs = torch.sigmoid(outputs.squeeze(1))
        preds = (probs > 0.5).long()
        targets_squeezed = targets.squeeze(1).long()

        test_dice += metrics.dice_score(preds, targets_squeezed)
        test_jaccard += metrics.jaccard_index(preds, targets_squeezed)
        test_balanced_acc += metrics.balanced_acc(preds, targets_squeezed)
        test_auc += metrics.auc(probs, targets_squeezed)


    test_loss /= len(test_loader)
    test_dice /= len(test_loader)
    test_jaccard /= len(test_loader)
    test_balanced_acc /= len(test_loader)
    test_auc /= len(test_loader)

print(f"""
    Epoch: {epoch + 1} of {NUM_EPOCHS}
    Test loss: {test_loss}
    Test Dice score: {test_dice}
    Test Jaccard: {test_jaccard}
    Test Balanced Accuracy: {test_balanced_acc}
    Test AUC: {test_auc}
    """)

torch.save(manet.state_dict(), "MANet_res34.pth")
print(f"Model saved to MANet_res34.pth")




