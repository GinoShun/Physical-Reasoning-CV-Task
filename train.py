import torch 
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import dataset 
from backbones.network import CNN
from backbones.network import SElayer
import torch.nn.functional as F
from utils.options import parse_args
from utils.average import AverageMeter
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler

train_loss_average = AverageMeter()
train_losses = []
train_counter = []
steps = 0  # Initialize steps

# 图像文件和标签的路径
image_dir = '/Users/lilywu/Desktop/CV Final Project/COMP90086_2024_Project_train/train'
csv_file = '/Users/lilywu/Desktop/CV Final Project/COMP90086_2024_Project_train/train.csv'

# Function to load data and create loaders
def load_data(args):
    # 创建训练集和验证集
    train_dataset = dataset.StackDataset(csv_file=csv_file, image_dir=image_dir, img_size=(256), stratify_column = 'stable_height', train=True)
    val_dataset = dataset.StackDataset(csv_file=csv_file, image_dir=image_dir, img_size=(256), stratify_column = 'stable_height', train=False)

    # 使用DataLoader加载数据
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    loaders = {
        'train': train_loader,
        'val': val_loader
    }

    return loaders

# Main training loop
def train_loop(args):
    # Load data
    loaders = load_data(args)
    
    # Initialize model
    model = CNN()

    # Move model to appropriate device (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()  # Use MSELoss for regression
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    # Epoch loop
    for epoch in range(args.n_epochs):
        train(epoch, model, loaders, args, criterion, optimizer, scheduler, device)
        validate(epoch, model, loaders, criterion, device)
        scheduler.step()

# Training function
def train(epoch, model, loaders, args, criterion, optimizer, scheduler, device):
    model.train()  # Set model to training mode
    running_loss = 0.0

    pbar = tqdm(loaders['train'], total=len(loaders['train']))
    pbar.set_description(f"Training! Epoch {epoch} ")

    for idx, (inputs, targets) in enumerate(loaders['train']):
        # Move inputs and targets to the device
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)  # outputs shape is (batch_size, 1)
        loss = criterion(outputs, targets.view(-1, 1))  # Match target shape with output shape

        # Backward pass
        loss.backward()

        # Optimize
        optimizer.step()

        # Update running loss
        running_loss += loss.item()

        # Progress logging
        if idx % 10 == 0 and idx != 0:
            print(f'Epoch [{epoch+1}], Step [{idx}/{len(loaders["train"])}], Loss: {loss.item():.4f}')
        pbar.update(1)

    print(f'Epoch [{epoch+1}], Average Loss: {running_loss / len(loaders["train"]):.4f}')
    
# Validation function
def validate(epoch, model, loaders, criterion, device):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0

    with torch.no_grad():
        for inputs, targets in loaders['val']:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1, 1))  # Match target shape with output shape
            val_loss += loss.item()

    avg_val_loss = val_loss / len(loaders['val'])
    print(f'Validation Loss after Epoch {epoch+1}: {avg_val_loss:.4f}')
if __name__ == '__main__':
    args = parse_args()
    train_loop(args)
