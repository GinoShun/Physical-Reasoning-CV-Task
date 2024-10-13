import torch 
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import dataset 
import torch.nn.functional as F
from utils.options import parse_args
from utils.average import AverageMeter
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import os, sys
import importlib

train_loss_average = AverageMeter()
train_losses = []
train_counter = []
steps = 0  # Initialize steps

def import_network(network_file):
    """Dynamically import the network module using the provided filename."""
    try:
        # Dynamically import the module using importlib
        network_module = importlib.import_module(f'backbones.{network_file}')
        return network_module
    except ModuleNotFoundError:
        raise ImportError(f"Could not find network file: backbones/{network_file}.py")

# Function to load data and create loaders
def load_data(args):
    train_dataset = dataset.StackDataset(csv_file=args.metadata_path, image_dir=args.picture_path, img_size=224, stable_height = 'stable_height', train=True)
    val_dataset = dataset.StackDataset(csv_file=args.metadata_path, image_dir=args.picture_path, img_size=224, stable_height = 'stable_height', train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_test, shuffle=False, num_workers=args.num_workers)

    loaders = {
        'train': train_loader,
        'val': val_loader
    }

    return loaders

def loss(outputs, targets):
    out_main, out_shapeset, out_type, out_instability, out_cam_angle = outputs
    target_main = targets['stable_height']
    target_shapeset = targets['shapeset']
    target_type = targets['type']
    target_instability = targets['instability_type']
    target_cam_angle = targets['cam_angle']

    # regression main task
    # loss_main = nn.MSELoss()(out_main.squeeze(), target_main.float())

    # classification main task
    loss_main = nn.CrossEntropyLoss()(out_main, target_main.long())

    # supplementary tasks
    loss_shapeset = nn.CrossEntropyLoss()(out_shapeset, target_shapeset.long())
    loss_type = nn.CrossEntropyLoss()(out_type, target_type.long())
    loss_instability = nn.CrossEntropyLoss()(out_instability, target_instability.long())
    loss_cam_angle = nn.CrossEntropyLoss()(out_cam_angle, target_cam_angle.long())

    # total loss
    total_loss = loss_main + 0.1 * (loss_shapeset + loss_type + loss_instability + loss_cam_angle)
    return total_loss


# Main training loop
def train_loop(args):
    # Load data
    loaders = load_data(args)
    
    # import network
    network_module = import_network(args.network_file)
    # Access the CNN class from the imported module
    CNN = getattr(network_module, 'CNN')  # Dynamically get CNN class

    # Initialize model
    model = CNN()

    # Move model to appropriate device (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss and optimizer
    criterion = loss
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)


    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=20)

    # Epoch loop
    for epoch in range(args.n_epochs):
        train(epoch, model, loaders, args, criterion, optimizer, scheduler, device)
        validate(epoch, model, loaders, criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current Learning Rate after Epoch {epoch+1}: {current_lr}')

        # Save the model after each epoch to the task_name directory
        model_save_path = os.path.join(args.task_name, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_save_path)
        # print(f"Model saved at {model_save_path}")

# Training function
def train(epoch, model, loaders, args, criterion, optimizer, scheduler, device):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    if not os.path.exists(args.task_name):
        os.makedirs(args.task_name)
        print(f"Directory '{args.task_name}' created for task outputs.")

    pbar = tqdm(loaders['train'], total=len(loaders['train']))
    pbar.set_description(f"Training! Epoch {epoch} ")

    for idx, (inputs, targets) in enumerate(pbar):
        # print(f"Batch {idx} loaded.")

        # Move inputs and targets to the device
        inputs = inputs.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        # targets = targets.long()

        # # Debugging: Print minimum and maximum values of targets
        # print(f"Min target: {targets.min()}, Max target: {targets.max()}")
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)  # outputs shape is (batch_size, 1)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Optimize
        optimizer.step()

        # Update running loss
        running_loss += loss.item()

        # Calculate accuracy
        out_main = outputs[0]  # label

        # For classification
        _, predicted = torch.max(out_main, 1)
        target_height = targets['stable_height'].long()
        correct += (predicted == target_height).sum().item()
        total += target_height.size(0)

        # # For regression
        # predicted_height = torch.round(out_main.squeeze())
        # predicted_height = torch.clamp(predicted_height, min=1, max=6)
        # predicted_height = predicted_height.long()

        # target_height = targets['stable_height'].long()
        # correct += (predicted_height == target_height).sum().item()
        # total += target_height.size(0)

        # Progress logging
        # if idx % 10 == 0 and idx != 0:
            # print(f'Epoch [{epoch+1}], Step [{idx}/{len(loaders["train"])}], Loss: {loss.item():.4f}')
        pbar.update(1)

    accuracy = 100 * correct / total
    print(f'\nEpoch [{epoch+1}], Average Loss: {running_loss / len(loaders["train"]):.4f}, Accuracy: {accuracy:.2f}%')
    
# Validation function
def validate(epoch, model, loaders, criterion, device):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loaders['val']:
            inputs = inputs.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            # targets = targets.long()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            # Calculate accuracy
            out_main = outputs[0]

            # For classification
            _, predicted = torch.max(out_main, 1)
            target_height = targets['stable_height'].long()

            correct += (predicted == target_height).sum().item()
            total += target_height.size(0)

            # # For regression
            # predicted_height = torch.round(out_main.squeeze())
            # predicted_height = torch.clamp(predicted_height, min=1, max=6)
            # predicted_height = predicted_height.long()

            # target_height = targets['stable_height'].long()
            # correct += (predicted_height == target_height).sum().item()
            # total += target_height.size(0)

    avg_val_loss = val_loss / len(loaders['val'])
    accuracy = 100 * correct / total
    print(f'Validation Loss after Epoch {epoch+1}: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')


if __name__ == '__main__':
    args = parse_args()
    train_loop(args)
