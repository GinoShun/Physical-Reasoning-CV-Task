import torch 
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import baseline_dataset as dataset
import torch.nn.functional as F
from utils.options import parse_args
from utils.average import AverageMeter
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import os, sys
import importlib
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore")


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
def load_data(args, singleTask=True):
    train_dataset = dataset.StackDataset(csv_file=args.metadata_path, image_dir=args.picture_path, img_size=224, stable_height = 'stable_height', train=True, singleTask=singleTask)
    val_dataset = dataset.StackDataset(csv_file=args.metadata_path, image_dir=args.picture_path, img_size=224, stable_height = 'stable_height', train=False, singleTask=singleTask)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_test, shuffle=False, num_workers=args.num_workers)

    loaders = {
        'train': train_loader,
        'val': val_loader
    }

    return loaders

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()

def loss_simple(outputs, targets, singleTask=True):
    # Unpack outputs and targets
    if singleTask:
        out_main = outputs
        target_main = targets.long()
        loss_main = FocalLoss(gamma=2)(out_main, target_main.long())
        return loss_main
    else:
        out_main, out_shapeset, out_type, out_total_height, out_instability, out_cam_angle = outputs
        
        target_main = targets['stable_height']
        target_shapeset = targets['shapeset']
        target_type = targets['type']
        target_total_height = targets['total_height']
        target_instability = targets['instability_type']
        target_cam_angle = targets['cam_angle']

        # Main Task Loss with Focal Loss
        loss_main = FocalLoss(gamma=2)(out_main, target_main.long())

        # Supplementary Tasks Losses
        loss_shapeset = nn.CrossEntropyLoss()(out_shapeset, target_shapeset.long())
        loss_type = nn.CrossEntropyLoss()(out_type, target_type.long())
        loss_total_height = nn.CrossEntropyLoss()(out_total_height, target_total_height.long())
        loss_instability = nn.CrossEntropyLoss()(out_instability, target_instability.long())
        loss_cam_angle = nn.CrossEntropyLoss()(out_cam_angle, target_cam_angle.long())

        # Define fixed weights for each task
        w_main = 1.0
        w_total_height = 0.5
        w_shapeset = 0.3
        w_type = 0.4
        w_instability = 0.5
        w_cam_angle = 0.2

        # human annotated weights! zhubao power!
        # total_loss = loss_main + 0.5 * (0.3 * loss_total_height + 
        # 0.2 * (loss_shapeset + loss_instability) + 
        # 0.1 * (loss_type + loss_cam_angle))

        # Total loss without uncertainty weighting
        total_loss = w_main * loss_main + \
                    w_total_height * loss_total_height + \
                    w_shapeset * loss_shapeset + \
                    w_type * loss_type + \
                    w_instability * loss_instability + \
                    w_cam_angle * loss_cam_angle

        return total_loss

def get_scheduler_with_warmup(optimizer, warmup_iters, cosine_T_max, last_epoch):
    import math

    # Define LambdaLR for warmup phase
    def cosine_warmup_lambda(epoch, warmup_iters):
        if epoch < warmup_iters:
            return 0.5 * (1 + math.cos(math.pi * epoch / warmup_iters))
        return 1.0

    warmup_scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: cosine_warmup_lambda(epoch, warmup_iters),
        last_epoch=last_epoch
    )

    # Define CosineAnnealingLR for after warmup
    cosine_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_T_max,
        last_epoch=last_epoch
    )
    
    return warmup_scheduler, cosine_scheduler

# Main training loop
def train_loop(args):
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(args.task_name, 'tensorboard_logs'))

    singleTask = True

    # Load data
    loaders = load_data(args, singleTask)
    
    # import network
    network_module = import_network(args.network_file)
    # Access the CNN class from the imported module
    CNN = getattr(network_module, 'CNN')  # Dynamically get CNN class

    # Initialize model
    model = CNN()

    # Move model to appropriate device (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    start_epoch = 0

    # Loss and optimizer
    criterion = loss_simple
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("start lr:", args.lr)

    # Warmup Scheduler
    warmup_iters = int(0.1 * args.n_epochs)  # 10% of total epochs
    cosine_T_max = 24

    # Get both schedulers
    warmup_scheduler, cosine_scheduler = get_scheduler_with_warmup(
        optimizer, warmup_iters, cosine_T_max, last_epoch=start_epoch - 1
    )

    # Epoch loop
    for epoch in range(start_epoch, args.n_epochs):
        # warmup
        if args.warmup == "True":
            if epoch < warmup_iters:
                print("warmup phase")
                scheduler = warmup_scheduler
            else:
                scheduler = cosine_scheduler
                # print("Finished warmup phase.")
        else:
            scheduler = cosine_scheduler
            # print("No warmup phase, using cosine annealing scheduler.")

        train(epoch, model, loaders, args, criterion, optimizer, scheduler, device, writer, singleTask)
        validate(epoch, model, loaders, criterion, device, writer, singleTask)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current Learning Rate after Epoch {epoch+1}: {current_lr}')

        # Log learning rate
        writer.add_scalar('Learning Rate', current_lr, epoch + 1)

        # Save the model after each epoch to the task_name directory
        model_save_path = os.path.join(args.task_name, f"model_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict()
            # 'optimizer_state_dict': optimizer.state_dict(),
            # 'scheduler_state_dict': scheduler.state_dict()
        }, model_save_path)
        # print(f"Model saved at {model_save_path}")

    writer.close()  # Close the writer when training finishes

# Training function
def train(epoch, model, loaders, args, criterion, optimizer, scheduler, device, writer, singleTask=True):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    # epsilon = 0.007

    if not os.path.exists(args.task_name):
        os.makedirs(args.task_name)
        print(f"Directory '{args.task_name}' created for task outputs.")

    pbar = tqdm(loaders['train'], total=len(loaders['train']))
    pbar.set_description(f"Training! Epoch {epoch} ")

    for idx, (inputs, targets) in enumerate(pbar):
        # print(f"Batch {idx} loaded.")

        inputs = inputs.to(device)

        if not singleTask:
            targets = {k: v.to(device) for k, v in targets.items()}
        else:
            targets = targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)  # outputs shape is (batch_size, 1)
        loss = criterion(outputs, targets, singleTask)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item()

        # Calculate accuracy
        if singleTask:
            out_main = outputs
        else:
            out_main = outputs[0]  # label

        # For classification
        _, predicted = torch.max(out_main, 1)
        if not singleTask:
            target_height = targets['stable_height'].long()
        else:
            target_height = targets.long()

        correct += (predicted == target_height).sum().item()
        total += target_height.size(0)

        pbar.update(1)

    # Calculate average loss and accuracy
    total_loss = running_loss / len(loaders['train']) 
    accuracy = 100 * correct / total

    # Log to TensorBoard
    writer.add_scalar('Training Loss', total_loss, epoch + 1)
    writer.add_scalar('Training Accuracy', accuracy, epoch + 1)

    print(f'\nEpoch [{epoch+1}], Average Loss: {running_loss / len(loaders["train"]):.4f}, Accuracy: {accuracy:.2f}%')
    
# Validation function
def validate(epoch, model, loaders, criterion, device, writer, singleTask=True):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loaders['val']:
            inputs = inputs.to(device)

            if not singleTask:
                targets = {k: v.to(device) for k, v in targets.items()}
            else:
                targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets, singleTask)
            val_loss += loss.item()

            # Calculate accuracy
            if singleTask:
                out_main = outputs
                # For classification
                _, predicted = torch.max(out_main, 1)
                target_height = targets.long()

            else:
                out_main = outputs[0]
                # For classification
                _, predicted = torch.max(out_main, 1)
                target_height = targets['stable_height'].long()

            correct += (predicted == target_height).sum().item()
            total += target_height.size(0)

    avg_val_loss = val_loss / len(loaders['val'])
    accuracy = 100 * correct / total

    # Log to TensorBoard
    writer.add_scalar('Validation Loss', avg_val_loss, epoch + 1)
    writer.add_scalar('Validation Accuracy', accuracy, epoch + 1)

    print(f'Validation Loss after Epoch {epoch+1}: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')


if __name__ == '__main__':
    args = parse_args()
    train_loop(args)
