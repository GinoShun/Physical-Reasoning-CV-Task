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

def loss_simple(outputs, targets):
    # Unpack outputs and targets
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

def loss(outputs, targets, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # out_main, out_shapeset, out_type, out_total_height, out_num_unstable, out_instability, out_cam_angle = outputs
    out_main, out_shapeset, out_type, out_total_height, out_instability, out_cam_angle = outputs
    target_main = targets['stable_height']
    target_shapeset = targets['shapeset']
    target_type = targets['type']
    target_total_height = targets['total_height']
    # target_num_unstable = targets['num_unstable']
    target_instability = targets['instability_type']
    target_cam_angle = targets['cam_angle']

    # Define additional weight for type=2 samples
    type2_weight = 1.5  # TODO: adjust

    # Compute a weight tensor where type=2 samples get a higher weight
    main_task_weight = torch.ones_like(target_type, dtype=torch.float32)
    main_task_weight[target_type == 2] = type2_weight

    # Ensure main_task_weight is on the same device as model outputs
    main_task_weight = main_task_weight.to(device)

    # Compute the CrossEntropyLoss per sample
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    per_sample_loss = loss_fn(out_main, target_main.long())

    # Multiply per-sample losses by the weights
    weighted_loss = main_task_weight * per_sample_loss

    # Take the mean of the weighted losses
    loss_main = weighted_loss.mean()

    # # classification main task
    # loss_main = nn.CrossEntropyLoss()(out_main, target_main.long())

    # regression main task
    # loss_main = nn.MSELoss()(out_main.squeeze(), target_main.float())

    # supplementary tasks
    loss_shapeset = nn.CrossEntropyLoss()(out_shapeset, target_shapeset.long())
    loss_type = nn.CrossEntropyLoss()(out_type, target_type.long())
    loss_total_height = nn.CrossEntropyLoss()(out_total_height, target_total_height.long())
    # loss_num_unstable = nn.CrossEntropyLoss()(out_num_unstable, target_num_unstable.long())
    loss_instability = nn.CrossEntropyLoss()(out_instability, target_instability.long())
    loss_cam_angle = nn.CrossEntropyLoss()(out_cam_angle, target_cam_angle.long())

    # set min and max values for log sigmas
    min_log_sigma = -3.0 
    max_log_sigma = 3.0

    # Clamping log_sigma to prevent extreme values
    log_sigma_main = torch.clamp(model.log_sigma_main, min=min_log_sigma, max=max_log_sigma)
    log_sigma_shapeset = torch.clamp(model.log_sigma_shapeset, min=min_log_sigma, max=max_log_sigma)
    log_sigma_type = torch.clamp(model.log_sigma_type, min=min_log_sigma, max=max_log_sigma)
    log_sigma_total_height = torch.clamp(model.log_sigma_total_height, min=min_log_sigma, max=max_log_sigma)
    log_sigma_instability = torch.clamp(model.log_sigma_instability, min=min_log_sigma, max=max_log_sigma)
    log_sigma_cam_angle = torch.clamp(model.log_sigma_cam_angle, min=min_log_sigma, max=max_log_sigma)
    # log_sigma_num_unstable = torch.clamp(model.log_sigma_num_unstable, min=min_log_sigma, max=max_log_sigma)

    # learnable log sig
    sigma_main = torch.exp(log_sigma_main)
    sigma_shapeset = torch.exp(log_sigma_shapeset)
    sigma_type = torch.exp(log_sigma_type)
    sigma_total_height = torch.exp(log_sigma_total_height)
    sigma_instability = torch.exp(log_sigma_instability)
    sigma_cam_angle = torch.exp(log_sigma_cam_angle)
    # sigma_num_unstable = torch.exp(log_sigma_num_unstable)

    # Adding constraint: num_unstable = total_height - stable_height
    predicted_total_height = torch.argmax(out_total_height, dim=1)
    predicted_stable_height = torch.argmax(out_main, dim=1)
    # predicted_num_unstable = torch.argmax(out_num_unstable, dim=1)

    # num_unstable_diff = torch.abs(predicted_num_unstable - (predicted_total_height - predicted_stable_height))
    # constraint_loss = num_unstable_diff.float().mean()

    # Adding constraint: total_height > stable_height
    constraint_loss = F.relu(predicted_stable_height - predicted_total_height).float().mean()

    # punishment factor
    lambda_constraint = 1.0

    # total loss
    instability_weight = 1.2  # TODO: adjust
    beta = 0.5
    
    # Total loss
    total_loss = (1 / (2 * sigma_main ** 2)) * loss_main + beta * log_sigma_main + \
                 (1 / (2 * sigma_total_height ** 2)) * loss_total_height + beta * log_sigma_total_height + \
                 (1 / (2 * sigma_shapeset ** 2)) * loss_shapeset + beta * log_sigma_shapeset + \
                 (1 / (2 * sigma_type ** 2)) * loss_type + beta * log_sigma_type + \
                 instability_weight  * (1 / (2 * sigma_instability ** 2)) * loss_instability + beta * log_sigma_instability + \
                 (1 / (2 * sigma_cam_angle ** 2)) * loss_cam_angle + beta * log_sigma_cam_angle + \
                 lambda_constraint * constraint_loss

    # Ensure total_loss is not negative
    total_loss = F.relu(total_loss)
    
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
    # criterion = loss
    criterion = loss_simple

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # # different lr for log sigmas
    # optimizer = torch.optim.AdamW([
    #     {'params': [param for name, param in model.named_parameters() if 'log_sigma' not in name]},
    #     {'params': [param for name, param in model.named_parameters() if 'log_sigma' in name], 'lr': args.lr * 0.1}
    # ], lr=args.lr, weight_decay=1e-4)


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

        # # Debugging: Print minimum and maximum values of target
        # print("Labels range:", targets['stable_height'].min(), targets['stable_height'].max())
        
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
