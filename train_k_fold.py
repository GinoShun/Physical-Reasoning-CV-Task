import torch 
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
import dataset 
import torch.nn.functional as F
from utils.options import parse_args
from utils.average import AverageMeter
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import os, sys
import importlib
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold

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
def load_data(args):
    # load whole
    dataset_instance = dataset.StackDataset(
        csv_file=args.metadata_path, 
        image_dir=args.picture_path, 
        img_size=224, 
        stable_height='stable_height', 
        train=True, 
        doSplit=False  # do not split the dataset
    )
    return dataset_instance

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

def loss_simple(outputs, targets, model):
    # Unpack outputs and targets
    out_main, out_shapeset, out_type, out_total_height, out_instability, out_cam_angle = outputs
    # out_main, out_shapeset, out_type, out_total_height, out_num_unstable, out_instability, out_cam_angle = outputs
    
    target_main = targets['stable_height']
    target_shapeset = targets['shapeset']
    target_type = targets['type']
    target_total_height = targets['total_height']
    target_instability = targets['instability_type']
    target_cam_angle = targets['cam_angle']
    # target_num_unstable = targets['num_unstable']

    # Main Task Loss with Focal Loss
    loss_main = FocalLoss(gamma=2)(out_main, target_main.long())

    # Supplementary Tasks Losses
    loss_shapeset = nn.CrossEntropyLoss()(out_shapeset, target_shapeset.long())
    loss_type = nn.CrossEntropyLoss()(out_type, target_type.long())
    loss_total_height = nn.CrossEntropyLoss()(out_total_height, target_total_height.long())
    loss_instability = nn.CrossEntropyLoss()(out_instability, target_instability.long())
    loss_cam_angle = nn.CrossEntropyLoss()(out_cam_angle, target_cam_angle.long())
    # loss_num_unstable = nn.CrossEntropyLoss()(out_num_unstable, target_num_unstable.long())

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

    # # Convert log sigma to sigma (exponential of log sigma)
    # sigma_main = torch.exp(model.log_sigma_main)
    # sigma_shapeset = torch.exp(model.log_sigma_shapeset)
    # sigma_type = torch.exp(model.log_sigma_type)
    # sigma_total_height = torch.exp(model.log_sigma_total_height)
    # sigma_instability = torch.exp(model.log_sigma_instability)
    # sigma_cam_angle = torch.exp(model.log_sigma_cam_angle)
    # # sigma_num_unstable = torch.exp(model.log_sigma_num_unstable)

    # # Total loss with uncertainty weighting
    # total_loss = (1 / (2 * sigma_main ** 2)) * loss_main + torch.log(sigma_main) + \
    #              (1 / (2 * sigma_shapeset ** 2)) * loss_shapeset + torch.log(sigma_shapeset) + \
    #              (1 / (2 * sigma_type ** 2)) * loss_type + torch.log(sigma_type) + \
    #              (1 / (2 * sigma_total_height ** 2)) * loss_total_height + torch.log(sigma_total_height) + \
    #              (1 / (2 * sigma_instability ** 2)) * loss_instability + torch.log(sigma_instability) + \
    #              (1 / (2 * sigma_cam_angle ** 2)) * loss_cam_angle + torch.log(sigma_cam_angle)

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

    # linear scale warmup
    # warmup_scheduler = lr_scheduler.LambdaLR(
    #     optimizer,
    #     lr_lambda=lambda epoch: epoch / warmup_iters if epoch < warmup_iters else 1,
    #     last_epoch=last_epoch
    # )

    # Define CosineAnnealingLR for after warmup
    cosine_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_T_max,
        last_epoch=last_epoch
    )
    
    return warmup_scheduler, cosine_scheduler

# Main training loop
def train_loop(args):
    use_new_optimizer = True
    use_new_scheduler = True

    # Load data
    dataset_instance = load_data(args)

    # Import network
    network_module = import_network(args.network_file)
    CNN = getattr(network_module, 'CNN')  # Dynamically get CNN class

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # k-fold cross-validation
    n_fold = args.n_fold
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset_instance), 1):
        print(f"Training fold {fold}/{n_fold}")

        writer = SummaryWriter(log_dir=os.path.join(args.task_name, f'tensorboard_logs/fold_{fold}'))

        # Create fold-specific data loaders
        train_subset = Subset(dataset_instance, train_idx)
        val_subset = Subset(dataset_instance, val_idx)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size_test, shuffle=False, num_workers=args.num_workers)

        loaders = {'train': train_loader, 'val': val_loader}

        model = CNN().to(device)
        criterion = loss_simple
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        warmup_iters = int(0.1 * args.n_epochs)
        cosine_T_max = 24
        warmup_scheduler, cosine_scheduler = get_scheduler_with_warmup(
            optimizer, warmup_iters, cosine_T_max, last_epoch=-1
        )

        best_val_accuracy = 0.0
        best_val_loss = float('inf')

        # Epoch loop
        for epoch in range(args.n_epochs):
            scheduler = warmup_scheduler if epoch < warmup_iters else cosine_scheduler

            train(epoch, model, loaders, args, criterion, optimizer, scheduler, device, writer, fold)
            avg_val_loss, val_accuracy = validate(epoch, model, loaders, criterion, device, writer, fold)
            scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar(f'Fold_{fold}/Learning Rate', current_lr, epoch + 1)

            if val_accuracy > best_val_accuracy or \
                (val_accuracy == best_val_accuracy and avg_val_loss < best_val_loss):
                best_val_accuracy = val_accuracy
                best_val_loss = avg_val_loss
                best_model_save_path = os.path.join(args.task_name, f"best_model_fold_{fold}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }, best_model_save_path)
                print(f"Best model for fold {fold} saved at epoch {epoch+1} with accuracy {val_accuracy:.4f} and loss {avg_val_loss:.4f}")

        writer.close()

# Training function
def train(epoch, model, loaders, args, criterion, optimizer, scheduler, device, writer, fold):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loaders['train'], total=len(loaders['train']))
    pbar.set_description(f"Training Fold {fold} | Epoch {epoch + 1}")

    for idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets, model)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item()

        # Calculate accuracy
        out_main = outputs[0]
        _, predicted = torch.max(out_main, 1)
        target_height = targets['stable_height'].long()
        correct += (predicted == target_height).sum().item()
        total += target_height.size(0)

        pbar.update(1)

    # Calculate average loss and accuracy
    total_loss = running_loss / len(loaders['train'])
    accuracy = 100 * correct / total

    # record by fold
    writer.add_scalar(f'Fold_{fold}/Training Loss', total_loss, epoch + 1)
    writer.add_scalar(f'Fold_{fold}/Training Accuracy', accuracy, epoch + 1)

    print(f'\n[Fold {fold}] Epoch [{epoch+1}] - Training Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
# Validation function
def validate(epoch, model, loaders, criterion, device, writer, fold):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loaders['val']:
            inputs = inputs.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets, model)
            val_loss += loss.item()

            # Calculate accuracy
            out_main = outputs[0]
            _, predicted = torch.max(out_main, 1)
            target_height = targets['stable_height'].long()
            correct += (predicted == target_height).sum().item()
            total += target_height.size(0)

    avg_val_loss = val_loss / len(loaders['val'])
    accuracy = 100 * correct / total

    # record by fold
    writer.add_scalar(f'Fold_{fold}/Validation Loss', avg_val_loss, epoch + 1)
    writer.add_scalar(f'Fold_{fold}/Validation Accuracy', accuracy, epoch + 1)

    print(f'[Fold {fold}] Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')

    return avg_val_loss, accuracy


if __name__ == '__main__':
    args = parse_args()
    train_loop(args)
