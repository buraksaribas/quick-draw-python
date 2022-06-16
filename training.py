import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from training_utils import create_model, train, eval_epoch

# GPU or CPU
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print(f'\nTraining in {device}\n')

# Create model and move it to the device
model = create_model()

model.to(device)
print(summary(model))

# Create the dataset and split it in (train, val, test)
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.9720, 0.9720, 0.9720), 
                         (0.1559, 0.1559, 0.1559)) 
    # Normalize with the mean and std of the whole dataset
])

dataset = ImageFolder(root='images', transform=transform)

n = len(dataset)  
train_dataset, validation_dataset, test_dataset = random_split(
    dataset, 
    lengths=[int(0.8 * n), int(0.1 * n), int(0.1 * n)], 
    generator=torch.Generator().manual_seed(42)
)


epochs = 50
batch_size = 32
lr = 0.001
num_workers=4
train_loader = DataLoader(
    train_dataset, 
    batch_size, 
    num_workers
)
validation_loader = DataLoader(
    validation_dataset, 
    batch_size, 
    num_workers
)
test_loader = DataLoader(
    test_dataset, 
    batch_size, 
    num_workers
)
writer = SummaryWriter(log_dir='runs/')

# Train the model, open TensorBoard to see the progress
train(
    model, 
    train_loader, 
    validation_loader, 
    device, 
    lr, 
    epochs, 
    writer=writer, 
    checkpoint_path='models/checkpoint.pt'
)
# Save the model
torch.save(model, 'models/model.pt')
# Add some metrics to evaluate different models and hyperparameters

_, train_acc = eval_epoch(model, train_loader, device)
_, val_acc = eval_epoch(model, validation_loader, device)
_, test_acc = eval_epoch(model, test_loader, device)

writer.add_hparams(
    hparam_dict={
        'lr': lr, 
        'batch_size': batch_size, 
        'epochs': epochs
    },
    metric_dict={
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'test_accuracy': test_acc
    },
    run_name='hparams'
)