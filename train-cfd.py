from dataclasses import dataclass
import pickle
import numpy as np
import torch 
from torch.utils.data import TensorDataset
from diffusers import UNet2DModel
from PIL import Image
from diffusers import DDPMScheduler
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline
import math
import os
from tqdm.auto import tqdm
from torchvision import transforms

torch.manual_seed(0)


@dataclass
class TrainingConfig:
    '''
    Class for training parameters
    '''
    
    image_size = 128  # the generated image resolution
    train_batch_size = 8
    eval_batch_size = 8  # how many images to sample during evaluation
    num_epochs = 200
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "/home/maddie/Documents/underwater/DeepCFD/output"  # the model name locally and on the HF Hub

    push_to_hub = True  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0


config = TrainingConfig()

# Load the pickle files using the exact file paths
x_data_path = "/home/maddie/Documents/underwater/DeepCFD/dataX.pkl"
y_data_path = "/home/maddie/Documents/underwater/DeepCFD/dataY.pkl"

with open(x_data_path, "rb") as f:
    x = pickle.load(f)

with open(y_data_path, "rb") as f:
    y = pickle.load(f)

    # Preprocess the data

# turn the input in a pytorch tensor
x = torch.FloatTensor(x)
y = torch.FloatTensor(y)

# Shuffle data 
indices = list(range(len(x)))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# Normalize the data [-1,1] for each channel 
def normalize_tensor(tensor):
    min_val = torch.amin(tensor, dim=(0, 2, 3), keepdim=True)  # min over all axes except channels
    max_val = torch.amax(tensor, dim=(0, 2, 3), keepdim=True)  # max over all axes except channels
    # Normalize to [-1, 1]
    tensor_normalized = 2 * ((tensor - min_val) / (max_val - min_val)) - 1
    return tensor_normalized

# normalize the input and output data
x = normalize_tensor(x)
y = normalize_tensor(y)


# Function for dividing the dataset 
def split_tensors(*tensors, ratio):
    preprocess = transforms.Compose(
        [
            transforms.Resize((128,128)),
            transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5]),
        ]
    )

    assert len(tensors) > 0
    split1, split2 = [], []
    count = len(tensors[0])
    for tensor in tensors:
        assert len(tensor) == count
        # Interpreting the last two dimensions as 128
        tensor = [preprocess(temp) for temp in tensor]
        tensor = torch.stack(tensor)
        split1.append(tensor[:int(len(tensor) * ratio)])
        split2.append(tensor[int(len(tensor) * ratio):])
    if len(tensors) == 1:
        split1, split2 = split1[0], split2[0]
    return split1, split2

# # Split the data into training and testing sets (70/30)
# train_data, test_data = split_tensors(x,y, ratio=0.7)
train_data, test_data = split_tensors(y,ratio=0.7)

# Create PyTorch datasets 
train_dataset = TensorDataset(train_data)

val_dataset = TensorDataset(test_data)
# test_x, test_y = test_data[:] # split test data into x and y

# Load data into PyTorch DataLoader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)

val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config.train_batch_size, shuffle=True)

# Create Scheduler 
noise_scheduler = DDPMScheduler(num_train_timesteps=1000) # DDPM scheduler with 1000 timesteps

# Setup NN model
model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

# Optimizer:  update the model's parameters based on the computed gradients during the training process
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

# Learning rate scheduler: adjust the learning rate of the optimizer during the training process
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

# Function for evaluation?
def evaluate(config, epoch, model, noise_scheduler, val_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    total_mse = 0
    with torch.no_grad():  # Disable gradient computation
        for batch in val_dataloader:
            clean_images = batch[0].to(device)
            noise = torch.randn(clean_images.shape).to(device)
            bs = clean_images.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=device
            ).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            noise_pred = model(noisy_images, timesteps)[0]
            mse = F.mse_loss(noise_pred, noise, reduction='sum').item()  # Calculate batch MSE
            total_mse += mse

    avg_mse = total_mse / len(val_dataloader.dataset)
    print(f"Epoch {epoch}: Average MSE = {avg_mse}")

    # Setup Method to evaluate the model

# Define the training loop function
def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    
    # Check if an output directory is specified and create it if it doesn't exist
    if config.output_dir is not None:
        os.makedirs(config.output_dir, exist_ok=True)
        # Initialize any desired loggers here, such as TensorBoard

    # Initialize a global step counter
    global_step = 0

    # GPU/CPU setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Start the training process
    for epoch in range(config.num_epochs):
        
        # Initialize a progress bar for the epoch
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}")
        
        # Iterate over the training data loader
        for _, batch in enumerate(train_dataloader):

            # Move the images from the current batch to the device
            clean_images = batch[0].to(device)

            # Generate random noise of the same shape as the images
            noise = torch.randn(clean_images.shape).to(device)

            # Get the batch size from the images shape
            bs = clean_images.shape[0]

            # Sample random timesteps for each image in the batch
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=device
            ).long()

            # Add noise to the clean images according to the noise magnitude at the sampled timesteps
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # print("noisy data:", noisy_images.shape)
            # print("timesteps:", timesteps.shape)

            # Predict the noise residual using the model
            noise_pred = model(noisy_images, timesteps).sample
            # print("noise_pred:", noise_pred.shape)
            
            # Calculate the mean squared error loss between the predicted noise and the actual noise
            loss = F.mse_loss(noise_pred, noise)

            # Perform backpropagation: zero the gradients, calculate gradients, and perform a single optimization step
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Calculate gradients through backpropagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip gradients to avoid exploding gradient problem
            optimizer.step()  # Update model parameters
            lr_scheduler.step()  # Update learning rate scheduler

            # Update the progress bar and log the loss and learning rate
            progress_bar.update(1)
            logs = {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            # Log your progress here, replacing 'accelerator.log' with your logger
            global_step += 1  # Increment the global step counter

        # After each epoch, evaluate your model and save it if necessary
        if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
            # Replace 'evaluate' with your evaluation function
            evaluate(config, epoch, model, noise_scheduler, val_dataloader)

        if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
            # Save your model parameters to the specified output directory
            torch.save(model.state_dict(), os.path.join(config.output_dir, f"model_epoch_{epoch}.pth"))
    
# Train the model 
train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)