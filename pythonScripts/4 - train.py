import os
import numpy as np
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

# Hyperparameters
LEARNING_RATE = 0.00045127039469197727
WEIGHT_DECAY = 0.000571817601139671
IMG_WIDTH, IMG_HEIGHT = 256, 128
BATCH_SIZE = 1
NUM_EPOCHS = 10000
SEED = 1234
CONTINUE_PREV_TRAINING = False
WEIGHTS_PATH = './177000.torch'
PNG_PATH = r"./generated_pngs/"
SH_PATH = r"./generated_sh/"
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# Set the random seed for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def load_checkpoint(filepath, model, optimizer, scheduler):
    """Load model checkpoint from a file."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['global_itr']


class CustomDataset(Dataset):
    """Custom dataset for loading and processing images and SH coefficients."""

    def __init__(self, png_path, sh_path):
        self.png_path = png_path
        self.sh_path = sh_path
        self.img_list = os.listdir(os.path.join(png_path))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        if ".png" in img_name:
            img = cv2.imread(os.path.join(self.png_path, img_name))
            base_name = img_name.replace(".png", "")
            sh_file = os.path.join(self.sh_path, base_name + ".txt")
            sh_values = []
            with open(sh_file) as f:
                lines = f.readlines()
                for line in lines:
                    values = line.split(",")
                    sh_values.extend(float(value) for value in values)
            img = self.transform(img)
            return img, torch.tensor(sh_values)


def main():
    dataset = CustomDataset(PNG_PATH, SH_PATH)

    # Split data into training and validation
    train_size = int(0.98 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=10, pin_memory=True)

    # Initialize and set up the model
    net = models.mobilenet_v2(pretrained=False, width_mult=1.0)
    net.classifier = torch.nn.Linear(in_features=1280, out_features=27, bias=True)
    net.to(DEVICE)

    optimizer = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    scaler = GradScaler()
    avg_loss = np.zeros([50])

    #if CONTINUE_PREV_TRAINING:
        #global_itr = load_checkpoint(WEIGHTS_PATH, net, optimizer, scheduler)
    #else:
    #    global_itr = 0

    writer = SummaryWriter()

    for epoch in range(NUM_EPOCHS):
        # Training loop
        for itr, (images, prop) in enumerate(train_dataloader):
            images = images.to(DEVICE)
            color = prop.to(DEVICE)

            with autocast():
                pred = net(images)
                loss = torch.abs(pred - color).mean()

            net.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            avg_loss[itr % 50] = loss.item()
            print(itr, ") Loss=", loss.item(), 'AverageLoss', avg_loss.mean())

            writer.add_scalar("Loss/train", loss.item(), global_itr)

            if global_itr % 1000 == 0:
                # Validation loop
                val_losses = []
                for val_images, val_prop in val_dataloader:
                    val_images = val_images.to(DEVICE)
                    val_color = val_prop.to(DEVICE)

                    with torch.no_grad():
                        val_pred = net(val_images)
                        val_loss = torch.abs(val_pred - val_color).mean()
                        val_losses.append(val_loss.item())

                avg_val_loss = np.mean(val_losses)
                writer.add_scalar("Loss/validation", avg_val_loss, global_itr)
                scheduler.step(avg_val_loss)

                print("Saving Model" + str(global_itr) + ".torch")
                torch.save(net.state_dict(), str(global_itr) + ".torch")

            global_itr += 1  # Increment the global iteration counter

    writer.close()


if __name__ == '__main__':
    main()

