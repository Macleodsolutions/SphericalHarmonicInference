import os
import cv2
import torch
import torchvision.transforms as tf
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

# Constants
IMG_WIDTH = 256
IMG_HEIGHT = 128
BATCH_SIZE = 16
EPOCHS = 4
PNG_PATH = r"./generated_pngs/"
SH_PATH = r"./generated_sh/"
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# Custom dataset for loading images and corresponding spherical harmonic values
class CustomDataset(Dataset):
    """Custom dataset for loading and processing images and SH coefficients."""

    def __init__(self, png_path, sh_path):
        self.png_path = png_path
        self.sh_path = sh_path
        self.img_list = os.listdir(os.path.join(png_path))
        self.transform = tf.Compose([tf.ToPILImage(), tf.Resize((IMG_HEIGHT, IMG_WIDTH)), tf.ToTensor(),
                                     tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

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


# Optuna objective function
def objective(trial):
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    dataset = CustomDataset(PNG_PATH, SH_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)

    model = models.mobilenet_v2(pretrained=False, width_mult=2.0)
    model.classifier = torch.nn.Linear(in_features=2560, out_features=27, bias=True)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    scaler = GradScaler()

    # Training loop
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for itr, (images, props) in enumerate(dataloader):
            images = torch.autograd.Variable(images, requires_grad=False).to(DEVICE)
            props = torch.autograd.Variable(props, requires_grad=False).to(DEVICE)

            with autocast():
                pred = model(images)
                loss = torch.abs(pred - props).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad()

            epoch_loss += loss.item()

        scheduler.step(epoch_loss)

    return epoch_loss / (itr + 1)


def main():
    # Optuna study for hyperparameter optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)  # reduced trials for demonstration purposes

    # Print the best trial information
    print("Best trial:")
    best_trial = study.best_trial
    print("  Value (evaluation metric):", best_trial.value)
    print("  Params:")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == '__main__':
    main()
