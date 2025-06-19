# %%
!unzip airbus-ship-detection.zip

# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
import torch.optim as optim
from tqdm.auto import tqdm
%matplotlib inline

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
#1.1 Model Parameters
BASE_DIR = ''
TRAIN_DIR = BASE_DIR + 'train_v2/'
TEST_DIR = BASE_DIR + 'test_v2/'

# %%
train = os.listdir(TRAIN_DIR)
test = os.listdir(TEST_DIR)

print(f"Train files: {len(train)}")
print(f"Test files :  {len(test)}")

# %%
masks = pd.read_csv(os.path.join(BASE_DIR, 'train_ship_segmentations_v2.csv'))
not_empty = pd.notna(masks.EncodedPixels)
print(not_empty.sum(), 'masks in', masks[not_empty].ImageId.nunique(), 'images')
print((~not_empty).sum(), 'empty images in', masks.ImageId.nunique(), 'total images')
masks.head()

# %%
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def masks_as_image(in_mask_list):
    all_masks = np.zeros((768, 768), dtype = np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return all_masks

# %%
masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)
masks.drop(['ships'], axis=1, inplace=True)

# %%
unique_img_ids['ships'].hist(bins=unique_img_ids['ships'].max())
print('Max of ships : ',unique_img_ids['ships'].max())
print('Avg of ships : ',unique_img_ids['ships'].mean())

# %%
SAMPLES_PER_GROUP = 4000
balanced_train_df = unique_img_ids.groupby('ships').apply(lambda x: x.sample(SAMPLES_PER_GROUP) if len(x) > SAMPLES_PER_GROUP else x)
balanced_train_df['ships'].hist(bins=balanced_train_df['ships'].max()+1)
print(balanced_train_df.shape[0], 'masks')

# %%
balanced_train_df.head()

# %%
class MyDataloader(Dataset):
    """
    Custom Dataset for images (.jpg) and masks from RLE in CSV.
    """
    def __init__(self, image_dir, csv_path, filelist, transform=None, mask_shape=(768, 768)):
        self.image_dir = image_dir
        self.csv_path = csv_path
        self.filelist = filelist
        self.transform = transform
        self.mask_shape = mask_shape

        # Read filelist
        try:
            with open(self.filelist, 'r') as f:
                self.image_filenames = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Error: Filelist not found at {self.filelist}")
            self.image_filenames = []
        except Exception as e:
            print(f"An error occurred while reading {self.filelist}: {e}")
            self.image_filenames = []

        if not self.image_filenames:
            print(f"Warning: No image filenames loaded from {self.filelist}. Dataset will be empty.")

        # Read CSV and build mapping: image_id -> list of RLEs
        self.masks_df = pd.read_csv(self.csv_path)
        self.imgid_to_rles = self.masks_df.groupby('ImageId')['EncodedPixels'].apply(list).to_dict()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        if idx >= len(self.image_filenames):
            raise IndexError("Index out of bounds")

        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_filename)

        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError as e:
            print(f"Error loading file: {e}")
            raise FileNotFoundError(f"Could not find image for index {idx}: {image_path}") from e
        except Exception as e:
            print(f"Error opening image at index {idx} ({image_filename}): {e}")
            raise RuntimeError(f"Error opening image at index {idx} ({image_filename})") from e

        # Get RLEs for this image
        rle_list = self.imgid_to_rles.get(image_filename, [])
        # Compose mask from all RLEs
        mask = np.zeros(self.mask_shape, dtype=np.uint8)
        for rle in rle_list:
            if isinstance(rle, str):
                mask |= rle_decode(rle, shape=self.mask_shape)

        image_np = np.array(image)
        mask_np = mask

        image_np = image_np[::3, ::3, ...]
        mask_np = mask_np[::3, ::3]

        if self.transform:
            try:
                transformed = self.transform(image=image_np, mask=mask_np)
                image = transformed['image'].float()
                mask = transformed['mask'].float()
                if mask.ndim == 3 and mask.shape[0] == 1:
                    mask = mask.squeeze(0)
            except Exception as e:
                print(f"Error during transformation for index {idx} ({image_filename}): {e}")
                raise RuntimeError(f"Error during transformation for index {idx} ({image_filename})") from e

        if mask.ndim != 2:
            print(f"Warning: Mask for index {idx} ({image_filename}) has unexpected dimensions: {mask.shape}. Attempting to fix.")
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)
            else:
                raise ValueError(f"Mask for index {idx} has unexpected shape {mask.shape} after transform.")

        return image, mask

# %%
from sklearn.model_selection import train_test_split

# Stratified split on balanced_train_df using 'has_ship'
trainval_df, test_df = train_test_split(
    balanced_train_df,
    test_size=0.2,
    stratify=balanced_train_df['has_ship'],
    random_state=42
)

train_df, val_df = train_test_split(
    trainval_df,
    test_size=0.2,
    stratify=trainval_df['has_ship'],
    random_state=42
)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Save filelists for your dataloader (image names with .jpg)
train_df['ImageId'].to_csv('train_filelist.txt', index=False, header=False)
val_df['ImageId'].to_csv('val_filelist.txt', index=False, header=False)
test_df['ImageId'].to_csv('test_filelist.txt', index=False, header=False)

# %%
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    #A.Resize(256, 256),
    A.Rotate(limit=45, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.25, rotate_limit=0, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Affine(shear=(-10, 10), p=0.5),
    A.RandomResizedCrop(size=(256, 256), scale=(0.9, 1.0), ratio=(1.0, 1.0), p=0.5),
    A.OneOf([
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_REFLECT, p=1.0),
    ], p=1.0),
    A.Normalize(normalization='min_max_per_channel'),
    ToTensorV2()
])

val_transform = A.Compose([
    #A.Resize(256, 256),
    A.Normalize(normalization='min_max_per_channel'),
    ToTensorV2()
])

test_transform = A.Compose([
    #A.Resize(256, 256),
    A.Normalize(normalization='min_max_per_channel'),
    ToTensorV2()
])

# Paths
CSV_PATH = os.path.join(BASE_DIR, 'train_ship_segmentations_v2.csv')
TRAIN_IMG_DIR = TRAIN_DIR  # usually 'train_v2/'

# Create datasets
train_dataset = MyDataloader(
    image_dir=TRAIN_IMG_DIR,
    csv_path=CSV_PATH,
    filelist='train_filelist.txt',
    transform=train_transform
)

val_dataset = MyDataloader(
    image_dir=TRAIN_IMG_DIR,
    csv_path=CSV_PATH,
    filelist='val_filelist.txt',
    transform=val_transform
)

test_dataset = MyDataloader(
    image_dir=TRAIN_IMG_DIR,
    csv_path=CSV_PATH,
    filelist='test_filelist.txt',
    transform=test_transform
)

# Create DataLoaders
from torch.utils.data import DataLoader

BATCH_SIZE = 50 # Set as appropriate

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# %%
NUM_EPOCHS = 10  # Set as needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.cuda()
criterion = criterion.to(device)

best_val_iou = 0.0  # Track best IoU

torch.autograd.set_detect_anomaly(True)

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    train_iou = 0.0
    # Add tqdm for training loop
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
        images = images.to(device)
        masks = masks.to(device)
        with torch.amp.autocast(device_type="cuda"):
            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.squeeze(1)  # (N, H, W)
            loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        train_iou += iou_pytorch(torch.sigmoid(outputs).detach().cpu(), masks.detach().cpu()) * images.size(0)

    train_loss /= len(train_loader.dataset)
    train_iou /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    # Add tqdm for validation loop
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)
            val_iou += iou_pytorch(torch.sigmoid(outputs).cpu(), masks.cpu()) * images.size(0)

    val_loss /= len(val_loader.dataset)
    val_iou /= len(val_loader.dataset)

    # Step the scheduler with validation loss
    scheduler.step(val_loss)

    # Save best model based on validation IoU
    if val_iou > best_val_iou:
        best_val_iou = val_iou
        torch.save(model.state_dict(), "best_model.pth")
        print(f"Best model saved at epoch {epoch+1} with Val IoU: {val_iou:.4f}")

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")
    
    if epoch % 5 == 0:
        print(f"Sample predictions: {torch.sigmoid(outputs[0]).min():.4f} to {torch.sigmoid(outputs[0]).max():.4f}")
        print(f"Sample targets: {masks[0].min():.4f} to {masks[0].max():.4f}")
        print(f"Raw loss value: {loss.item():.6f}")
    

# %%
if train_loader:
    print("\nChecking one batch from train_loader...")
    try:
        images, masks = next(iter(train_loader))
        print(f"Image batch shape: {images.shape}")
        print(f"Mask batch shape: {masks.shape}")
        print(f"Image batch dtype: {images.dtype}")
        print(f"Mask batch dtype: {masks.dtype}")
        print("Successfully loaded one batch from train_loader.")
    except StopIteration:
        print("Train loader is empty.")
    except Exception as e:
        print(f"Error loading batch from train_loader: {e}")


if val_loader:
    print("\nChecking one batch from val_loader...")
    try:
        images, masks = next(iter(val_loader))
        print(f"Image batch shape: {images.shape}")
        print(f"Mask batch shape: {masks.shape}")
        print(f"Image batch dtype: {images.dtype}")
        print(f"Mask batch dtype: {masks.dtype}")
        print("Successfully loaded one batch from val_loader.")
    except StopIteration:
          print("Validation loader is empty.")
    except Exception as e:
        print(f"Error loading batch from val_loader: {e}")

if test_loader:
    print("\nChecking one batch from test_loader...")
    try:
        images, masks = next(iter(test_loader))
        print(f"Image batch shape: {images.shape}")
        print(f"Mask batch shape: {masks.shape}")
        print(f"Image batch dtype: {images.dtype}")
        print(f"Mask batch dtype: {masks.dtype}")
        print("Successfully loaded one batch from test_loader.")
    except StopIteration:
          print("Test loader is empty.")
    except Exception as e:
        print(f"Error loading batch from test_loader: {e}")

# %%
# Get one batch
images, masks = next(iter(train_loader))

# Show up to 4 samples
n = min(4, images.shape[0])
plt.figure(figsize=(12, 6))
for i in range(n):
    plt.subplot(2, n, i + 1)
    img = images[i].cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
    img = (img - img.min()) / (img.max() - img.min())  # Normalize for display
    plt.imshow(img)
    plt.axis('off')
    plt.title('Image')
    
    plt.subplot(2, n, n + i + 1)
    mask = masks[i].cpu().numpy()
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.title('Mask')
plt.tight_layout()
plt.show()

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, input_channels=3, upsample_mode='deconv'):
        super(UNet, self).__init__()
        self.upsample_mode = upsample_mode

        # Encoder
        self.c1 = self.conv_block(input_channels, 16, dropout=0.1)
        self.p1 = nn.MaxPool2d(2)
        self.c2 = self.conv_block(16, 32, dropout=0.1)
        self.p2 = nn.MaxPool2d(2)
        self.c3 = self.conv_block(32, 64, dropout=0.2)
        self.p3 = nn.MaxPool2d(2)
        self.c4 = self.conv_block(64, 128, dropout=0.2)
        self.p4 = nn.MaxPool2d(2)

        # Bottleneck
        self.c5 = self.conv_block(128, 256, dropout=0.3)

        # Decoder
        self.u6 = self.upsample(256, 128)
        self.c6 = self.conv_block(256, 128, dropout=0.2)
        self.u7 = self.upsample(128, 64)
        self.c7 = self.conv_block(128, 64, dropout=0.2)
        self.u8 = self.upsample(64, 32)
        self.c8 = self.conv_block(64, 32, dropout=0.1)
        self.u9 = self.upsample(32, 16)
        self.c9 = self.conv_block(32, 16, dropout=0.1)

        self.final = nn.Conv2d(16, 1, kernel_size=1)

        self._init_weights()  # Initialize weights with He normal

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def conv_block(self, in_channels, out_channels, dropout=0.0):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*layers)

    def upsample(self, in_channels, out_channels):
        if self.upsample_mode == 'deconv':
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )

    def forward(self, x):
        # Encoder
        c1 = self.c1(x)
        p1 = self.p1(c1)
        c2 = self.c2(p1)
        p2 = self.p2(c2)
        c3 = self.c3(p2)
        p3 = self.p3(c3)
        c4 = self.c4(p3)
        p4 = self.p4(c4)

        # Bottleneck
        c5 = self.c5(p4)

        # Decoder
        u6 = self.u6(c5)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = self.c6(u6)
        u7 = self.u7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.c7(u7)
        u8 = self.u8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.c8(u8)
        u9 = self.u9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.c9(u9)

        out = self.final(c9)
        #out = torch.sigmoid(out)
        return out

# %%
model = UNet(input_channels=3, upsample_mode='deconv')  # or 'simple'

# %%
NUM_CLASSES = 1
INPUT_CHANNELS = 3

# %%
dummy_input = torch.randn(4, INPUT_CHANNELS, 256, 256)

try:
    output = model(dummy_input)
    print(f"Model Instantiated Successfully.")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    assert output.shape == (4, NUM_CLASSES, 256, 256), "Output shape is incorrect!"
    print("Output shape is correct.")

except Exception as e:
    print(f"Error during model forward pass: {e}")

# %%
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Use BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        
        # Get probabilities for focal loss calculation
        p_t = torch.sigmoid(inputs)
        p_t = torch.where(targets == 1, p_t, 1 - p_t)
        
        # Calculate focal loss
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * bce_loss
        
        return focal_loss.mean()

# %%
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: (N, 1, H, W) or (N, H, W), raw logits
        # targets: (N, H, W) or (N, 1, H, W)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        bce = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='mean')
        probs = torch.sigmoid(inputs)
        intersection = (probs * targets).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return bce + dice_loss

# %%
LEARNING_RATE = 1e-5

# %%
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

# %%
def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor, threshold=0.5, eps=1e-6):
    """
    Computes IoU for binary segmentation.
    Args:
        outputs: (N, 1, H, W) or (N, H, W) - sigmoid output from model
        labels: (N, H, W) or (N, 1, H, W) - ground truth mask
        threshold: threshold to binarize outputs
        eps: small value to avoid division by zero
    Returns:
        IoU score (float)
    """
    if outputs.shape != labels.shape:
        outputs = outputs.squeeze(1)
    preds = (outputs > threshold).float()
    labels = labels.float()
    intersection = (preds * labels).sum(dim=(1,2))
    union = ((preds + labels) >= 1).sum(dim=(1,2))
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()

# %%
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA")

# %%
print(torch.version.cuda)

# %%
for images, masks in train_loader:
    pos_ratio = masks.float().mean().item()
    print(f"Positive pixel ratio in batch: {pos_ratio:.4f}")
    if pos_ratio < 0.01 or pos_ratio > 0.99:
        print("WARNING: Severe class imbalance detected!")
    break

# %%
NUM_EPOCHS = 10  # Set as needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.cuda()
criterion = criterion.to(device)

best_val_iou = 0.0  # Track best IoU

torch.autograd.set_detect_anomaly(True)

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    train_iou = 0.0
    # Add tqdm for training loop
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
        images = images.to(device)
        masks = masks.to(device)
        with torch.amp.autocast(device_type="cuda"):
            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.squeeze(1)  # (N, H, W)
            loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        train_iou += iou_pytorch(torch.sigmoid(outputs).detach().cpu(), masks.detach().cpu()) * images.size(0)

    train_loss /= len(train_loader.dataset)
    train_iou /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    # Add tqdm for validation loop
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)
            val_iou += iou_pytorch(torch.sigmoid(outputs).cpu(), masks.cpu()) * images.size(0)

    val_loss /= len(val_loader.dataset)
    val_iou /= len(val_loader.dataset)

    # Step the scheduler with validation loss
    scheduler.step(val_loss)

    # Save best model based on validation IoU
    if val_iou > best_val_iou:
        best_val_iou = val_iou
        torch.save(model.state_dict(), "best_model.pth")
        print(f"Best model saved at epoch {epoch+1} with Val IoU: {val_iou:.4f}")

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")
    
    if epoch % 5 == 0:
        print(f"Sample predictions: {torch.sigmoid(outputs[0]).min():.4f} to {torch.sigmoid(outputs[0]).max():.4f}")
        print(f"Sample targets: {masks[0].min():.4f} to {masks[0].max():.4f}")
        print(f"Raw loss value: {loss.item():.6f}")
    

# %%
# Load best model weights
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# Get a batch from the test loader
images, masks = next(iter(test_loader))
images = images.to(device)
masks = masks.to(device)

with torch.no_grad():
    outputs = model(images)
    preds = torch.sigmoid(outputs)
    #preds = (preds > 0.5).float()

# Show up to 4 samples
n = min(4, images.shape[0])
plt.figure(figsize=(12, 9))
for i in range(n):
    # Input image
    plt.subplot(n, 3, i*3 + 1)
    img = images[i].cpu().numpy().transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min())
    plt.imshow(img)
    plt.axis('off')
    plt.title('Image')

    # Ground truth mask
    plt.subplot(n, 3, i*3 + 2)
    plt.imshow(masks[i].cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.title('Ground Truth')

    # Predicted mask
    plt.subplot(n, 3, i*3 + 3)
    plt.imshow(preds[i, 0].cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.title('Prediction')

plt.tight_layout()
plt.show()

# %%
preds

# %%
for i, (images, masks) in enumerate(val_loader):
    if i == 0:
        print(f"Val batch - Images: {images.shape}, Masks: {masks.shape}")
        print(f"Mask stats: min={masks.min():.4f}, max={masks.max():.4f}, mean={masks.mean():.4f}")
        break


