import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import timm
import pandas as pd
import numpy as np
import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import scipy
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

warnings.filterwarnings("ignore")

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

# PHASE 2 — GLOBAL CONFIGURATION
class CFG:
    TRAIN_CSV = "/home/dheeraj/Documents/Age Detection/final_imdb_train.csv"
    VAL_CSV = "/home/dheeraj/Documents/Age Detection/final_imdb_val.csv"
    TEST_CSV = "/home/dheeraj/Documents/Age Detection/final_imdb_test.csv"
    IMAGE_DIR = "/home/dheeraj/Documents/Age Detection/final_imdb_dataset"
    IMAGE_SIZE = 224
    IN_CHANNELS = 3
    MIN_AGE = 0
    MAX_AGE = 100
    NUM_CLASSES = MAX_AGE - MIN_AGE + 1 
    LDL_SIGMA = 2 
    MODEL_NAME = 'swin_tiny_patch4_window7_224' 
    PRETRAINED = True
    BATCH_SIZE = 16 
    NUM_WORKERS = 4 
    EPOCHS_PHASE1 = 5
    LR_PHASE1 = 1e-3
    EPOCHS_PHASE2 = 15
    WARMUP_EPOCHS = 2 
    LR_PHASE2 = 1e-5 
    WEIGHT_DECAY = 1e-4
    SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG.SEED)

# PHASE 3 — GPU VERIFICATION & MIXED PRECISION
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_vram = torch.cuda.get_device_properties(i).total_memory / (1024**3)
else:
    print("WARNING: No GPU detected. Training will fall back to CPU.")

use_amp = torch.cuda.is_available() 
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# PHASE 4 — LOAD DATASET SPLITS
try:
    train_df = pd.read_csv(CFG.TRAIN_CSV)
    val_df = pd.read_csv(CFG.VAL_CSV)
    test_df = pd.read_csv(CFG.TEST_CSV)
except FileNotFoundError as e:
    print(f"ERROR: Dataset loading failed. Details: {e}")

# PHASE 5 — LABEL DISTRIBUTION GENERATION
age_bins = np.arange(CFG.MIN_AGE, CFG.MAX_AGE + 1)

def generate_label_distribution(age, sigma=CFG.LDL_SIGMA, num_classes=CFG.NUM_CLASSES):
    bins = np.arange(num_classes)
    distribution = np.exp(-0.5 * ((bins - age) / sigma) ** 2)
    distribution = distribution / np.sum(distribution)
    return distribution

# PHASE 6 — DATASET CLASS DESIGN
class AgeDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = row['file_name']
        age = row['age']
        image_path = os.path.join(self.image_dir, file_name)

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found or corrupted at path: {image_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        label_dist = generate_label_distribution(age)
        label_tensor = torch.tensor(label_dist, dtype=torch.float32)

        return image, label_tensor

# PHASE 7 — DATA AUGMENTATION & DATALOADERS
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_train_transforms(image_size=CFG.IMAGE_SIZE):
    return A.Compose([
        A.SmallestMaxSize(max_size=256),
        A.CenterCrop(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5), 
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5), 
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2()
    ])

def get_val_transforms(image_size=CFG.IMAGE_SIZE):
    return A.Compose([
        A.SmallestMaxSize(max_size=256),
        A.CenterCrop(height=image_size, width=image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2()
    ])

train_dataset = AgeDataset(train_df, CFG.IMAGE_DIR, transform=get_train_transforms())
val_dataset = AgeDataset(val_df, CFG.IMAGE_DIR, transform=get_val_transforms())
test_dataset = AgeDataset(test_df, CFG.IMAGE_DIR, transform=get_val_transforms())

# PHASE 8 — DATASET IMBALANCE HANDLING
train_ages = train_df['age'].values.astype(int)
bucket_counts = np.bincount(train_ages, minlength=CFG.NUM_CLASSES)
bucket_weights = np.where(bucket_counts > 0, 1.0 / np.sqrt(bucket_counts), 0.0)

sample_weights = bucket_weights[train_ages]
sample_weights_tensor = torch.from_numpy(sample_weights).float()

train_sampler = WeightedRandomSampler(weights=sample_weights_tensor, num_samples=len(sample_weights_tensor), replacement=True)

# PHASE 9 — DATALOADER OPTIMIZATION
if 'train_dataset' in locals():
    train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, sampler=train_sampler, num_workers=CFG.NUM_WORKERS, pin_memory=True, prefetch_factor=2, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True, prefetch_factor=2, persistent_workers=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True, prefetch_factor=2, persistent_workers=True, drop_last=False)

# PHASE 10 — MODEL INITIALIZATION
class AgeSwinTransformer(nn.Module):
    def __init__(self, model_name=CFG.MODEL_NAME, pretrained=CFG.PRETRAINED, num_classes=CFG.NUM_CLASSES):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features 
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits 

model = AgeSwinTransformer(model_name=CFG.MODEL_NAME, pretrained=CFG.PRETRAINED, num_classes=CFG.NUM_CLASSES).to(CFG.DEVICE)

# PHASE 11 — LOSS FUNCTION (KL DIVERGENCE)
criterion = nn.KLDivLoss(reduction='batchmean')

# PHASE 12 — SMOKE TEST
optimizer = optim.AdamW(model.parameters(), lr=CFG.LR_PHASE1, weight_decay=CFG.WEIGHT_DECAY)

def run_smoke_test(model, loader, criterion, optimizer, scaler, device):
    print("Initiating smoke test on a single batch...\n")
    model.train()
    optimizer.zero_grad()
    
    try:
        images, targets = next(iter(loader))
        images, targets = images.to(device), targets.to(device)
        
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(images)
                log_probs = F.log_softmax(logits, dim=1)
                loss = criterion(log_probs, targets)
            scaler.scale(loss).backward()
        else:
            logits = model(images)
            log_probs = F.log_softmax(logits, dim=1)
            loss = criterion(log_probs, targets)
            loss.backward()
            
        if not torch.isfinite(loss):
            raise ValueError("Loss is completely exploding (NaN or Inf)!")
            
        has_grads = any(param.grad is not None for param in model.parameters())
        if not has_grads:
            raise RuntimeError("Backpropagation failed. No gradients found.")
            
        optimizer.zero_grad()
        print("\n🚀 SMOKE TEST PASSED! No CUDA OOM. Pipeline is ready for full training.\n")
        
    except RuntimeError as e:
        print(f"\n❌ SMOKE TEST FAILED: {e}")

if 'train_loader' in locals():
    run_smoke_test(model, train_loader, criterion, optimizer, scaler, CFG.DEVICE)

# PHASE 13 — PHASE 1 TRAINING (HEAD ONLY)
for param in model.backbone.parameters(): param.requires_grad = False
for param in model.head.parameters(): param.requires_grad = True

trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer_phase1 = optim.AdamW(trainable_params, lr=CFG.LR_PHASE1, weight_decay=CFG.WEIGHT_DECAY)

best_val_loss = float('inf')
history_phase1 = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': [], 'val_decade_acc': [], 'val_tol5_acc': []}
age_bins_tensor = torch.arange(CFG.MIN_AGE, CFG.MAX_AGE + 1, dtype=torch.float32).to(CFG.DEVICE)

print("=== STARTING PHASE 1: HEAD ONLY ===")
for epoch in range(CFG.EPOCHS_PHASE1):
    model.train()
    train_loss = 0.0
    train_preds, train_trues = [], []
    
    train_pbar = tqdm(train_loader, desc=f"Phase 1 Epoch [{epoch+1}/{CFG.EPOCHS_PHASE1}] Train", leave=False)
    for images, targets in train_pbar:
        images, targets = images.to(CFG.DEVICE), targets.to(CFG.DEVICE)
        optimizer_phase1.zero_grad()
        
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(images)
                log_probs = F.log_softmax(logits, dim=1)
                loss = criterion(log_probs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer_phase1)
            scaler.update()
        else:
            logits = model(images)
            log_probs = F.log_softmax(logits, dim=1)
            loss = criterion(log_probs, targets)
            loss.backward()
            optimizer_phase1.step()
            
        train_loss += loss.item() * images.size(0)
        
        with torch.no_grad():
            probabilities = torch.softmax(logits, dim=1)
            batch_pred_ages = torch.sum(probabilities * age_bins_tensor, dim=1)
            batch_true_ages = torch.argmax(targets, dim=1).float()
            train_preds.extend(batch_pred_ages.cpu().numpy())
            train_trues.extend(batch_true_ages.cpu().numpy())
            
        train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    epoch_train_loss = train_loss / len(train_loader.dataset)
    epoch_train_mae = mean_absolute_error(train_trues, train_preds)
    
    model.eval()
    val_loss, all_preds, all_trues = 0.0, [], []
    
    val_pbar = tqdm(val_loader, desc=f"Phase 1 Epoch [{epoch+1}/{CFG.EPOCHS_PHASE1}] Val", leave=False)
    with torch.no_grad():
        for images, targets in val_pbar:
            images, targets = images.to(CFG.DEVICE), targets.to(CFG.DEVICE)
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(images)
                    log_probs = F.log_softmax(logits, dim=1)
                    loss = criterion(log_probs, targets)
            else:
                logits = model(images)
                log_probs = F.log_softmax(logits, dim=1)
                loss = criterion(log_probs, targets)
                
            val_loss += loss.item() * images.size(0)
            
            probabilities = torch.softmax(logits, dim=1)
            pred_ages = torch.sum(probabilities * age_bins_tensor, dim=1)
            true_ages = torch.argmax(targets, dim=1).float() 
            
            all_preds.extend(pred_ages.cpu().numpy())
            all_trues.extend(true_ages.cpu().numpy())
            val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
    epoch_val_loss = val_loss / len(val_loader.dataset)
    all_preds, all_trues = np.array(all_preds), np.array(all_trues)
    
    epoch_val_mae = mean_absolute_error(all_trues, all_preds)
    epoch_val_decade_acc = np.mean((all_preds // 10) == (all_trues // 10))
    epoch_val_tol5_acc = np.mean(np.abs(all_preds - all_trues) <= 5)
    
    history_phase1['train_loss'].append(epoch_train_loss)
    history_phase1['val_loss'].append(epoch_val_loss)
    history_phase1['train_mae'].append(epoch_train_mae)
    history_phase1['val_mae'].append(epoch_val_mae)
    history_phase1['val_decade_acc'].append(epoch_val_decade_acc)
    history_phase1['val_tol5_acc'].append(epoch_val_tol5_acc)
    
    print(f"Epoch [{epoch+1}/{CFG.EPOCHS_PHASE1}] | Train KL: {epoch_train_loss:.4f} | Val KL: {epoch_val_loss:.4f} | Train MAE: {epoch_train_mae:.2f} | Val MAE: {epoch_val_mae:.2f} | Decade Acc: {epoch_val_decade_acc:.4f} | ±5 Acc: {epoch_val_tol5_acc:.4f}")
    
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), 'best_model_phase1.pth')

# PHASE 14 & 16 & 18 & 19 — CONSOLIDATED PHASE 2 FULL FINE-TUNING
model.load_state_dict(torch.load('best_model_phase1.pth'))

for param in model.parameters(): param.requires_grad = True

optimizer_phase2 = optim.AdamW(model.parameters(), lr=CFG.LR_PHASE2, weight_decay=CFG.WEIGHT_DECAY)

warmup = optim.lr_scheduler.LinearLR(optimizer_phase2, start_factor=0.1, total_iters=CFG.WARMUP_EPOCHS)
cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer_phase2, T_max=(CFG.EPOCHS_PHASE2 - CFG.WARMUP_EPOCHS), eta_min=1e-6)
scheduler = optim.lr_scheduler.SequentialLR(optimizer_phase2, schedulers=[warmup, cosine], milestones=[CFG.WARMUP_EPOCHS])

best_val_mae = float('inf')
patience = 5
epochs_no_improve = 0

history_phase2 = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': [], 'val_decade_acc': [], 'val_tol5_acc': []}

print("\n=== STARTING PHASE 2: FULL FINE-TUNING ===")
for epoch in range(CFG.EPOCHS_PHASE2):
    model.train()
    train_loss = 0.0
    train_preds, train_trues = [], []
    
    train_pbar = tqdm(train_loader, desc=f"Phase 2 Epoch [{epoch+1}/{CFG.EPOCHS_PHASE2}] Train", leave=False)
    for images, targets in train_pbar:
        images, targets = images.to(CFG.DEVICE), targets.to(CFG.DEVICE)
        optimizer_phase2.zero_grad()
        
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(images)
                log_probs = F.log_softmax(logits, dim=1)
                loss = criterion(log_probs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer_phase2)
            scaler.update()
        else:
            logits = model(images)
            log_probs = F.log_softmax(logits, dim=1)
            loss = criterion(log_probs, targets)
            loss.backward()
            optimizer_phase2.step()
            
        train_loss += loss.item() * images.size(0)
        
        with torch.no_grad():
            probabilities = torch.softmax(logits, dim=1)
            batch_pred_ages = torch.sum(probabilities * age_bins_tensor, dim=1)
            batch_true_ages = torch.argmax(targets, dim=1).float()
            train_preds.extend(batch_pred_ages.cpu().numpy())
            train_trues.extend(batch_true_ages.cpu().numpy())
            
        train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    scheduler.step()
    epoch_train_loss = train_loss / len(train_loader.dataset)
    epoch_train_mae = mean_absolute_error(train_trues, train_preds)
    current_lr = scheduler.get_last_lr()[0]
    
    model.eval()
    val_loss, all_preds, all_trues = 0.0, [], []
    
    val_pbar = tqdm(val_loader, desc=f"Phase 2 Epoch [{epoch+1}/{CFG.EPOCHS_PHASE2}] Val", leave=False)
    with torch.no_grad():
        for images, targets in val_pbar:
            images, targets = images.to(CFG.DEVICE), targets.to(CFG.DEVICE)
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(images)
                    log_probs = F.log_softmax(logits, dim=1)
                    loss = criterion(log_probs, targets)
            else:
                logits = model(images)
                log_probs = F.log_softmax(logits, dim=1)
                loss = criterion(log_probs, targets)
                
            val_loss += loss.item() * images.size(0)
            
            probabilities = torch.softmax(logits, dim=1)
            pred_ages = torch.sum(probabilities * age_bins_tensor, dim=1)
            true_ages = torch.argmax(targets, dim=1).float() 
            
            all_preds.extend(pred_ages.cpu().numpy())
            all_trues.extend(true_ages.cpu().numpy())
            val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
    epoch_val_loss = val_loss / len(val_loader.dataset)
    all_preds, all_trues = np.array(all_preds), np.array(all_trues)
    
    epoch_val_mae = mean_absolute_error(all_trues, all_preds)
    epoch_val_decade_acc = np.mean((all_preds // 10) == (all_trues // 10))
    epoch_val_tol5_acc = np.mean(np.abs(all_preds - all_trues) <= 5)
    
    history_phase2['train_loss'].append(epoch_train_loss)
    history_phase2['val_loss'].append(epoch_val_loss)
    history_phase2['train_mae'].append(epoch_train_mae)
    history_phase2['val_mae'].append(epoch_val_mae)
    history_phase2['val_decade_acc'].append(epoch_val_decade_acc)
    history_phase2['val_tol5_acc'].append(epoch_val_tol5_acc)
    
    print(f"Epoch [{epoch+1}/{CFG.EPOCHS_PHASE2}] | LR: {current_lr:.6e} | Train KL: {epoch_train_loss:.4f} | Val KL: {epoch_val_loss:.4f} | Train MAE: {epoch_train_mae:.2f} | Val MAE: {epoch_val_mae:.2f} | Decade Acc: {epoch_val_decade_acc:.4f} | ±5 Acc: {epoch_val_tol5_acc:.4f}")
    
    if epoch_val_mae < best_val_mae:
        best_val_mae = epoch_val_mae
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer_phase2.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, 'best_age_model.pth')
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print("Early stopping triggered. Training stopped.")
        break

# PHASE 17 — AGE PREDICTION (INFERENCE HELPER)
def predict_age(model, image_tensor, device=CFG.DEVICE):
    model.eval()
    age_bins = torch.arange(CFG.MIN_AGE, CFG.MAX_AGE + 1, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(image_tensor)
        else:
            logits = model(image_tensor)
            
        probabilities = torch.softmax(logits, dim=1)
        predicted_ages = torch.sum(probabilities * age_bins, dim=1)
        
    return predicted_ages.cpu().numpy()

# PHASE 20 — FINAL TEST EVALUATION
print("\n=== STARTING FINAL TEST EVALUATION ===")
checkpoint = torch.load('best_age_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

test_preds, test_trues = [], []

test_pbar = tqdm(test_loader, desc="Testing Model")
with torch.no_grad():
    for images, targets in test_pbar:
        images = images.to(CFG.DEVICE)
        
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(images)
        else:
            logits = model(images)
            
        probabilities = torch.softmax(logits, dim=1)
        pred_ages = torch.sum(probabilities * age_bins_tensor, dim=1)
        true_ages = torch.argmax(targets, dim=1).float()
        
        test_preds.extend(pred_ages.cpu().numpy())
        test_trues.extend(true_ages.cpu().numpy())

test_preds, test_trues = np.array(test_preds), np.array(test_trues)

test_mae = mean_absolute_error(test_trues, test_preds)
test_decade_acc = np.mean((test_preds // 10) == (test_trues // 10))

print(f"\nFinal Test MAE: {test_mae:.4f}")
print(f"Final Test Decade Accuracy: {test_decade_acc:.4f}")

per_age_mae = {}
for age in range(CFG.MIN_AGE, CFG.MAX_AGE + 1):
    mask = (test_trues == age)
    if np.sum(mask) > 0:
        per_age_mae[age] = mean_absolute_error(test_trues[mask], test_preds[mask])

print(f"Computed Per-Age MAE for {len(per_age_mae)} unique ages in test set.")

# PHASE 21 — PLOTTING
total_epochs = len(history_phase1['train_loss']) + len(history_phase2['train_loss'])
epochs_range = range(1, total_epochs + 1)
phase1_end = len(history_phase1['train_loss'])

fig, axs = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Age Detection Training Metrics & Evaluation', fontsize=16)

axs[0, 0].plot(epochs_range, history_phase1['train_loss'] + history_phase2['train_loss'], label='Train KL Loss')
axs[0, 0].plot(epochs_range, history_phase1['val_loss'] + history_phase2['val_loss'], label='Val KL Loss')
axs[0, 0].axvline(x=phase1_end, color='r', linestyle='--', label='Unfreeze Backbone')
axs[0, 0].set_title('Training and Validation Loss')
axs[0, 0].set_xlabel('Epochs')
axs[0, 0].set_ylabel('Loss')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Updated MAE subplot to include Train MAE
axs[0, 1].plot(epochs_range, history_phase1['train_mae'] + history_phase2['train_mae'], label='Train MAE', color='blue')
axs[0, 1].plot(epochs_range, history_phase1['val_mae'] + history_phase2['val_mae'], label='Validation MAE', color='orange')
axs[0, 1].axvline(x=phase1_end, color='r', linestyle='--')
axs[0, 1].set_title('Train & Validation Mean Absolute Error')
axs[0, 1].set_xlabel('Epochs')
axs[0, 1].set_ylabel('MAE (Years)')
axs[0, 1].legend()
axs[0, 1].grid(True)

axs[1, 0].plot(epochs_range, history_phase1['val_decade_acc'] + history_phase2['val_decade_acc'], label='Decade Accuracy', color='green')
axs[1, 0].plot(epochs_range, history_phase1['val_tol5_acc'] + history_phase2['val_tol5_acc'], label='±5 Years Accuracy', color='purple')
axs[1, 0].axvline(x=phase1_end, color='r', linestyle='--')
axs[1, 0].set_title('Validation Accuracies')
axs[1, 0].set_xlabel('Epochs')
axs[1, 0].set_ylabel('Accuracy')
axs[1, 0].legend()
axs[1, 0].grid(True)

axs[1, 1].bar(list(per_age_mae.keys()), list(per_age_mae.values()), color='teal', alpha=0.7)
axs[1, 1].set_title('Test Set MAE per Age')
axs[1, 1].set_xlabel('True Age')
axs[1, 1].set_ylabel('Average Error (Years)')
axs[1, 1].grid(axis='y')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('training_metrics_plots.png', dpi=300)
print("\nTraining complete! Metrics saved to 'training_metrics_plots.png'.")