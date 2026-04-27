"""
Train UNet and MANet on augmented training sets (original + one augmented copy),
evaluate the best checkpoint on the original clean test set, and log per-epoch metrics.

Output layout:
  results/augmented_training/test_results.csv          — final test metrics for all runs
  results/augmented_training/<exp>_<arch>_epochs.csv   — per-epoch metrics
  results/augmented_training/<exp>_<arch>_best.pth     — best model weights (by val dice)
"""

import os
import csv
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import segmentation_models_pytorch as smp

import datasets
import metrics
import transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

OCTA500_PATH = 'OCTA500_3mm'
TARGET_SIZE  = (320, 320)
TRAIN_SIZE, TEST_SIZE, VAL_SIZE = 160, 20, 20
NUM_EPOCHS   = 100
LR           = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE   = 8
RESULTS_DIR  = os.path.join('results', 'augmented_training')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Data ──────────────────────────────────────────────────────────────────────

def build_loaders(transform=None, transform_label=False, elastic_transform=None):
    """
    train_loader : original 160 samples + augmented 160 samples (320 total)
    val_loader   : clean 20-sample validation split
    test_loader  : clean 20-sample test split

    Both splits use the same fixed seed-42 indices as the rest of the project,
    so the test set is identical to the one used in experiments.py.
    """
    clean = datasets.OCTA5003M_Dataset(OCTA500_PATH, target_size=TARGET_SIZE)

    def split(dataset):
        return data.random_split(
            dataset, [TRAIN_SIZE, TEST_SIZE, VAL_SIZE],
            generator=torch.Generator().manual_seed(42),
        )

    clean_train, clean_test, clean_val = split(clean)

    if transform is not None or elastic_transform is not None:
        aug = datasets.OCTA5003M_Dataset(
            OCTA500_PATH, target_size=TARGET_SIZE,
            transform=transform,
            transform_label=transform_label,
            elastic_transform=elastic_transform,
        )
        aug_train, _, _ = split(aug)
        combined_train = data.ConcatDataset([clean_train, aug_train])
    else:
        combined_train = clean_train

    train_loader = data.DataLoader(combined_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = data.DataLoader(clean_val,      batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = data.DataLoader(clean_test,     batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader


# ── Model ─────────────────────────────────────────────────────────────────────

def make_model(arch):
    kwargs = dict(encoder_name="resnet34", encoder_weights="imagenet",
                  in_channels=1, classes=1)
    if arch == 'UNet':
        return smp.Unet(**kwargs).to(device)
    if arch == 'MANet':
        return smp.MAnet(**kwargs).to(device)
    raise ValueError(f"Unknown arch: {arch}")


# ── Training / evaluation helpers ─────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, n = 0.0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(1), targets.squeeze(1).float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / n


def eval_epoch(model, loader, criterion):
    model.eval()
    totals = dict(loss=0.0, dice=0.0, jaccard=0.0, balanced_acc=0.0, auc=0.0)
    n = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(1), targets.squeeze(1).float())
            totals['loss'] += loss.item()

            probs = torch.sigmoid(outputs).squeeze(1)
            preds = (probs > 0.5).long()
            tgt   = targets.squeeze(1).long()
            totals['dice']         += metrics.dice_score(preds, tgt)
            totals['jaccard']      += metrics.jaccard_index(preds, tgt).item()
            totals['balanced_acc'] += metrics.balanced_acc(preds, tgt).item()
            totals['auc']          += metrics.auc(probs, tgt).item()
            n += 1
    return {k: v / n for k, v in totals.items()}


# ── Per-experiment training run ────────────────────────────────────────────────

def train_and_evaluate(exp_name, arch, train_loader, val_loader, test_loader):
    """
    Train from scratch, evaluating val and test every epoch.
    Best epoch is selected by val_dice; its test metrics are returned.
    """
    model     = make_model(arch)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    epoch_csv = os.path.join(RESULTS_DIR, f'{exp_name}_{arch}_epochs.csv')
    epoch_fields = [
        'epoch',
        'train_loss',
        'val_loss',   'val_dice',   'val_jaccard',   'val_balanced_acc',   'val_auc',
        'test_loss',  'test_dice',  'test_jaccard',  'test_balanced_acc',  'test_auc',
    ]

    best_val_dice = -1.0
    best_row      = None
    best_path     = os.path.join(RESULTS_DIR, f'{exp_name}_{arch}_best.pth')

    with open(epoch_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=epoch_fields)
        writer.writeheader()

        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss   = train_epoch(model, train_loader, criterion, optimizer)
            val_metrics  = eval_epoch(model, val_loader,   criterion)
            test_metrics = eval_epoch(model, test_loader,  criterion)

            row = {
                'epoch':             epoch,
                'train_loss':        round(train_loss,                     4),
                'val_loss':          round(val_metrics['loss'],            4),
                'val_dice':          round(val_metrics['dice'],            4),
                'val_jaccard':       round(val_metrics['jaccard'],         4),
                'val_balanced_acc':  round(val_metrics['balanced_acc'],    4),
                'val_auc':           round(val_metrics['auc'],             4),
                'test_loss':         round(test_metrics['loss'],           4),
                'test_dice':         round(test_metrics['dice'],           4),
                'test_jaccard':      round(test_metrics['jaccard'],        4),
                'test_balanced_acc': round(test_metrics['balanced_acc'],   4),
                'test_auc':          round(test_metrics['auc'],            4),
            }
            writer.writerow(row)
            f.flush()

            if val_metrics['dice'] > best_val_dice:
                best_val_dice = val_metrics['dice']
                best_row      = row
                torch.save(model.state_dict(), best_path)

            print(f"  [{arch}|{exp_name}] {epoch:3d}/{NUM_EPOCHS}  "
                  f"train_loss={row['train_loss']:.4f}  "
                  f"val_dice={row['val_dice']:.4f}  "
                  f"test_dice={row['test_dice']:.4f}  "
                  f"val_auc={row['val_auc']:.4f}")

    print(f"\n  [{arch}|{exp_name}] BEST epoch={best_row['epoch']}  "
          f"test_dice={best_row['test_dice']:.4f}  "
          f"test_auc={best_row['test_auc']:.4f}\n")

    return best_row


# ── Augmentation experiment definitions ───────────────────────────────────────

AUG_EXPERIMENTS = [
    # Baseline — train on original 160 samples only
    {
        'name': 'baseline',
        'transform': None,
        'transform_label': False,
        'elastic_transform': None,
    },

    # ── Spatial augmentations (applied to image + mask) ───────────────────────
    {
        'name': 'flip_horizontal',
        'transform': T.ImFlip(flip_code=1),
        'transform_label': True,
        'elastic_transform': None,
    },
    {
        'name': 'flip_vertical',
        'transform': T.ImFlip(flip_code=0),
        'transform_label': True,
        'elastic_transform': None,
    },
    {
        'name': 'flip_both',
        'transform': T.ImFlip(flip_code=-1),
        'transform_label': True,
        'elastic_transform': None,
    },
    {
        'name': 'rotate_90cw',
        'transform': T.Rotation(cv2.ROTATE_90_CLOCKWISE),
        'transform_label': True,
        'elastic_transform': None,
    },
    {
        'name': 'rotate_90ccw',
        'transform': T.Rotation(cv2.ROTATE_90_COUNTERCLOCKWISE),
        'transform_label': True,
        'elastic_transform': None,
    },
    {
        'name': 'rotate_180',
        'transform': T.Rotation(cv2.ROTATE_180),
        'transform_label': True,
        'elastic_transform': None,
    },

    # ── Elastic deformation (stochastic spatial, applied to image + mask) ─────
    {
        'name': 'elastic_mild',
        'transform': None,
        'transform_label': False,
        'elastic_transform': T.ElasticDeform(sigma=2, points=3),
    },
    {
        'name': 'elastic_moderate',
        'transform': None,
        'transform_label': False,
        'elastic_transform': T.ElasticDeform(sigma=5, points=6),
    },
    {
        'name': 'elastic_strong',
        'transform': None,
        'transform_label': False,
        'elastic_transform': T.ElasticDeform(sigma=10, points=6),
    },

    # ── Intensity augmentations (image only, mask unchanged) ──────────────────
    {
        'name': 'gaussian_noise_mild',
        'transform': T.GaussianNoiseTransform(std_dev=0.02),
        'transform_label': False,
        'elastic_transform': None,
    },
    {
        'name': 'gaussian_noise_moderate',
        'transform': T.GaussianNoiseTransform(std_dev=0.05),
        'transform_label': False,
        'elastic_transform': None,
    },
    {
        'name': 'contrast_low',
        'transform': T.ContrastBrightness(alpha=0.75, beta=0),
        'transform_label': False,
        'elastic_transform': None,
    },
    {
        'name': 'contrast_high',
        'transform': T.ContrastBrightness(alpha=1.5, beta=0),
        'transform_label': False,
        'elastic_transform': None,
    },
    {
        'name': 'contrast_high_bright',
        'transform': T.ContrastBrightness(alpha=1.5, beta=30),
        'transform_label': False,
        'elastic_transform': None,
    },
]


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all():
    test_csv    = os.path.join(RESULTS_DIR, 'test_results.csv')
    test_fields = ['experiment', 'model', 'best_epoch', 'test_loss', 'test_dice',
                   'test_jaccard', 'test_balanced_acc', 'test_auc']

    with open(test_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=test_fields)
        writer.writeheader()

        for exp in AUG_EXPERIMENTS:
            print(f"\n{'='*65}")
            print(f"Experiment: {exp['name']}")
            print('='*65)

            train_loader, val_loader, test_loader = build_loaders(
                transform=exp['transform'],
                transform_label=exp['transform_label'],
                elastic_transform=exp['elastic_transform'],
            )

            for arch in ['UNet', 'MANet']:
                best = train_and_evaluate(
                    exp['name'], arch, train_loader, val_loader, test_loader
                )
                row = {
                    'experiment':        exp['name'],
                    'model':             arch,
                    'best_epoch':        best['epoch'],
                    'test_loss':         best['test_loss'],
                    'test_dice':         best['test_dice'],
                    'test_jaccard':      best['test_jaccard'],
                    'test_balanced_acc': best['test_balanced_acc'],
                    'test_auc':          best['test_auc'],
                }
                writer.writerow(row)
                f.flush()

    print(f"\nTest results saved to {test_csv}")


if __name__ == '__main__':
    run_all()
