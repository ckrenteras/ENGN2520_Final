"""
Load the best checkpoints from results/augmented_training/ and evaluate every
model on five evaluation splits:

  clean_val            — same clean 20-sample val split used during training
  clean_test           — same clean 20-sample test split used during training
  test_contrast_low    — test images with alpha=0.75  (reduced contrast)
  test_contrast_high   — test images with alpha=1.50  (boosted contrast)
  test_contrast_bright — test images with alpha=1.50, beta=30 (contrast + brightness)

Output:
  results/augmented_training/eval_results.csv
"""

import os
import csv
import glob
import torch
import torch.nn as nn
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
BATCH_SIZE   = 8
RESULTS_DIR  = os.path.join('results', 'augmented_training')


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split(dataset):
    return data.random_split(
        dataset, [TRAIN_SIZE, TEST_SIZE, VAL_SIZE],
        generator=torch.Generator().manual_seed(42),
    )


def build_eval_loaders():
    """Return a dict of name -> DataLoader for every evaluation configuration."""
    loaders = {}

    # Clean splits (no transform)
    clean = datasets.OCTA5003M_Dataset(OCTA500_PATH, target_size=TARGET_SIZE)
    _, clean_test, clean_val = _split(clean)
    loaders['clean_val']  = data.DataLoader(clean_val,  batch_size=BATCH_SIZE, shuffle=False)
    loaders['clean_test'] = data.DataLoader(clean_test, batch_size=BATCH_SIZE, shuffle=False)

    # Contrast-shifted test sets
    contrast_configs = [
        ('test_contrast_low',    T.ContrastBrightness(alpha=0.75, beta=0)),
        ('test_contrast_high',   T.ContrastBrightness(alpha=1.50, beta=0)),
        ('test_contrast_bright', T.ContrastBrightness(alpha=1.50, beta=30)),
    ]
    for name, transform in contrast_configs:
        ds = datasets.OCTA5003M_Dataset(
            OCTA500_PATH, target_size=TARGET_SIZE, transform=transform,
        )
        _, test_split, _ = _split(ds)
        loaders[name] = data.DataLoader(test_split, batch_size=BATCH_SIZE, shuffle=False)

    return loaders


def make_model(arch):
    kwargs = dict(encoder_name="resnet34", encoder_weights=None,
                  in_channels=1, classes=1)
    if arch == 'UNet':
        return smp.Unet(**kwargs).to(device)
    if arch == 'MANet':
        return smp.MAnet(**kwargs).to(device)
    raise ValueError(f"Unknown arch: {arch}")


def eval_loader(model, loader, criterion):
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
    return {k: round(v / n, 4) for k, v in totals.items()}


def parse_pth(filename):
    """Return (exp_name, arch) from e.g. 'flip_horizontal_UNet_best.pth'."""
    stem = filename.replace('_best.pth', '')
    for arch in ('MANet', 'UNet'):
        if stem.endswith(f'_{arch}'):
            exp_name = stem[: -(len(arch) + 1)]
            return exp_name, arch
    raise ValueError(f"Cannot parse arch from: {filename}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    eval_loaders = build_eval_loaders()
    criterion    = nn.BCEWithLogitsLoss()

    split_names  = list(eval_loaders.keys())
    metric_names = ['loss', 'dice', 'jaccard', 'balanced_acc', 'auc']
    fieldnames   = ['experiment', 'model'] + [
        f'{split}_{m}' for split in split_names for m in metric_names
    ]

    out_csv = os.path.join(RESULTS_DIR, 'eval_results.csv')
    pth_files = sorted(glob.glob(os.path.join(RESULTS_DIR, '*_best.pth')))

    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for pth_path in pth_files:
            fname = os.path.basename(pth_path)
            exp_name, arch = parse_pth(fname)

            print(f"\nLoading {fname}")
            model = make_model(arch)
            model.load_state_dict(torch.load(pth_path, map_location=device))

            row = {'experiment': exp_name, 'model': arch}
            for split_name, loader in eval_loaders.items():
                result = eval_loader(model, loader, criterion)
                for m, v in result.items():
                    row[f'{split_name}_{m}'] = v
                print(f"  {split_name:25s}  dice={result['dice']:.4f}  auc={result['auc']:.4f}")

            writer.writerow(row)
            f.flush()

    print(f"\nResults saved to {out_csv}")


if __name__ == '__main__':
    run()
