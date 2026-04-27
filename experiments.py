import os
import csv
import cv2
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
TARGET_SIZE = (320, 320)
TRAIN_SIZE, TEST_SIZE, VAL_SIZE = 160, 20, 20
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_unet(path="UNet_res34.pth"):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=1,
        classes=1,
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def load_manet(path="MANet_res34.pth"):
    model = smp.MAnet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=1,
        classes=1,
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


# ── Data loading ──────────────────────────────────────────────────────────────

def get_test_loader(transform=None, transform_label=False, elastic_transform=None, batch_size=8):
    """Return a DataLoader for the fixed test split under the given transforms."""
    full_dataset = datasets.OCTA5003M_Dataset(
        OCTA500_PATH,
        transform=transform,
        target_size=TARGET_SIZE,
        transform_label=transform_label,
        elastic_transform=elastic_transform,
    )
    _, test_set, _ = data.random_split(
        full_dataset,
        [TRAIN_SIZE, TEST_SIZE, VAL_SIZE],
        generator=torch.Generator().manual_seed(42),
    )
    return data.DataLoader(test_set, batch_size=batch_size, shuffle=False)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, loader):
    """Run inference and return averaged metrics over the loader."""
    criterion = nn.BCEWithLogitsLoss()
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
            targets_sq = targets.squeeze(1).long()

            totals['dice']         += metrics.dice_score(preds, targets_sq)
            totals['jaccard']      += metrics.jaccard_index(preds, targets_sq).item()
            totals['balanced_acc'] += metrics.balanced_acc(preds, targets_sq).item()
            totals['auc']          += metrics.auc(probs, targets_sq).item()
            n += 1

    return {k: v / n for k, v in totals.items()}


# ── Experiment configurations ─────────────────────────────────────────────────

EXPERIMENTS = [
    # Baseline — no augmentation
    {
        'name': 'baseline',
        'params': 'none',
        'transform': None,
        'transform_label': False,
        'elastic_transform': None,
    },

    # ── Gaussian noise (intensity only — mask unchanged) ──────────────────────
    *[
        {
            'name': 'gaussian_noise',
            'params': f'std_dev={std_dev}',
            'transform': T.GaussianNoiseTransform(std_dev=std_dev),
            'transform_label': False,
            'elastic_transform': None,
        }
        for std_dev in [0.02, 0.05, 0.10, 0.20, 0.35]
    ],

    # ── Contrast / brightness (intensity only) ────────────────────────────────
    *[
        {
            'name': 'contrast_brightness',
            'params': f'alpha={alpha},beta={beta}',
            'transform': T.ContrastBrightness(alpha=alpha, beta=beta),
            'transform_label': False,
            'elastic_transform': None,
        }
        for alpha, beta in [
            (0.5,   0),   # low contrast
            (0.75,  0),
            (1.5,   0),   # high contrast
            (2.0,   0),   # very high contrast
            (1.0,  30),   # brighter
            (1.0,  60),
            (1.0, -30),   # darker
            (1.5,  30),   # high contrast + bright
        ]
    ],

    # ── Rotation (spatial — apply to both image and mask) ─────────────────────
    *[
        {
            'name': 'rotation',
            'params': f'angle={angle_name}',
            'transform': T.Rotation(rotation=rot_code),
            'transform_label': True,
            'elastic_transform': None,
        }
        for angle_name, rot_code in [
            ('90cw',  cv2.ROTATE_90_CLOCKWISE),
            ('90ccw', cv2.ROTATE_90_COUNTERCLOCKWISE),
            ('180',   cv2.ROTATE_180),
        ]
    ],

    # ── Flip (spatial — apply to both image and mask) ─────────────────────────
    *[
        {
            'name': 'flip',
            'params': f'flip_code={flip_code}({label})',
            'transform': T.ImFlip(flip_code=flip_code),
            'transform_label': True,
            'elastic_transform': None,
        }
        for flip_code, label in [(0, 'vertical'), (1, 'horizontal'), (-1, 'both')]
    ],

    # ── Elastic deformation (spatial — applied jointly to image and mask) ─────
    *[
        {
            'name': 'elastic_deform',
            'params': f'sigma={sigma},points={points}',
            'transform': None,
            'transform_label': False,
            'elastic_transform': T.ElasticDeform(sigma=sigma, points=points),
        }
        for sigma, points in [
            (2,  3),   # mild
            (3,  6),
            (5,  6),   # moderate
            (8,  6),
            (10, 6),   # strong
            (5,  9),   # more control points
            (10, 9),
        ]
    ],
]


# ── Runner ────────────────────────────────────────────────────────────────────

def run_experiments():
    models = {
        'UNet':  load_unet(),
        'MANet': load_manet(),
    }

    csv_path = os.path.join(RESULTS_DIR, 'experiment_results.csv')
    fieldnames = ['model', 'transform', 'params', 'loss', 'dice', 'jaccard', 'balanced_acc', 'auc']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for model_name, model in models.items():
            for exp in EXPERIMENTS:
                print(f"[{model_name}] {exp['name']} | {exp['params']}")

                loader = get_test_loader(
                    transform=exp['transform'],
                    transform_label=exp['transform_label'],
                    elastic_transform=exp['elastic_transform'],
                )
                result = evaluate(model, loader)

                row = {
                    'model':        model_name,
                    'transform':    exp['name'],
                    'params':       exp['params'],
                    'loss':         round(result['loss'],         4),
                    'dice':         round(result['dice'],         4),
                    'jaccard':      round(result['jaccard'],      4),
                    'balanced_acc': round(result['balanced_acc'], 4),
                    'auc':          round(result['auc'],          4),
                }
                writer.writerow(row)
                f.flush()

                print(f"  loss={row['loss']:.4f}  dice={row['dice']:.4f}  "
                      f"jaccard={row['jaccard']:.4f}  bal_acc={row['balanced_acc']:.4f}  "
                      f"auc={row['auc']:.4f}")

    print(f"\nResults saved to {csv_path}")


if __name__ == '__main__':
    run_experiments()
