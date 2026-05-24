"""Generates notebook.ipynb for the Aquarium Detection project."""
import json
from pathlib import Path

def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}

def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": list(lines)}

cells = []

cells.append(md(
    "# Proiect 3 - Sisteme Inteligente\n",
    "## Object Detection pe Aquarium Dataset cu Arhitectura Multi-Head\n",
    "\n",
    "**Autor:** Nume Prenume  \n",
    "**Curs:** Sisteme Inteligente  \n",
    "**Dataset:** [Aquarium Combined (Roboflow Universe)](https://universe.roboflow.com/brad-dwyer/aquarium-combined)\n",
    "\n",
    "---\n",
    "\n",
    "Acest notebook implementeaza un sistem complet de detectie de obiecte pe imagini din acvarii, folosind o arhitectura **multi-head**:\n",
    "- **Backbone:** ResNet-50 preantrenat pe ImageNet (extractor de caracteristici)\n",
    "- **Detection Head:** head principal pentru predictie bounding box + clasa (per-celula pe grila)\n",
    "- **Auxiliary Classification Head:** head suplimentar pentru clasificare multi-label la nivel de imagine (regularizator)\n",
    "\n",
    "Loss combinat: `L = lambda_det * L_detection + lambda_cls * L_auxiliary`"
))

cells.append(md(
    "## 0. Configuratie Globala\n\n",
    "Toate hyperparametrii sunt grupati aici pentru reproducibilitate si usurinta modificarii. Setarea seed-ului asigura ca rezultatele pot fi replicate."
))

cells.append(code(
    "import os\n",
    "import random\n",
    "import numpy as np\n",
    "import torch\n",
    "\n",
    "# === GLOBAL CONFIGURATION ===\n",
    "SEED = 42\n",
    "BATCH_SIZE = 16\n",
    "NUM_EPOCHS_FROZEN = 5\n",
    "NUM_EPOCHS_UNFROZEN = 10\n",
    "LR_HEAD = 1e-3\n",
    "LR_BACKBONE = 1e-4\n",
    "IMG_SIZE = 416\n",
    "NUM_CLASSES = 7\n",
    "LAMBDA_DET = 1.0\n",
    "LAMBDA_CLS = 0.3\n",
    "GRID_SIZE = 13\n",
    "\n",
    "DATA_DIR = 'data/'\n",
    "MODELS_DIR = 'models/'\n",
    "OUTPUTS_DIR = 'outputs/'\n",
    "FIGURES_DIR = os.path.join(OUTPUTS_DIR, 'figures')\n",
    "CHECKPOINTS_DIR = os.path.join(OUTPUTS_DIR, 'checkpoints')\n",
    "\n",
    "CLASS_NAMES = ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']\n",
    "\n",
    "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "\n",
    "def set_seed(seed=SEED):\n",
    "    random.seed(seed)\n",
    "    np.random.seed(seed)\n",
    "    torch.manual_seed(seed)\n",
    "    if torch.cuda.is_available():\n",
    "        torch.cuda.manual_seed_all(seed)\n",
    "    torch.backends.cudnn.deterministic = True\n",
    "    torch.backends.cudnn.benchmark = False\n",
    "\n",
    "set_seed(SEED)\n",
    "os.makedirs(FIGURES_DIR, exist_ok=True)\n",
    "os.makedirs(CHECKPOINTS_DIR, exist_ok=True)\n",
    "\n",
    "print(f'Random seed: {SEED}')\n",
    "print(f'Device: {DEVICE}')\n",
    "print(f'Classes ({NUM_CLASSES}): {CLASS_NAMES}')"
))

cells.append(md(
    "## 1. Imports & Reproducibilitate\n\n",
    "Importurile sunt grupate logic:\n",
    "- **PyTorch / torchvision** - framework si modele preantrenate\n",
    "- **numpy / pandas** - manipulare date\n",
    "- **matplotlib / PIL** - vizualizare\n",
    "- **albumentations** - augmentari *bbox-safe*\n",
    "- **torchmetrics** - calcul mAP standardizat"
))

cells.append(code(
    "import json, glob, hashlib, shutil, zipfile\n",
    "from pathlib import Path\n",
    "from collections import Counter, defaultdict\n",
    "\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.patches as patches\n",
    "from PIL import Image\n",
    "\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import torchvision\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "\n",
    "import albumentations as A\n",
    "from albumentations.pytorch import ToTensorV2\n",
    "\n",
    "from tqdm.auto import tqdm\n",
    "\n",
    "try:\n",
    "    from torchmetrics.detection import MeanAveragePrecision\n",
    "    HAS_TM = True\n",
    "except Exception:\n",
    "    HAS_TM = False\n",
    "\n",
    "print('Imports OK')\n",
    "print(f'torch {torch.__version__} | torchvision {torchvision.__version__}')"
))

cells.append(md(
    "## 2. Dataset Download & Explorare\n\n",
    "### 2.1 Download Aquarium Dataset\n\n",
    "Folosim **Aquarium Combined** de pe Roboflow Universe (~640 imagini, 7 specii, format COCO JSON). Optiuni de download:\n",
    "1. Roboflow API (necesita cheie API gratuita)\n",
    "2. Download manual de pe https://public.roboflow.com/object-detection/aquarium\n",
    "3. gdown daca exista un mirror Google Drive"
))

cells.append(code(
    "# Metoda 1: Roboflow API\n",
    "# from roboflow import Roboflow\n",
    "# rf = Roboflow(api_key='YOUR_KEY')\n",
    "# rf.workspace('brad-dwyer').project('aquarium-combined').version(2).download('coco', location=DATA_DIR + 'raw')\n",
    "\n",
    "raw_dir = Path(DATA_DIR) / 'raw'\n",
    "raw_dir.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "found = list(raw_dir.rglob('_annotations.coco.json'))\n",
    "print(f'Found {len(found)} annotation files in {raw_dir}')\n",
    "for f in found:\n",
    "    print(' -', f)\n",
    "\n",
    "if not found:\n",
    "    print('\\n[!] Dataset-ul nu a fost gasit. Descarca de la:')\n",
    "    print('    https://public.roboflow.com/object-detection/aquarium')\n",
    "    print('    Dezarhiveaza in data/raw/ (structura: train/, valid/, test/ cu _annotations.coco.json).')"
))

cells.append(md(
    "### 2.2 Parsare anotatii COCO\n\n",
    "Construim un index `{split: [samples]}` cu calea imaginii, bbox-urile (format COCO `[x,y,w,h]`) si etichetele."
))

cells.append(code(
    "def load_coco_split(split_dir):\n",
    "    ann_path = Path(split_dir) / '_annotations.coco.json'\n",
    "    if not ann_path.exists():\n",
    "        return [], {}\n",
    "    with open(ann_path) as f:\n",
    "        coco = json.load(f)\n",
    "    cat_id_to_name = {c['id']: c['name'] for c in coco['categories']}\n",
    "    name_to_idx = {n: i for i, n in enumerate(CLASS_NAMES)}\n",
    "    img_id_to_anns = defaultdict(list)\n",
    "    for ann in coco['annotations']:\n",
    "        img_id_to_anns[ann['image_id']].append(ann)\n",
    "    samples = []\n",
    "    for img in coco['images']:\n",
    "        boxes, labels = [], []\n",
    "        for ann in img_id_to_anns[img['id']]:\n",
    "            cls_name = cat_id_to_name[ann['category_id']]\n",
    "            if cls_name not in name_to_idx:\n",
    "                continue\n",
    "            x, y, w, h = ann['bbox']\n",
    "            if w <= 1 or h <= 1:\n",
    "                continue\n",
    "            boxes.append([x, y, w, h])\n",
    "            labels.append(name_to_idx[cls_name])\n",
    "        samples.append({\n",
    "            'image_path': str(Path(split_dir) / img['file_name']),\n",
    "            'width': img['width'], 'height': img['height'],\n",
    "            'boxes': boxes, 'labels': labels,\n",
    "        })\n",
    "    return samples, cat_id_to_name\n",
    "\n",
    "splits = {}\n",
    "for split in ['train', 'valid', 'test']:\n",
    "    sdir = raw_dir / split\n",
    "    if sdir.exists():\n",
    "        s, _ = load_coco_split(sdir)\n",
    "        splits[split] = s\n",
    "        print(f'{split}: {len(s)} imagini')\n",
    "    else:\n",
    "        splits[split] = []\n",
    "        print(f'{split}: directorul nu exista')"
))

cells.append(md(
    "### 2.3 Analiza Exploratorie (EDA)\n\n",
    "Vizualizam mostre adnotate, distributia claselor si statistici globale."
))

cells.append(code(
    "CLASS_COLORS = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))\n",
    "\n",
    "def draw_sample(ax, sample, title=''):\n",
    "    img = Image.open(sample['image_path']).convert('RGB')\n",
    "    ax.imshow(img); ax.axis('off'); ax.set_title(title, fontsize=9)\n",
    "    for (x, y, w, h), cls in zip(sample['boxes'], sample['labels']):\n",
    "        rect = patches.Rectangle((x, y), w, h, linewidth=2,\n",
    "                                  edgecolor=CLASS_COLORS[cls], facecolor='none')\n",
    "        ax.add_patch(rect)\n",
    "        ax.text(x, y - 3, CLASS_NAMES[cls], fontsize=8, color='white',\n",
    "                bbox=dict(facecolor=CLASS_COLORS[cls], alpha=0.7, pad=1))\n",
    "\n",
    "train = splits.get('train', [])\n",
    "if train:\n",
    "    set_seed(SEED)\n",
    "    idxs = random.sample(range(len(train)), min(6, len(train)))\n",
    "    fig, axes = plt.subplots(2, 3, figsize=(15, 10))\n",
    "    for ax, idx in zip(axes.flat, idxs):\n",
    "        draw_sample(ax, train[idx], title=f'Sample #{idx}')\n",
    "    plt.suptitle('Aquarium Dataset - Mostre adnotate', fontsize=14)\n",
    "    plt.tight_layout()\n",
    "    plt.savefig(os.path.join(FIGURES_DIR, 'eda_samples.png'), dpi=100, bbox_inches='tight')\n",
    "    plt.show()"
))

cells.append(code(
    "if train:\n",
    "    cls_counter = Counter()\n",
    "    for s in train:\n",
    "        cls_counter.update(s['labels'])\n",
    "    counts = [cls_counter.get(i, 0) for i in range(NUM_CLASSES)]\n",
    "    fig, ax = plt.subplots(figsize=(10, 5))\n",
    "    bars = ax.bar(CLASS_NAMES, counts, color=CLASS_COLORS)\n",
    "    ax.set_ylabel('Numar instante'); ax.set_title('Distributia claselor (train set)')\n",
    "    for b, c in zip(bars, counts):\n",
    "        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1, str(c), ha='center', fontsize=10)\n",
    "    plt.xticks(rotation=20); plt.tight_layout()\n",
    "    plt.savefig(os.path.join(FIGURES_DIR, 'class_distribution.png'), dpi=100)\n",
    "    plt.show()\n",
    "    print(f'Total instante train: {sum(counts)}')\n",
    "    print(f'Cea mai rara: {CLASS_NAMES[np.argmin(counts)]} ({min(counts)})')\n",
    "    print(f'Cea mai frecventa: {CLASS_NAMES[np.argmax(counts)]} ({max(counts)})')\n",
    "    print(f'Dezechilibru: {max(counts)/max(min(counts),1):.1f}x')"
))

cells.append(code(
    "all_imgs = sum((splits[s] for s in splits), [])\n",
    "if all_imgs:\n",
    "    widths = [s['width'] for s in all_imgs]\n",
    "    heights = [s['height'] for s in all_imgs]\n",
    "    obj_per_img = [len(s['boxes']) for s in all_imgs]\n",
    "    print(f'Total imagini: {len(all_imgs)}')\n",
    "    print(f'  train: {len(splits[\"train\"])}, val: {len(splits[\"valid\"])}, test: {len(splits[\"test\"])}')\n",
    "    print(f'Rezolutie medie: {np.mean(widths):.0f} x {np.mean(heights):.0f}')\n",
    "    print(f'Obiecte/imagine: mean={np.mean(obj_per_img):.2f}, max={max(obj_per_img)}')\n",
    "    print(f'Seed: {SEED}')"
))

cells.append(md(
    "### 2.4 Justificare alegere dataset\n\n",
    "**Aquarium Combined** este o alegere potrivita:\n",
    "- **Dimensiune optima (~640 imagini):** suficient pentru fine-tuning, antrenabil in ~1-2h pe Colab T4.\n",
    "- **7 clase vizual distincte:** *fish, jellyfish, penguin, puffin, shark, starfish, stingray*.\n",
    "- **Adnotare profesionala:** bbox-uri verificate, format **COCO JSON** standard.\n",
    "- **Split oficial:** train/valid/test definit.\n",
    "- **Provocari reale:** ocluzii, obiecte mici, variatii de iluminare.\n\n",
    "Trade-off principal: la **416x416** pierdem detalii fine dar incape in VRAM."
))

cells.append(md(
    "## 3. Data Cleaning & Preprocessing\n\n",
    "### 3.1 Verificare integritate\n\n",
    "Eliminam imagini corupte, anotatii lipsa si duplicate (hash MD5)."
))

cells.append(code(
    "def md5sum(path, chunk=8192):\n",
    "    h = hashlib.md5()\n",
    "    with open(path, 'rb') as f:\n",
    "        for c in iter(lambda: f.read(chunk), b''):\n",
    "            h.update(c)\n",
    "    return h.hexdigest()\n",
    "\n",
    "def clean_split(samples, name):\n",
    "    corrupted = missing = dupes = 0\n",
    "    seen, keep = set(), []\n",
    "    for s in samples:\n",
    "        p = s['image_path']\n",
    "        if not os.path.exists(p): missing += 1; continue\n",
    "        try:\n",
    "            with Image.open(p) as im: im.verify()\n",
    "        except Exception:\n",
    "            corrupted += 1; continue\n",
    "        h = md5sum(p)\n",
    "        if h in seen: dupes += 1; continue\n",
    "        seen.add(h); keep.append(s)\n",
    "    print(f'[{name}] Found {corrupted} corrupted, {missing} missing, {dupes} duplicates. '\n",
    "          f'Removed {corrupted+missing+dupes} total. Kept {len(keep)}/{len(samples)}.')\n",
    "    return keep\n",
    "\n",
    "for k in list(splits.keys()):\n",
    "    if splits[k]:\n",
    "        splits[k] = clean_split(splits[k], k)"
))

cells.append(md(
    "### 3.2 Decizia privind clasele\n\n",
    "Verificam ca fiecare clasa are minim 20 instante in train. Pastram toate cele 7 clase."
))

cells.append(code(
    "if splits.get('train'):\n",
    "    cls_counter = Counter()\n",
    "    for s in splits['train']: cls_counter.update(s['labels'])\n",
    "    print('Instante per clasa in train:')\n",
    "    for i, n in enumerate(CLASS_NAMES):\n",
    "        c = cls_counter.get(i, 0)\n",
    "        flag = '[OK]' if c >= 30 else '[!]' if c >= 10 else '[X]'\n",
    "        print(f'  {flag} {n:12s}: {c}')\n",
    "    print('\\nDecizie: pastram toate 7 clasele. Cele cu <30 vor avea performanta mai slaba.')"
))

cells.append(md(
    "### 3.3 Pipeline preprocessing\n\n",
    "1. **Resize la 416x416** cu pastrare aspect ratio + padding (letterbox)\n",
    "2. **Normalizare ImageNet** (mean/std standard)\n",
    "3. **Ajustare bbox** dupa resize\n",
    "4. **Conversie la format YOLO** `[xc, yc, w, h]` normalizat [0,1]"
))

cells.append(code(
    "def coco_to_yolo(boxes, img_w, img_h):\n",
    "    out = []\n",
    "    for x, y, w, h in boxes:\n",
    "        out.append([(x + w/2)/img_w, (y + h/2)/img_h, w/img_w, h/img_h])\n",
    "    return out\n",
    "\n",
    "def yolo_to_xyxy(boxes, img_w, img_h):\n",
    "    out = []\n",
    "    for xc, yc, w, h in boxes:\n",
    "        x1 = (xc - w/2)*img_w; y1 = (yc - h/2)*img_h\n",
    "        x2 = (xc + w/2)*img_w; y2 = (yc + h/2)*img_h\n",
    "        out.append([x1, y1, x2, y2])\n",
    "    return out\n",
    "\n",
    "print('Functii preprocessing OK.')"
))

cells.append(md(
    "### 3.4 Train/Val/Test split\n\n",
    "Folosim split-ul oficial Roboflow (~70/20/10), deja stratificat."
))

cells.append(code(
    "for k, v in splits.items():\n",
    "    print(f'{k}: {len(v)} imagini')\n",
    "print(f'Seed: {SEED}')"
))

cells.append(md(
    "## 4. Augmentari\n\n",
    "Folosim **albumentations** pentru augmentari *bbox-safe*.\n\n",
    "**Justificare per-augmentare:**\n",
    "- **HorizontalFlip:** invarianta la reflecție stanga-dreapta\n",
    "- **RandomBrightnessContrast:** robustete la iluminare variabila\n",
    "- **HueSaturationValue:** variatii de culoare a apei\n",
    "- **Affine (scale + translate + rotate):** invarianta la distanta si unghi\n",
    "- **Normalize:** obligatoriu pentru backbone preantrenat\n\n",
    "*Nu folosim VerticalFlip (pestii nu inoata invers) si nici rotatii mari (>15 grade).*"
))

cells.append(code(
    "train_transform = A.Compose([\n",
    "    A.LongestMaxSize(max_size=IMG_SIZE),\n",
    "    A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, fill=0),\n",
    "    A.HorizontalFlip(p=0.5),\n",
    "    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),\n",
    "    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),\n",
    "    A.Affine(scale=(0.85, 1.15), translate_percent=(-0.05, 0.05), rotate=(-10, 10), p=0.4),\n",
    "    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),\n",
    "    ToTensorV2(),\n",
    "], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels'], min_visibility=0.3))\n",
    "\n",
    "val_transform = A.Compose([\n",
    "    A.LongestMaxSize(max_size=IMG_SIZE),\n",
    "    A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, fill=0),\n",
    "    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),\n",
    "    ToTensorV2(),\n",
    "], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))\n",
    "\n",
    "print('Pipeline augmentari construit.')"
))

cells.append(md(
    "### 4.1 Exemple before/after\n\n",
    "Verificam vizual ca bbox-urile raman corecte dupa augmentare."
))

cells.append(code(
    "IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])\n",
    "IMAGENET_STD = np.array([0.229, 0.224, 0.225])\n",
    "\n",
    "def denormalize(t):\n",
    "    img = t.permute(1, 2, 0).cpu().numpy()\n",
    "    img = img * IMAGENET_STD + IMAGENET_MEAN\n",
    "    return np.clip(img, 0, 1)\n",
    "\n",
    "def show_aug(sample, transform, ax_orig, ax_aug):\n",
    "    img = np.array(Image.open(sample['image_path']).convert('RGB'))\n",
    "    boxes, labels = sample['boxes'], sample['labels']\n",
    "    ax_orig.imshow(img); ax_orig.axis('off'); ax_orig.set_title('Original')\n",
    "    for (x, y, w, h), c in zip(boxes, labels):\n",
    "        ax_orig.add_patch(patches.Rectangle((x, y), w, h, lw=2,\n",
    "                                              edgecolor=CLASS_COLORS[c], facecolor='none'))\n",
    "    out = transform(image=img, bboxes=boxes, class_labels=labels)\n",
    "    aimg = denormalize(out['image'])\n",
    "    ax_aug.imshow(aimg); ax_aug.axis('off'); ax_aug.set_title('Augmented')\n",
    "    for (x, y, w, h), c in zip(out['bboxes'], out['class_labels']):\n",
    "        ax_aug.add_patch(patches.Rectangle((x, y), w, h, lw=2,\n",
    "                                            edgecolor=CLASS_COLORS[c], facecolor='none'))\n",
    "\n",
    "if splits.get('train'):\n",
    "    set_seed(SEED)\n",
    "    fig, axes = plt.subplots(3, 2, figsize=(10, 14))\n",
    "    for i in range(3):\n",
    "        s = random.choice(splits['train'])\n",
    "        show_aug(s, train_transform, axes[i, 0], axes[i, 1])\n",
    "    plt.tight_layout()\n",
    "    plt.savefig(os.path.join(FIGURES_DIR, 'augmentations.png'), dpi=100, bbox_inches='tight')\n",
    "    plt.show()"
))

cells.append(md(
    "## 5. Model & Arhitectura\n\n",
    "### 5.1 Dataset class\n\n",
    "`AquariumDataset` intoarce `(image_tensor, target_dict)` cu:\n",
    "- `boxes`: tensor [N, 4] format YOLO normalizat\n",
    "- `labels`: tensor [N]\n",
    "- `image_labels`: vector [NUM_CLASSES] multi-hot pentru aux head\n",
    "- `detection_target`: tensor [NUM_CLASSES+5, GRID, GRID] pentru detection loss"
))

cells.append(code(
    "class AquariumDataset(Dataset):\n",
    "    def __init__(self, samples, transform=None, grid=GRID_SIZE, num_classes=NUM_CLASSES):\n",
    "        self.samples = samples; self.transform = transform\n",
    "        self.grid = grid; self.num_classes = num_classes\n",
    "\n",
    "    def __len__(self): return len(self.samples)\n",
    "\n",
    "    def build_target_grid(self, yolo_boxes, labels):\n",
    "        G = self.grid\n",
    "        t = torch.zeros(5 + self.num_classes, G, G)\n",
    "        for (xc, yc, w, h), cls in zip(yolo_boxes, labels):\n",
    "            gx = min(int(xc * G), G - 1)\n",
    "            gy = min(int(yc * G), G - 1)\n",
    "            if t[0, gy, gx] == 1: continue\n",
    "            t[0, gy, gx] = 1.0\n",
    "            t[1, gy, gx] = xc * G - gx\n",
    "            t[2, gy, gx] = yc * G - gy\n",
    "            t[3, gy, gx] = w\n",
    "            t[4, gy, gx] = h\n",
    "            t[5 + cls, gy, gx] = 1.0\n",
    "        return t\n",
    "\n",
    "    def __getitem__(self, idx):\n",
    "        s = self.samples[idx]\n",
    "        img = np.array(Image.open(s['image_path']).convert('RGB'))\n",
    "        boxes = list(s['boxes']); labels = list(s['labels'])\n",
    "        if self.transform:\n",
    "            out = self.transform(image=img, bboxes=boxes, class_labels=labels)\n",
    "            img_t = out['image']\n",
    "            boxes = list(out['bboxes']); labels = list(out['class_labels'])\n",
    "        else:\n",
    "            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.\n",
    "        yolo = coco_to_yolo(boxes, IMG_SIZE, IMG_SIZE) if boxes else []\n",
    "        det_target = self.build_target_grid(yolo, labels)\n",
    "        img_labels = torch.zeros(self.num_classes)\n",
    "        for c in labels: img_labels[c] = 1.0\n",
    "        return img_t, {\n",
    "            'boxes': torch.tensor(yolo, dtype=torch.float32) if yolo else torch.zeros((0, 4)),\n",
    "            'labels': torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long),\n",
    "            'image_labels': img_labels,\n",
    "            'detection_target': det_target,\n",
    "        }\n",
    "\n",
    "def collate_fn(batch):\n",
    "    imgs = torch.stack([b[0] for b in batch])\n",
    "    targets = [b[1] for b in batch]\n",
    "    return imgs, {\n",
    "        'detection_target': torch.stack([t['detection_target'] for t in targets]),\n",
    "        'image_labels': torch.stack([t['image_labels'] for t in targets]),\n",
    "        'boxes_list': [t['boxes'] for t in targets],\n",
    "        'labels_list': [t['labels'] for t in targets],\n",
    "    }\n",
    "\n",
    "print('AquariumDataset definit.')"
))

cells.append(md(
    "### 5.2 Arhitectura MultiHeadDetector\n\n",
    "**Rolul componentelor:**\n",
    "- **Backbone (ResNet-50):** extrage features ierarhice. Straturile timpurii detecteaza muchii/texturi; cele adanci detecteaza concepte semantice (inotatoare, tentacule). Pretraining-ul pe ImageNet aduce 'cunoastere generala'.\n",
    "- **Detection Head:** pentru fiecare celula din grila 13x13 (output spatial dupa backbone pentru 416x416), prezice: (a) objectness, (b) coordonatele bbox-ului, (c) clasa.\n",
    "- **Auxiliary Classification Head:** prezice multi-label *care* specii apar in imagine, la nivel global. Actioneaza ca **regularizator** - forteaza backbone-ul sa invete reprezentari utile si pentru clasificare globala.\n\n",
    "**Loss combinat:** `L = 1.0 * L_detection + 0.3 * L_auxiliary`"
))

cells.append(code(
    "class MultiHeadDetector(nn.Module):\n",
    "    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):\n",
    "        super().__init__()\n",
    "        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None\n",
    "        resnet = torchvision.models.resnet50(weights=weights)\n",
    "        self.backbone = nn.Sequential(*list(resnet.children())[:-2])\n",
    "        self.detection_head = nn.Sequential(\n",
    "            nn.Conv2d(2048, 512, 3, padding=1),\n",
    "            nn.BatchNorm2d(512), nn.ReLU(inplace=True),\n",
    "            nn.Conv2d(512, 256, 3, padding=1),\n",
    "            nn.BatchNorm2d(256), nn.ReLU(inplace=True),\n",
    "            nn.Conv2d(256, num_classes + 5, 1),\n",
    "        )\n",
    "        self.aux_cls_head = nn.Sequential(\n",
    "            nn.AdaptiveAvgPool2d(1), nn.Flatten(),\n",
    "            nn.Linear(2048, 256), nn.ReLU(inplace=True),\n",
    "            nn.Dropout(0.3), nn.Linear(256, num_classes),\n",
    "        )\n",
    "\n",
    "    def freeze_backbone(self):\n",
    "        for p in self.backbone.parameters(): p.requires_grad = False\n",
    "\n",
    "    def unfreeze_backbone(self):\n",
    "        for p in self.backbone.parameters(): p.requires_grad = True\n",
    "\n",
    "    def forward(self, x):\n",
    "        feats = self.backbone(x)\n",
    "        return self.detection_head(feats), self.aux_cls_head(feats)\n",
    "\n",
    "model = MultiHeadDetector(num_classes=NUM_CLASSES).to(DEVICE)\n",
    "n_params = sum(p.numel() for p in model.parameters())\n",
    "n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)\n",
    "print(f'Total params: {n_params/1e6:.2f}M | Trainable: {n_trainable/1e6:.2f}M')"
))

cells.append(code(
    "with torch.no_grad():\n",
    "    dummy = torch.randn(2, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)\n",
    "    det, cls = model(dummy)\n",
    "    print(f'Input:        {tuple(dummy.shape)}')\n",
    "    print(f'Detection out: {tuple(det.shape)}  (expected [2, {NUM_CLASSES+5}, {GRID_SIZE}, {GRID_SIZE}])')\n",
    "    print(f'Aux cls out:   {tuple(cls.shape)}  (expected [2, {NUM_CLASSES}])')"
))

cells.append(md(
    "## 6. Fine-tuning\n\n",
    "### 6.1 Loss functions\n\n",
    "**Detection loss:** combinatie BCE (objectness) + Smooth L1 (bbox) + BCE (clase, masked pe celulele pozitive).  \n",
    "**Auxiliary loss:** BCE multi-label (sigmoid pe fiecare clasa)."
))

cells.append(code(
    "class SimpleDetectionLoss(nn.Module):\n",
    "    def __init__(self, num_classes=NUM_CLASSES, lambda_coord=5.0, lambda_noobj=0.5):\n",
    "        super().__init__()\n",
    "        self.C = num_classes\n",
    "        self.lambda_coord = lambda_coord\n",
    "        self.lambda_noobj = lambda_noobj\n",
    "        self.bce = nn.BCEWithLogitsLoss(reduction='none')\n",
    "        self.sl1 = nn.SmoothL1Loss(reduction='none')\n",
    "\n",
    "    def forward(self, pred, target):\n",
    "        obj_p = pred[:, 0]; xy_p = pred[:, 1:3]; wh_p = pred[:, 3:5]; cls_p = pred[:, 5:]\n",
    "        obj_t = target[:, 0]; xy_t = target[:, 1:3]; wh_t = target[:, 3:5]; cls_t = target[:, 5:]\n",
    "        mask = obj_t.unsqueeze(1)\n",
    "        bce_obj = self.bce(obj_p, obj_t)\n",
    "        loss_obj = (bce_obj * obj_t).mean() + self.lambda_noobj * (bce_obj * (1 - obj_t)).mean()\n",
    "        loss_xy = (self.sl1(torch.sigmoid(xy_p), xy_t) * mask).sum() / (mask.sum() * 2 + 1e-6)\n",
    "        loss_wh = (self.sl1(torch.sigmoid(wh_p), wh_t) * mask).sum() / (mask.sum() * 2 + 1e-6)\n",
    "        loss_box = self.lambda_coord * (loss_xy + loss_wh)\n",
    "        loss_cls = (self.bce(cls_p, cls_t) * mask).sum() / (mask.sum() * self.C + 1e-6)\n",
    "        return loss_obj + loss_box + loss_cls, {\n",
    "            'obj': loss_obj.item(), 'box': loss_box.item(), 'cls': loss_cls.item()\n",
    "        }\n",
    "\n",
    "criterion_det = SimpleDetectionLoss().to(DEVICE)\n",
    "criterion_cls = nn.BCEWithLogitsLoss().to(DEVICE)\n",
    "print('Loss functions construite.')"
))

cells.append(md(
    "### 6.2 Optimizer & Scheduler\n\n",
    "**Justificare:**\n",
    "- **AdamW vs SGD:** adapteaza LR per parametru, convergenta mai rapida pe dataset-uri mici; weight decay decuplat de gradient.\n",
    "- **CosineAnnealingLR:** scade LR dupa un cosinus de la valoarea initiala la 0, evita oscilatii.\n",
    "- **LR_HEAD=1e-3 >> LR_BACKBONE=1e-4:** head-ul (init aleator) necesita LR mare; backbone-ul (preantrenat) trebuie ajustat fin pentru a evita *catastrophic forgetting*.\n",
    "- **LAMBDA_CLS=0.3:** auxiliary loss ca regularizator usor.\n",
    "- **Weight decay 1e-4:** L2 light pe ~640 imagini.\n",
    "- **Grad clipping (max_norm=1.0):** previne exploding gradients."
))

cells.append(code(
    "def make_optimizer(model, lr_head=LR_HEAD, lr_backbone=LR_BACKBONE):\n",
    "    head_params = list(model.detection_head.parameters()) + list(model.aux_cls_head.parameters())\n",
    "    return torch.optim.AdamW([\n",
    "        {'params': model.backbone.parameters(), 'lr': lr_backbone},\n",
    "        {'params': head_params, 'lr': lr_head},\n",
    "    ], weight_decay=1e-4)\n",
    "\n",
    "print('Optimizer factory OK.')"
))

cells.append(md("### 6.3 Training & Validation loops"))

cells.append(code(
    "def train_epoch(model, loader, optimizer, criterion_det, criterion_cls, device, use_aux=True):\n",
    "    model.train()\n",
    "    total, n = 0.0, 0\n",
    "    pbar = tqdm(loader, desc='train', leave=False)\n",
    "    for imgs, targets in pbar:\n",
    "        imgs = imgs.to(device)\n",
    "        det_t = targets['detection_target'].to(device)\n",
    "        img_lbls = targets['image_labels'].to(device)\n",
    "        optimizer.zero_grad()\n",
    "        det_out, cls_out = model(imgs)\n",
    "        loss_det, _ = criterion_det(det_out, det_t)\n",
    "        if use_aux:\n",
    "            loss_aux = criterion_cls(cls_out, img_lbls)\n",
    "            loss = LAMBDA_DET * loss_det + LAMBDA_CLS * loss_aux\n",
    "        else:\n",
    "            loss = loss_det\n",
    "        loss.backward()\n",
    "        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n",
    "        optimizer.step()\n",
    "        total += loss.item() * imgs.size(0); n += imgs.size(0)\n",
    "        pbar.set_postfix(loss=f'{loss.item():.4f}')\n",
    "    return total / max(n, 1)\n",
    "\n",
    "@torch.no_grad()\n",
    "def validate(model, loader, criterion_det, criterion_cls, device, use_aux=True):\n",
    "    model.eval()\n",
    "    total, n, correct_cls, total_cls = 0.0, 0, 0, 0\n",
    "    for imgs, targets in loader:\n",
    "        imgs = imgs.to(device)\n",
    "        det_t = targets['detection_target'].to(device)\n",
    "        img_lbls = targets['image_labels'].to(device)\n",
    "        det_out, cls_out = model(imgs)\n",
    "        loss_det, _ = criterion_det(det_out, det_t)\n",
    "        if use_aux:\n",
    "            loss = LAMBDA_DET * loss_det + LAMBDA_CLS * criterion_cls(cls_out, img_lbls)\n",
    "        else:\n",
    "            loss = loss_det\n",
    "        total += loss.item() * imgs.size(0); n += imgs.size(0)\n",
    "        pred = (torch.sigmoid(cls_out) > 0.5).float()\n",
    "        correct_cls += (pred == img_lbls).float().sum().item()\n",
    "        total_cls += img_lbls.numel()\n",
    "    return total / max(n, 1), correct_cls / max(total_cls, 1)\n",
    "\n",
    "print('Train/val loops definite.')"
))

cells.append(md(
    "### 6.4 Antrenare in doua faze\n\n",
    "**Faza 1 (5 epoci):** backbone inghetat, antrenam doar head-urile.  \n",
    "**Faza 2 (10 epoci):** dezghetam backbone, LR mai mic, CosineAnnealing.\n\n",
    "Aceasta strategie previne distrugerea features-urilor preantrenate la inceput, cand gradientii head-urilor sunt mari."
))

cells.append(code(
    "if all(len(splits[k]) > 0 for k in ['train', 'valid']):\n",
    "    train_ds = AquariumDataset(splits['train'], transform=train_transform)\n",
    "    val_ds = AquariumDataset(splits['valid'], transform=val_transform)\n",
    "    test_ds = AquariumDataset(splits['test'], transform=val_transform) if splits['test'] else None\n",
    "    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,\n",
    "                              collate_fn=collate_fn, num_workers=0)\n",
    "    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,\n",
    "                            collate_fn=collate_fn, num_workers=0)\n",
    "    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,\n",
    "                             collate_fn=collate_fn, num_workers=0) if test_ds else None\n",
    "    print(f'Loaders OK. Train batches: {len(train_loader)} | Val: {len(val_loader)}')\n",
    "else:\n",
    "    print('[!] Splits incomplete.')"
))

cells.append(code(
    "def train_full(model, train_loader, val_loader, use_aux=True, tag='multihead'):\n",
    "    history = {'train': [], 'val': [], 'aux_acc': []}\n",
    "    optimizer = make_optimizer(model)\n",
    "    model.freeze_backbone()\n",
    "    print(f'\\n=== Phase 1: Frozen backbone ({NUM_EPOCHS_FROZEN} epochs) ===')\n",
    "    for epoch in range(1, NUM_EPOCHS_FROZEN + 1):\n",
    "        tl = train_epoch(model, train_loader, optimizer, criterion_det, criterion_cls, DEVICE, use_aux)\n",
    "        vl, va = validate(model, val_loader, criterion_det, criterion_cls, DEVICE, use_aux)\n",
    "        history['train'].append(tl); history['val'].append(vl); history['aux_acc'].append(va)\n",
    "        print(f'  Ep {epoch:02d} | train {tl:.4f} | val {vl:.4f} | aux_acc {va:.4f}')\n",
    "    model.unfreeze_backbone()\n",
    "    optimizer = make_optimizer(model)\n",
    "    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS_UNFROZEN)\n",
    "    print(f'\\n=== Phase 2: Unfrozen backbone ({NUM_EPOCHS_UNFROZEN} epochs) ===')\n",
    "    best_val = float('inf')\n",
    "    for epoch in range(1, NUM_EPOCHS_UNFROZEN + 1):\n",
    "        tl = train_epoch(model, train_loader, optimizer, criterion_det, criterion_cls, DEVICE, use_aux)\n",
    "        vl, va = validate(model, val_loader, criterion_det, criterion_cls, DEVICE, use_aux)\n",
    "        scheduler.step()\n",
    "        history['train'].append(tl); history['val'].append(vl); history['aux_acc'].append(va)\n",
    "        print(f'  Ep {epoch:02d} | train {tl:.4f} | val {vl:.4f} | aux_acc {va:.4f} | lr {scheduler.get_last_lr()[0]:.2e}')\n",
    "        if vl < best_val:\n",
    "            best_val = vl\n",
    "            torch.save({'model_state_dict': model.state_dict(), 'val_loss': vl, 'epoch': epoch},\n",
    "                       os.path.join(CHECKPOINTS_DIR, f'best_{tag}.pth'))\n",
    "    return history\n",
    "\n",
    "print('train_full OK.')"
))

cells.append(code(
    "set_seed(SEED)\n",
    "model = MultiHeadDetector(num_classes=NUM_CLASSES).to(DEVICE)\n",
    "history_mh = train_full(model, train_loader, val_loader, use_aux=True, tag='multihead')"
))

cells.append(md("## 7. Evaluare & Experimente\n\n### 7.1 Loss curves"))

cells.append(code(
    "def plot_history(history, title='Loss curves', save=None):\n",
    "    epochs = range(1, len(history['train']) + 1)\n",
    "    fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n",
    "    axes[0].plot(epochs, history['train'], label='train', marker='o')\n",
    "    axes[0].plot(epochs, history['val'], label='val', marker='s')\n",
    "    axes[0].axvline(NUM_EPOCHS_FROZEN + 0.5, color='gray', ls='--', label='unfreeze')\n",
    "    axes[0].set_xlabel('Epoca'); axes[0].set_ylabel('Loss'); axes[0].set_title(title)\n",
    "    axes[0].legend(); axes[0].grid(alpha=0.3)\n",
    "    axes[1].plot(epochs, history['aux_acc'], marker='o', color='green')\n",
    "    axes[1].set_xlabel('Epoca'); axes[1].set_ylabel('Aux multi-label acc')\n",
    "    axes[1].set_title('Auxiliary classification accuracy'); axes[1].grid(alpha=0.3)\n",
    "    plt.tight_layout()\n",
    "    if save: plt.savefig(save, dpi=100, bbox_inches='tight')\n",
    "    plt.show()\n",
    "\n",
    "plot_history(history_mh, title='Multi-head: Train vs Val',\n",
    "             save=os.path.join(FIGURES_DIR, 'loss_curves.png'))"
))

cells.append(md(
    "**Interpretare:**\n",
    "- Salt vizibil la `epoch = NUM_EPOCHS_FROZEN + 1` (dezghetare backbone): loss scade brusc, tot modelul se adapteaza.\n",
    "- *train loss << val loss* → overfitting (mai multa augmentare, weight decay mai mare).\n",
    "- *ambele plate la valori mari* → underfitting (capacitate insuficienta, LR prea mic).\n",
    "- Punctul optim de oprire: epoca cu val loss minim (salvam checkpoint atunci)."
))

cells.append(md("### 7.2 Decoder predictii + mAP"))

cells.append(code(
    "@torch.no_grad()\n",
    "def decode_predictions(det_out, conf_thresh=0.3):\n",
    "    B, _, G, _ = det_out.shape\n",
    "    obj = torch.sigmoid(det_out[:, 0])\n",
    "    xy = torch.sigmoid(det_out[:, 1:3])\n",
    "    wh = torch.sigmoid(det_out[:, 3:5])\n",
    "    cls_probs = torch.softmax(det_out[:, 5:], dim=1)\n",
    "    results = []\n",
    "    for b in range(B):\n",
    "        boxes, scores, labels = [], [], []\n",
    "        for gy in range(G):\n",
    "            for gx in range(G):\n",
    "                conf = obj[b, gy, gx].item()\n",
    "                if conf < conf_thresh: continue\n",
    "                cls_p, cls_id = cls_probs[b, :, gy, gx].max(0)\n",
    "                score = conf * cls_p.item()\n",
    "                if score < conf_thresh: continue\n",
    "                xc = (gx + xy[b, 0, gy, gx].item()) / G\n",
    "                yc = (gy + xy[b, 1, gy, gx].item()) / G\n",
    "                w = wh[b, 0, gy, gx].item()\n",
    "                h = wh[b, 1, gy, gx].item()\n",
    "                x1 = (xc - w/2) * IMG_SIZE; y1 = (yc - h/2) * IMG_SIZE\n",
    "                x2 = (xc + w/2) * IMG_SIZE; y2 = (yc + h/2) * IMG_SIZE\n",
    "                boxes.append([x1, y1, x2, y2]); scores.append(score); labels.append(cls_id.item())\n",
    "        if boxes:\n",
    "            bt = torch.tensor(boxes); st = torch.tensor(scores); lt = torch.tensor(labels)\n",
    "            keep = torchvision.ops.nms(bt, st, iou_threshold=0.5)\n",
    "            results.append((bt[keep], st[keep], lt[keep]))\n",
    "        else:\n",
    "            results.append((torch.zeros((0,4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long)))\n",
    "    return results\n",
    "\n",
    "print('Decoder OK.')"
))

cells.append(code(
    "@torch.no_grad()\n",
    "def compute_map(model, loader, device, conf_thresh=0.2):\n",
    "    if not HAS_TM:\n",
    "        print('[!] torchmetrics nu e disponibil; skip mAP.')\n",
    "        return None\n",
    "    metric = MeanAveragePrecision(box_format='xyxy', iou_thresholds=[0.5])\n",
    "    model.eval()\n",
    "    for imgs, targets in loader:\n",
    "        imgs = imgs.to(device)\n",
    "        det_out, _ = model(imgs)\n",
    "        preds_b = decode_predictions(det_out, conf_thresh=conf_thresh)\n",
    "        preds, gts = [], []\n",
    "        for i in range(imgs.size(0)):\n",
    "            pb, ps, pl = preds_b[i]\n",
    "            preds.append({'boxes': pb.cpu(), 'scores': ps.cpu(), 'labels': pl.cpu()})\n",
    "            yb = targets['boxes_list'][i]\n",
    "            xyxy = torch.tensor(yolo_to_xyxy(yb.tolist(), IMG_SIZE, IMG_SIZE)) if yb.numel() > 0 else torch.zeros((0,4))\n",
    "            gts.append({'boxes': xyxy, 'labels': targets['labels_list'][i]})\n",
    "        metric.update(preds, gts)\n",
    "    return metric.compute()\n",
    "\n",
    "map_mh = compute_map(model, val_loader, DEVICE) if HAS_TM else None\n",
    "if map_mh: print('mAP multi-head:', {k: float(v) for k, v in map_mh.items() if v.numel() == 1})"
))

cells.append(md(
    "### 7.3 Ablation: model fara auxiliary head\n\n",
    "Antrenam un model identic dar **fara** auxiliary loss pentru a izola contributia head-ului suplimentar."
))

cells.append(code(
    "set_seed(SEED)\n",
    "model_no_aux = MultiHeadDetector(num_classes=NUM_CLASSES).to(DEVICE)\n",
    "history_no_aux = train_full(model_no_aux, train_loader, val_loader, use_aux=False, tag='no_aux')\n",
    "map_no_aux = compute_map(model_no_aux, val_loader, DEVICE) if HAS_TM else None"
))

cells.append(md("### 7.4 Tabel comparativ"))

cells.append(code(
    "def mAP_val(m):\n",
    "    if m is None: return float('nan')\n",
    "    for k in ('map_50', 'map'):\n",
    "        if k in m: return float(m[k])\n",
    "    return float('nan')\n",
    "\n",
    "rows = [\n",
    "    {'Model': 'Baseline (zero-shot)', 'mAP@0.5': 'n/a', 'Aux Acc': '-', 'Loss final': '-',\n",
    "     'Parametri': f'{sum(p.numel() for p in model.parameters())/1e6:.2f}M'},\n",
    "    {'Model': 'Fine-tuned (detection only)', 'mAP@0.5': f'{mAP_val(map_no_aux):.4f}',\n",
    "     'Aux Acc': '-', 'Loss final': f'{history_no_aux[\"val\"][-1]:.4f}',\n",
    "     'Parametri': f'{sum(p.numel() for p in model_no_aux.parameters())/1e6:.2f}M'},\n",
    "    {'Model': 'Multi-head (det + aux cls)', 'mAP@0.5': f'{mAP_val(map_mh):.4f}',\n",
    "     'Aux Acc': f'{history_mh[\"aux_acc\"][-1]:.4f}', 'Loss final': f'{history_mh[\"val\"][-1]:.4f}',\n",
    "     'Parametri': f'{sum(p.numel() for p in model.parameters())/1e6:.2f}M'},\n",
    "]\n",
    "df = pd.DataFrame(rows)\n",
    "print(df.to_string(index=False))\n",
    "df.to_csv(os.path.join(OUTPUTS_DIR, 'comparison.csv'), index=False)"
))

cells.append(md(
    "**Discutie:** Daca **multi-head** are mAP mai mare decat **fine-tuned only**, auxiliary head-ul a stabilizat reprezentarea. Posibilitati:\n",
    "1. *Auxiliary ajuta* (mAP cu ~1-3% mai mare): regularizare reusita, reprezentari mai discriminative.\n",
    "2. *Auxiliary nu schimba mult*: dataset are deja diversitate suficienta, sau LAMBDA_CLS ar trebui ajustat.\n\n",
    "Baseline-ul nu poate fi evaluat direct (head produce output aleator), deci `n/a`."
))

cells.append(md("### 7.5 Vizualizari predictii"))

cells.append(code(
    "@torch.no_grad()\n",
    "def visualize_predictions(model, dataset, indices, conf=0.3, save=None):\n",
    "    model.eval()\n",
    "    n = len(indices)\n",
    "    rows = (n + 1) // 2\n",
    "    fig, axes = plt.subplots(rows, 2, figsize=(14, 6 * rows))\n",
    "    flat = axes.flat if n > 1 else [axes]\n",
    "    for ax, idx in zip(flat, indices):\n",
    "        img_t, target = dataset[idx]\n",
    "        det_out, _ = model(img_t.unsqueeze(0).to(DEVICE))\n",
    "        preds = decode_predictions(det_out, conf_thresh=conf)[0]\n",
    "        ax.imshow(denormalize(img_t)); ax.axis('off')\n",
    "        for box, score, lbl in zip(*preds):\n",
    "            x1, y1, x2, y2 = box.tolist()\n",
    "            ax.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, lw=2,\n",
    "                                            edgecolor=CLASS_COLORS[lbl.item()], facecolor='none'))\n",
    "            ax.text(x1, y1 - 4, f'{CLASS_NAMES[lbl.item()]} {score:.2f}', fontsize=8,\n",
    "                     color='white', bbox=dict(facecolor=CLASS_COLORS[lbl.item()], alpha=0.7, pad=1))\n",
    "        ax.set_title(f'Sample #{idx}')\n",
    "    plt.tight_layout()\n",
    "    if save: plt.savefig(save, dpi=100, bbox_inches='tight')\n",
    "    plt.show()\n",
    "\n",
    "if test_ds is not None and len(test_ds) > 0:\n",
    "    set_seed(SEED)\n",
    "    idxs = random.sample(range(len(test_ds)), min(6, len(test_ds)))\n",
    "    visualize_predictions(model, test_ds, idxs,\n",
    "                          save=os.path.join(FIGURES_DIR, 'sample_predictions.png'))"
))

cells.append(md(
    "**Analiza esecuri tipice:**\n",
    "- *Obiecte mici la 416x416:* stele de mare sau pesti la distanta sunt sub-rezolutionati → confidence scazut.\n",
    "- *Ocluzii:* bancuri de pesti suprapusi → modelul produce un bbox 'agregat' in loc de mai multe.\n",
    "- *Clase rare* (puffin, stingray): predictii cu scor mic sau confuzie cu clase asemanatoare vizual.\n",
    "- *Grid 13x13 limitat:* doua obiecte cu acelasi centru de celula → unul pierdut (decoder simplificat)."
))

cells.append(md("## 8. Export requirements.txt"))

cells.append(code(
    "import subprocess, sys\n",
    "try:\n",
    "    out = subprocess.run([sys.executable, '-m', 'pip', 'freeze'],\n",
    "                          capture_output=True, text=True, timeout=60)\n",
    "    with open('requirements.txt', 'w', encoding='utf-8') as f:\n",
    "        f.write(out.stdout)\n",
    "    print(f'requirements.txt saved ({len(out.stdout.splitlines())} packages)')\n",
    "except Exception as e:\n",
    "    print(f'[!] pip freeze a esuat: {e}. Foloseste fisierul existent.')"
))

cells.append(md(
    "---\n\n## Concluzii\n\n",
    "Am construit un sistem complet de **object detection** pentru Aquarium Dataset cu:\n",
    "1. **Pipeline reproducibil:** seed fix, split oficial, augmentari bbox-safe.\n",
    "2. **Arhitectura multi-head:** ResNet-50 + Detection Head + Auxiliary Classification Head, loss combinat.\n",
    "3. **Fine-tuning in doua faze:** frozen → unfrozen backbone, LR diferentiat, CosineAnnealing.\n",
    "4. **Evaluare comparativa:** ablation cu/fara auxiliary head, mAP@0.5, vizualizari predictii.\n\n",
    "**Limitari:** dataset mic (~640 imagini), decoder simplificat fara anchor boxes, val set redus.  \n",
    "**Directii viitoare:** anchor boxes multiple/celula, Feature Pyramid Network pentru obiecte mici, focal loss pentru clase dezechilibrate."
))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"}
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent / "notebook.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook scris ({len(cells)} celule): {out}")
