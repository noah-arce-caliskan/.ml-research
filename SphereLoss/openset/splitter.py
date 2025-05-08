import os, shutil, argparse, random

def parse_args():
    p = argparse.ArgumentParser() # openset data train/test/val splitter
    p.add_argument('--dataset', required=True)
    p.add_argument('--out_train', required=True)
    p.add_argument('--out_val', required=True)
    p.add_argument('--out_test', required=True)
    p.add_argument('--train_ratio',   type=float, default=0.6)
    p.add_argument('--val_ratio',     type=float, default=0.1)
    p.add_argument('--min_total_images', type=int, default=5) # min img per class to be used
    p.add_argument('--min_test_images',  type=int, default=2) # min img in each test class
    return p.parse_args()

def make_dir(path): 
    os.makedirs(path, exist_ok=True)

def copy_identity(identity, src_root, dst_root):
    src_dir = os.path.join(src_root, identity)
    dst_dir = os.path.join(dst_root, identity)
    make_dir(dst_dir)
    for fname in os.listdir(src_dir):
        src_path = os.path.join(src_dir, fname)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, os.path.join(dst_dir, fname))

args = parse_args()
random.seed(args.seed)

make_dir(args.out_train)
make_dir(args.out_val)
make_dir(args.out_test)

all_ids = [d for d in os.listdir(args.dataset) if os.path.isdir(os.path.join(args.dataset, d))]

eligible = []
for identity in all_ids:
    img_dir = os.path.join(args.dataset, identity)
    imgs = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
    if len(imgs) >= args.min_total_images: eligible.append(identity)
    else: copy_identity(identity, args.dataset, args.out_train)

random.shuffle(eligible)
n = len(eligible)
n_train = int(n * args.train_ratio)
n_val = int(n * args.val_ratio)
train_ids = set(eligible[:n_train])
val_ids = set(eligible[n_train:n_train + n_val])
test_ids = set(eligible[n_train + n_val:])

# copy into each split
for identity in train_ids: copy_identity(identity, args.dataset, args.out_train)
for identity in val_ids: copy_identity(identity, args.dataset, args.out_val)
for identity in test_ids:
    img_dir = os.path.join(args.dataset, identity)
    imgs = [f for f in os.listdir(img_dir)
            if os.path.isfile(os.path.join(img_dir, f))]
    if len(imgs) >= args.min_test_images:
        copy_identity(identity, args.dataset, args.out_test)
    else:
        copy_identity(identity, args.dataset, args.out_train) # push to train if too little to test