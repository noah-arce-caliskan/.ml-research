import os
import shutil
import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser() # closed set test/train splitter
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--out_train', required=True)
    parser.add_argument('--out_test', required=True)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--min_images', type=int, default=0) # min amt of images for a class to be included
    return parser.parse_args()

def split_class_images(class_dir, test_ratio, rng):
    imgs = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
    rng.shuffle(imgs)
    n = len(imgs)
    n_test = int(n * test_ratio)
    if n_test < 1 and n >= 2: n_test = 1
    test_imgs = imgs[:n_test]
    train_imgs = imgs[n_test:]
    return train_imgs, test_imgs


def copy_images(images, src_dir, dst_root):
    class_name = os.path.basename(src_dir)
    dst_dir = os.path.join(dst_root, class_name)
    os.makedirs(dst_dir, exist_ok=True)
    for img in images:
        src_path = os.path.join(src_dir, img)
        dst_path = os.path.join(dst_dir, img)
        shutil.copy2(src_path, dst_path)


args = parse_args()
random.seed(args.seed)
rng = random.Random(args.seed)

os.makedirs(args.out_train, exist_ok=True)
os.makedirs(args.out_test, exist_ok=True)

identities = [d for d in os.listdir(args.dataset) if os.path.isdir(os.path.join(args.dataset, d))]

if args.min_images > 0:
    identities = [id for id in identities if len(os.listdir(os.path.join(args.dataset, id))) >= args.min_images]

for identity in identities:
    class_dir = os.path.join(args.dataset, identity)
    train_imgs, test_imgs = split_class_images(class_dir, args.test_ratio, rng)
    copy_images(train_imgs, class_dir, args.out_train)
    if test_imgs: copy_images(test_imgs, class_dir, args.out_test)
