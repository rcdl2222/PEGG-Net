import os 
import sys
import argparse
import logging

from utils.data import get_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Process GG-CNN grasp maps")

    # Network
    parser.add_argument('--input-size', type=int, default=480, help='Size of input image (must be a multiple of 8)')

    # Dataset & Training
    parser.add_argument('--dataset', type=str, help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str, help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=0, help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1, help='Use RGB image for training (0/1)')
    parser.add_argument('--random-seed', type=int, default=10, help='Random seed for dataset shuffling.')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)

    train_dataset = Dataset(file_path=args.dataset_path,
                    output_size=args.input_size,
                    random_seed=args.random_seed,
                    include_depth=args.use_depth, 
                    include_rgb=args.use_rgb)
    
    train_dataset.save_grasp_map_images()
        
if __name__ == "__main__":
    main()