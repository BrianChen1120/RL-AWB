import numpy as np
import random
import torch
from stable_baselines3.common.base_class import BaseAlgorithm
import os
import csv
import time
import argparse
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from env import IlluminantEstimationEnv

SEED = 531


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    BaseAlgorithm.set_random_seed(seed)


def make_env(seed=0, dataset_name='NCC', init_action=None):
    def _init():
        env = IlluminantEstimationEnv(max_steps=100, training=False, dataset_name=dataset_name, init_action=init_action)
        env.seed(seed)
        env.action_space.seed(seed)
        return env

    return _init


def get_init_action(dataset_name):
    init_actions = {
        'NCC': [0.50, 3.5, 0.045, 0.3, 10, 3, 0, 0.90, 7, 3],
        'LEVI': [0.5, 2, 0.025, 0.35, 10, 3, 0, 0.9, 7, 3],
        'Gehler': [0.5, 2, 0.025, 0.35, 10, 3, 0, 0.9, 7, 3]
    }
    return init_actions.get(dataset_name)


def get_dataset_size(dataset_name):
    dataset_sizes = {
        'NCC': 514,
        'LEVI': 701,
        'Gehler': 560
    }
    return dataset_sizes.get(dataset_name, 514)


def evaluate(errors):
    errors = np.sort(errors)
    n = len(errors)
    f05 = errors[int(np.floor(0.5 * n)) - 1]
    f025 = errors[int(np.floor(0.25 * n)) - 1]
    f075 = errors[int(np.floor(0.75 * n)) - 1]
    med = np.median(errors)
    men = np.mean(errors)
    trimean = 0.25 * (f025 + 2 * f05 + f075)
    bst25 = np.mean(errors[:int(np.floor(0.25 * n))])
    wst25 = np.mean(errors[int(np.floor(0.75 * n) - 1):])

    return men, med, trimean, bst25, wst25


def inference(model_path, dataset_name, output_dir='./results', image_idx=None, seed=SEED):
    set_seed(seed)

    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading SAC model from {model_path}")
    model = SAC.load(model_path, device="cpu", seed=seed)

    # Get init_action for dataset
    init_action = get_init_action(dataset_name)
    if init_action is None:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose 'NCC', 'LEVI', or 'Gehler'.")

    # Create environment
    env = DummyVecEnv([make_env(seed=seed, dataset_name=dataset_name, init_action=init_action)])
    test_env = env.envs[0]

    # Prepare output files
    os.makedirs(output_dir, exist_ok=True)

    # Determine which images to process
    if image_idx is not None:
        image_indices = [image_idx]
        result_file = os.path.join(output_dir, f"{dataset_name.lower()}_image_{image_idx}.csv")
        summary_file = os.path.join(output_dir, f"{dataset_name.lower()}_image_{image_idx}_summary.txt")
    else:
        dataset_size = get_dataset_size(dataset_name)
        image_indices = range(1, dataset_size)
        result_file = os.path.join(output_dir, f"{dataset_name.lower()}_inference_detailed.csv")
        summary_file = os.path.join(output_dir, f"{dataset_name.lower()}_inference_summary.txt")

    # Write CSV results
    with open(result_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'image_idx', 'step', 'arr', 'arr_rep',
            'EvaLum_r', 'EvaLum_g', 'EvaLum_b',
            'action0', 'action1'
        ])

        start_time = time.time()

        for idx in image_indices:
            reset_ret = test_env.reset(image_index=idx)
            obs = reset_ret[0] if isinstance(reset_ret, (tuple, list)) else reset_ret

            # Step 0
            a0, a1 = np.asarray(test_env.init_action).flatten()[:2]
            writer.writerow([idx, 0, reset_ret[1]["arr"], reset_ret[1]["arr_rep"],
                             *reset_ret[1]['EvaLum'].tolist(), a0, a1])

            # Run steps
            for step in range(1, test_env.max_steps + 1):
                if not reset_ret[1]["for_RL"]:
                    print(f'Image {idx}, step {step}: env not for RL! arr: {reset_ret[-1]["arr"]:.4f}')
                    break

                action, _ = model.predict(obs, deterministic=True)
                step_ret = test_env.step(action)
                obs = step_ret[0] if isinstance(step_ret, (tuple, list)) else step_ret

                a0, a1 = np.asarray(action).flatten()[:2]
                writer.writerow([idx, step, step_ret[-1]["arr"], step_ret[-1]["arr_rep"],
                                 *step_ret[-1]['EvaLum'].tolist(), a0, a1])

                if step_ret[2]:
                    print(f'Image {idx}, step {step}: done! reward: {step_ret[1]:.5f}, '
                          f'init arr: {test_env.init_arr:.4f}, final arr: {step_ret[-1]["arr"]:.4f}')
                    break

        elapsed = time.time() - start_time

    print(f"Inference completed in {elapsed:.2f} sec.")
    print(f"Results saved to {result_file}")

    # Compute statistics if processing full dataset
    if image_idx is None:
        print("\n" + "=" * 60)
        print("PERFORMANCE ANALYSIS")
        print("=" * 60)

        df = pd.read_csv(result_file)

        # Get the last step for each image (in case of early termination)
        last_step_data = df.loc[df.groupby('image_idx')['step'].idxmax()]
        start_data = df[df['step'] == 0]

        # Merge to ensure same order
        start_arr = start_data.set_index('image_idx')['arr'].sort_index()
        end_arr = last_step_data.set_index('image_idx')['arr'].sort_index()
        start_arr_rep = start_data.set_index('image_idx')['arr_rep'].sort_index()
        end_arr_rep = last_step_data.set_index('image_idx')['arr_rep'].sort_index()

        mean_perf, median_perf, trimean_perf, bst25, wst25 = evaluate(np.array(start_arr))
        print("\nStart Performance (binary errors) [median, mean, trimean, best25%, worst25%]:")
        print(f"  {median_perf:.4f}, {mean_perf:.4f}, {trimean_perf:.4f}, {bst25:.4f}, {wst25:.4f}")

        mean_rep, median_rep, trimean_rep, bst25_rep, wst25_rep = evaluate(np.array(start_arr_rep))
        print("Start Performance (rep errors) [median, mean, trimean, best25%, worst25%]:")
        print(f"  {median_rep:.4f}, {mean_rep:.4f}, {trimean_rep:.4f}, {bst25_rep:.4f}, {wst25_rep:.4f}")

        end_mean_perf, end_median_perf, end_trimean_perf, end_bst25, end_wst25 = evaluate(np.array(end_arr))
        print("\nEnd Performance (binary errors) [median, mean, trimean, best25%, worst25%]:")
        print(f"  {end_median_perf:.4f}, {end_mean_perf:.4f}, {end_trimean_perf:.4f}, {end_bst25:.4f}, {end_wst25:.4f}")

        end_mean_rep, end_median_rep, end_trimean_rep, end_bst25_rep, end_wst25_rep = evaluate(np.array(end_arr_rep))
        print("End Performance (rep errors) [median, mean, trimean, best25%, worst25%]:")
        print(
            f"  {end_median_rep:.4f}, {end_mean_rep:.4f}, {end_trimean_rep:.4f}, {end_bst25_rep:.4f}, {end_wst25_rep:.4f}")

        increased = (end_arr < start_arr).sum()
        decreased = (end_arr > start_arr).sum()
        equal = (end_arr == start_arr).sum()

        print(f"\nImprovement Summary:")
        print(f"  Improved: {increased} images")
        print(f"  Worsened: {decreased} images")
        print(f"  Unchanged: {equal} images")
        print("=" * 60 + "\n")

    # Write summary
    with open(summary_file, 'w') as f:
        f.write(f"Dataset: {dataset_name}\n")
        if image_idx is not None:
            f.write(f"Image Index: {image_idx}\n")
        else:
            f.write(f"Images: 1-{get_dataset_size(dataset_name) - 1} (total: {get_dataset_size(dataset_name) - 1})\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Inference Time: {elapsed:.2f} sec\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Init Action: {init_action}\n\n")
        f.write(f"Detailed results saved to: {result_file}\n")
        f.write(f"Run inference_analysis.py on the CSV file to get performance metrics.\n")

    print(f"Summary saved to {summary_file}")
    return result_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference with SAC model on illuminant estimation task')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model file (.zip)')
    parser.add_argument('--dataset', type=str, required=True, choices=['NCC', 'LEVI', 'Gehler'],
                        help='Dataset name')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--image_idx', type=int, default=None, help='Single image index to process')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')

    args = parser.parse_args()

    inference(
        model_path=args.model_path,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        image_idx=args.image_idx,
        seed=args.seed
    )