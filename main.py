import argparse
import os

import torch
from tqdm import tqdm

from datasets.daphnet import get_dataloader
from networks.diffusion import CSDI
from utils.misc import load_config, set_seed


def train(net, train_dl, optimizer, lr_scheduler, epoch):
    avg_loss = 0
    net.train()
    with tqdm(train_dl) as it:
        for batch_idx, data in enumerate(it, start=1):
            optimizer.zero_grad()

            loss = net(data, train=True)
            loss.backward()
            avg_loss += loss.item()
            optimizer.step()
            it.set_postfix(
                ordered_dict={
                    "avg_epoch_loss": avg_loss / batch_idx,
                    "epoch": epoch,
                },
                refresh=False,
            )
        lr_scheduler.step()
    return avg_loss / len(train_dl)


def validate(net, val_dl, n_samples=20):
    with torch.no_grad():
        net.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        with tqdm(val_dl) as it:
            for batch_idx, data in enumerate(it, start=1):
                output = net.evaluate(data, n_samples)
                samples, c_target, eval_points, _, _ = output
                samples = samples.permute(0, 1, 3, 2)  # (B, nsample, L, K)
                c_target = c_target.permute(0, 2, 1)  # (B, L, K)
                eval_points = eval_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)

                mse_current = ((samples_median.values - c_target) * eval_points) ** 2
                mae_current = torch.abs(
                    (samples_median.values - c_target) * eval_points
                )

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "mse_total": mse_total / evalpoints_total,
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_idx,
                    },
                    refresh=True,
                )
    return mse_total / evalpoints_total


def main(config):
    set_seed(config["etc"]["seed"])
    device = torch.device(
        config["etc"]["device"] if torch.cuda.is_available() else "cpu"
    )

    if not os.path.exists(config["training"]["work_dir"]):
        os.makedirs(config["training"]["work_dir"])

    train_dl, val_dl, test_0_dl, test_1_dl = get_dataloader(config)

    net = CSDI(config, device).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=config["training"]["lr"], weight_decay=1e-6)

    p1 = int(0.75 * config["training"]["epochs"])
    p2 = int(0.9 * config["training"]["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    minimum_mse_score = 10000
    patience = 0

    with tqdm(range(1, config["training"]["epochs"] + 1)) as it:
        for epoch in it:
            train_mse_score = train(net, train_dl, optimizer, lr_scheduler, epoch)
            val_mse_score = validate(net, val_dl, 1)
            it.set_postfix(
                ordered_dict={
                    "train mse": train_mse_score,
                    "val mse": val_mse_score,
                    "epoch": epoch,
                },
                refresh=False,
            )

            if val_mse_score < minimum_mse_score:
                patience = 0
                minimum_mse_score = val_mse_score
                print("Minimum MSE score reached: ", minimum_mse_score)
                output_path = f"{config['training']['work_dir']}/best-model.pth"
                # trunk-ignore(bandit/B614)
                torch.save(net.state_dict(), output_path)
            else:
                patience += 1

            if patience > 5:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="config_baseline.json")
    args = parser.parse_args()

    config = load_config(args.config)

    main(config)
