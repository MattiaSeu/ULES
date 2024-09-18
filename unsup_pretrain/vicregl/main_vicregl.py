# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
import argparse
import json
import os
import sys
import time

from collections import defaultdict
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from datasets import build_loader
from optimizers import build_optimizer
import utils

import tqdm


def get_arguments():
    parser = argparse.ArgumentParser(description="Pretraining with VICRegL", add_help=False)

    # Checkpoints and Logs
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--log-tensors-interval", type=int, default=30)
    parser.add_argument("--checkpoint-freq", type=int, default=1)

    # Data
    parser.add_argument("--dataset", type=str, default="imagenet1k")
    parser.add_argument("--dataset_from_numpy", action="store_true")
    parser.add_argument("--size-crops", type=int, nargs="+", default=[224, 96])
    parser.add_argument("--num-crops", type=int, nargs="+", default=[2, 6])
    parser.add_argument("--min_scale_crops", type=float, nargs="+", default=[0.4, 0.08])
    parser.add_argument("--max_scale_crops", type=float, nargs="+", default=[1, 0.4])
    parser.add_argument("--no-flip-grid", type=int, default=1)

    # Model
    parser.add_argument("--arch", type=str, default="convnext_small")
    parser.add_argument("--single_backbone", action="store_true")
    parser.add_argument("--fusion_type", type=str, default="add")
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--layer-scale-init-value", type=float, default=0.0)
    parser.add_argument("--mlp", default="8192-8192-8192")
    parser.add_argument("--maps-mlp", default="512-512-512")

    # Loss Function
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument(
        "--num_matches",
        type=int,
        nargs="+",
        default=[20, 4],
        help="Number of spatial matches in a feature map",
    )
    parser.add_argument("--l2_all_matches", type=int, default=1)
    parser.add_argument("--inv-coeff", type=float, default=25.0)
    parser.add_argument("--var-coeff", type=float, default=25.0)
    parser.add_argument("--cov-coeff", type=float, default=1.0)
    parser.add_argument("--fast-vc-reg", type=int, default=0)

    # Optimization
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--base-lr", type=float, default=0.0005)
    parser.add_argument("--end-lr-ratio", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.05)

    # Evaluation
    parser.add_argument("--val-batch-size", type=int, default=-1)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--eval-freq", type=int, default=10)
    parser.add_argument("--maps-lr-ratio", type=float, default=0.1)

    # Running
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # Distributed
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    # torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    # init_distributed_mode(args)
    print(args)
    global double_backbone
    double_backbone = not args.single_backbone
    gpu = torch.device(args.device)

    # Ensures that stats_file is initialized when calling evaluate(),
    # even if only the rank 0 process will use it
    stats_file = None
    if args.local_rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
        dir = os.getcwd().split("/")[-1]
        print(" ".join([dir] + sys.argv))
        print(" ".join([dir] + sys.argv), file=stats_file)

    # args.stats_file = stats_file

    train_loader= build_loader(args, is_train=True)
    if args.evaluate:
        val_loader, _ = build_loader(args, is_train=False)

    model = DualVICRegL(args).cuda(gpu)
    print(model)
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    optimizer = build_optimizer(args, model)

    # if os.path.exists(str(args.exp_dir) + "model_resnet50.pth"):
    if os.path.exists("experiments/model_resnet50.pth"):
        print("resuming from checkpoint")
        ckpt = torch.load(args.exp_dir / "model_resnet50.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0

    if args.evaluate_only:
        evaluate(model, {}, val_loader, args, 0, 0.0, stats_file, gpu)
        exit(1)

    start_time = last_logging = time.time()
    scaler = torch.cuda.amp.GradScaler()
    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            # train_sampler.set_epoch(epoch)
            # breakpoint()
            step = 0
            n_steps = len(train_loader.dataset) // args.batch_size
            prog_bar = tqdm.tqdm(train_loader,  desc=f"Epoch{epoch+1}/{args.epochs}") # wrap loader inside progress bar
            for step, inputs in enumerate(prog_bar, start=epoch * len(train_loader)):
                lr = utils.learning_schedule(
                    global_step=step,
                    batch_size=args.batch_size,
                    base_lr=args.base_lr,
                    end_lr_ratio=args.end_lr_ratio,
                    total_steps=args.epochs * len(train_loader.dataset) // args.batch_size,
                    warmup_steps=args.warmup_epochs
                    * len(train_loader.dataset)
                    // args.batch_size,
                )
                for g in optimizer.param_groups:
                    if "__MAPS_TOKEN__" in g.keys():
                        g["lr"] = lr * args.maps_lr_ratio
                    else:
                        g["lr"] = lr
                # if epoch % 4 == 0:
                optimizer.zero_grad()
                if args.fp16:
                    with torch.cuda.amp.autocast():
                        loss, logs = model.forward(make_inputs(inputs, gpu))
                    # if epoch % 4 == 0:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss, logs = model.forward(make_inputs(inputs, gpu))
                    loss.backward()
                    optimizer.step()

                # logging
                # for v in logs.values():
                #     torch.distributed.reduce(v.div_(args.world_size), 0)
                current_time = time.time()
                prog_bar.set_postfix({"loss": loss.item()}) # update progress bar

                if (
                    # args.rank == 0
                    current_time - last_logging > args.log_tensors_interval
                ):
                    logs = {key: utils.round_log(key, value) for key, value in logs.items()}
                    stats = dict(
                        ep=epoch,
                        st=step,
                        lr=lr,
                        t=format(int(current_time - start_time), "09d"),
                    )
                    stats.update(logs)
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
                    last_logging = current_time
            extra_str = "_db_v3" if double_backbone else "_sb_v3"
            utils.checkpoint(args, epoch + 1, step, model, optimizer, name = args.arch + "_" + args.dataset + extra_str)

            # evaluate
            if (epoch + 1) % args.eval_freq == 0:
                if args.evaluate:
                    evaluate(model, logs, val_loader, args, epoch, lr, stats_file, gpu)
    except KeyboardInterrupt:
        extra_str = "_db_v3" if double_backbone else "_sb_v3"
        utils.checkpoint(args, epoch, step, model, optimizer, name = args.arch + "_" + args.dataset + extra_str)

def evaluate(model, logs, val_loader, args, epoch, lr, stats_file, gpu):
    model.eval()

    iters = 0
    cumulative_logs = defaultdict(float)
    for inputs in val_loader:
        with torch.no_grad():
            if args.fp16:
                with torch.cuda.amp.autocast():
                    loss, logs = model.forward(make_inputs(inputs, gpu), is_val=True)
            else:
                loss, logs = model.forward(make_inputs(inputs, gpu), is_val=True)

        # logging
        # for v in logs.values():
        #     torch.distributed.reduce(v.div_(args.world_size), 0)
        for key, value in logs.items():
            cumulative_logs[key] += value.item()
        iters += 1

    if args.rank == 0:
        stats = dict(ep=epoch, lr=lr)
        cumulative_logs = {
            key: utils.round_log(key, value, item=False, iters=iters)
            for key, value in cumulative_logs.items()
        }
        stats.update(cumulative_logs)
        print("Val: ", json.dumps(stats))
        print("Val: ", json.dumps(stats), file=stats_file)


def make_inputs(inputs, gpu):
    # breakpoint()
    if isinstance(inputs[0][1][1], list):
        (val_view, (views, locations)) = inputs
        return dict(
            val_view=val_view.cuda(gpu, non_blocking=True),
            views=[view.cuda(gpu, non_blocking=True) for view in views],
            locations=[location.cuda(gpu, non_blocking=True) for location in locations],
        )
    if double_backbone:
        (views1, views2), locations = inputs
    else:
        (views, locations) = inputs

    if double_backbone:
        return dict(
             views=[(view1.cuda(gpu, non_blocking=True), view2.cuda(gpu, non_blocking=True)) \
                    for view1, view2 in zip(views1, views2)],
            locations=[location.cuda(gpu, non_blocking=True) for location in locations],
        )
    else:
        return dict(
            views=[view.cuda(gpu, non_blocking=True) for view in views[0]],
            locations=[location.cuda(gpu, non_blocking=True) for location in locations],
        )

class DualVICRegL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding_dim = int(args.mlp.split("-")[-1])
        double_backbone = not args.single_backbone


        import convnext
        import resnet
        if double_backbone:
            self.backbone1, self.representation_dim1 = self._create_backbone(args.arch, ch=3)
            self.backbone2, self.representation_dim2 = self._create_backbone(args.  arch)
        else:
            self.backbone, self.representation_dim = self._create_backbone(args.arch, ch=3)

        norm_layer = "layer_norm" if "convnext" in args.arch else "batch_norm"

        # Feature fusion options
        self.fusion_type = args.fusion_type
        if double_backbone:
            if self.fusion_type == "concat":
                self.fused_dim = self.representation_dim1 + self.representation_dim2
            elif self.fusion_type in ["add", "multiply"]:
                assert self.representation_dim1 == self.representation_dim2, "Dimensions must match for add/multiply fusion"
                self.fused_dim = self.representation_dim1
            elif self.fusion_type == "attention":
                self.attention = nn.MultiheadAttention(self.representation_dim1, num_heads=8)
                self.fused_dim = self.representation_dim1
        else:
            self.fused_dim = self.representation_dim

        if self.args.alpha < 1.0:
            self.maps_projector = utils.MLP(args.maps_mlp, self.fused_dim, norm_layer)

        if self.args.alpha > 0.0:
            self.projector = utils.MLP(args.mlp, self.fused_dim, norm_layer)



    def _create_backbone(self, arch, ch: int = 1):
        if "convnext" in arch:
            import convnext
            backbone, representation_dim = convnext.__dict__[arch](
                drop_path_rate=self.args.drop_path_rate,
                layer_scale_init_value=self.args.layer_scale_init_value,
            )
        elif "resnet" in arch:
            import resnet
            backbone, representation_dim = resnet.__dict__[arch](zero_init_residual=True)
            if ch != 3:
                backbone.conv1 = nn.Conv2d(ch, 64, kernel_size=7, stride=2, padding=2, bias=False)
        else:
            raise Exception(f"Unsupported backbone {arch}.")
        return backbone, representation_dim

    def fuse_features(self, x1, x2):
        if self.fusion_type == "concat":
            return torch.cat([x1, x2], dim=-1)
        elif self.fusion_type == "add":
            return x1 + x2
        elif self.fusion_type == "multiply":
            return x1 * x2
        elif self.fusion_type == "attention":
            return self.attention(x1, x2, x2)[0]

    def _vicreg_loss(self, x, y):
        repr_loss = self.args.inv_coeff * F.mse_loss(x, y)

        x = utils.gather_center(x)
        y = utils.gather_center(y)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = self.args.var_coeff * (
            torch.mean(F.relu(1.0 - std_x)) / 2 + torch.mean(F.relu(1.0 - std_y)) / 2
        )

        x = x.permute((1, 0, 2))
        y = y.permute((1, 0, 2))

        *_, sample_size, num_channels = x.shape
        non_diag_mask = ~torch.eye(num_channels, device=x.device, dtype=torch.bool)
        # Center features
        # centered.shape = NC
        x = x - x.mean(dim=-2, keepdim=True)
        y = y - y.mean(dim=-2, keepdim=True)

        cov_x = torch.einsum("...nc,...nd->...cd", x, x) / (sample_size - 1)
        cov_y = torch.einsum("...nc,...nd->...cd", y, y) / (sample_size - 1)
        cov_loss = (cov_x[..., non_diag_mask].pow(2).sum(-1) / num_channels) / 2 + (
            cov_y[..., non_diag_mask].pow(2).sum(-1) / num_channels
        ) / 2
        cov_loss = cov_loss.mean()
        cov_loss = self.args.cov_coeff * cov_loss

        return repr_loss, std_loss, cov_loss

    def _local_loss(
        self, maps_1, maps_2, location_1, location_2
    ):
        inv_loss = 0.0
        var_loss = 0.0
        cov_loss = 0.0

        # L2 distance based bacthing
        if self.args.l2_all_matches:
            num_matches_on_l2 = [None, None]
        else:
            num_matches_on_l2 = self.args.num_matches

        maps_1_filtered, maps_1_nn = neirest_neighbores_on_l2(
            maps_1, maps_2, num_matches=num_matches_on_l2[0]
        )
        maps_2_filtered, maps_2_nn = neirest_neighbores_on_l2(
            maps_2, maps_1, num_matches=num_matches_on_l2[1]
        )

        if self.args.fast_vc_reg:
            inv_loss_1 = F.mse_loss(maps_1_filtered, maps_1_nn)
            inv_loss_2 = F.mse_loss(maps_2_filtered, maps_2_nn)
        else:
            inv_loss_1, var_loss_1, cov_loss_1 = self._vicreg_loss(maps_1_filtered, maps_1_nn)
            inv_loss_2, var_loss_2, cov_loss_2 = self._vicreg_loss(maps_2_filtered, maps_2_nn)
            var_loss = var_loss + (var_loss_1 / 2 + var_loss_2 / 2)
            cov_loss = cov_loss + (cov_loss_1 / 2 + cov_loss_2 / 2)

        inv_loss = inv_loss + (inv_loss_1 / 2 + inv_loss_2 / 2)

        # Location based matching
        location_1 = location_1.flatten(1, 2)
        location_2 = location_2.flatten(1, 2)

        maps_1_filtered, maps_1_nn = neirest_neighbores_on_location(
            location_1,
            location_2,
            maps_1,
            maps_2,
            num_matches=self.args.num_matches[0],
        )
        maps_2_filtered, maps_2_nn = neirest_neighbores_on_location(
            location_2,
            location_1,
            maps_2,
            maps_1,
            num_matches=self.args.num_matches[1],
        )

        if self.args.fast_vc_reg:
            inv_loss_1 = F.mse_loss(maps_1_filtered, maps_1_nn)
            inv_loss_2 = F.mse_loss(maps_2_filtered, maps_2_nn)
        else:
            inv_loss_1, var_loss_1, cov_loss_1 = self._vicreg_loss(maps_1_filtered, maps_1_nn)
            inv_loss_2, var_loss_2, cov_loss_2 = self._vicreg_loss(maps_2_filtered, maps_2_nn)
            var_loss = var_loss + (var_loss_1 / 2 + var_loss_2 / 2)
            cov_loss = cov_loss + (cov_loss_1 / 2 + cov_loss_2 / 2)

        inv_loss = inv_loss + (inv_loss_1 / 2 + inv_loss_2 / 2)

        return inv_loss, var_loss, cov_loss

    def local_loss(self, maps_embedding, locations):
        num_views = len(maps_embedding)
        inv_loss = 0.0
        var_loss = 0.0
        cov_loss = 0.0
        iter_ = 0
        for i in range(2):
            for j in np.delete(np.arange(np.sum(num_views)), i):
                inv_loss_this, var_loss_this, cov_loss_this = self._local_loss(
                    maps_embedding[i], maps_embedding[j], locations[i], locations[j],
                )
                inv_loss = inv_loss + inv_loss_this
                var_loss = var_loss + var_loss_this
                cov_loss = cov_loss + cov_loss_this
                iter_ += 1

        if self.args.fast_vc_reg:
            inv_loss = self.args.inv_coeff * inv_loss / iter_
            var_loss = 0.0
            cov_loss = 0.0
            iter_ = 0
            for i in range(num_views):
                x = utils.gather_center(maps_embedding[i])
                std_x = torch.sqrt(x.var(dim=0) + 0.0001)
                var_loss = var_loss + torch.mean(torch.relu(1.0 - std_x))
                x = x.permute(1, 0, 2)
                *_, sample_size, num_channels = x.shape
                non_diag_mask = ~torch.eye(num_channels, device=x.device, dtype=torch.bool)
                x = x - x.mean(dim=-2, keepdim=True)
                cov_x = torch.einsum("...nc,...nd->...cd", x, x) / (sample_size - 1)
                cov_loss = cov_x[..., non_diag_mask].pow(2).sum(-1) / num_channels
                cov_loss = cov_loss + cov_loss.mean()
                iter_ = iter_ + 1
            var_loss = self.args.var_coeff * var_loss / iter_
            cov_loss = self.args.cov_coeff * cov_loss / iter_
        else:
            inv_loss = inv_loss / iter_
            var_loss = var_loss / iter_
            cov_loss = cov_loss / iter_

        return inv_loss, var_loss, cov_loss

    def global_loss(self, embedding, maps=False):
        num_views = len(embedding)
        inv_loss = 0.0
        iter_ = 0
        for i in range(2):
            for j in np.delete(np.arange(np.sum(num_views)), i):
                inv_loss = inv_loss + F.mse_loss(embedding[i], embedding[j])
                iter_ = iter_ + 1
        inv_loss = self.args.inv_coeff * inv_loss / iter_

        var_loss = 0.0
        cov_loss = 0.0
        iter_ = 0
        for i in range(num_views):
            x = utils.gather_center(embedding[i])
            std_x = torch.sqrt(x.var(dim=0) + 0.0001)
            var_loss = var_loss + torch.mean(torch.relu(1.0 - std_x))
            cov_x = (x.T @ x) / (x.size(0) - 1)
            cov_loss = cov_loss + utils.off_diagonal(cov_x).pow_(2).sum().div(
                self.embedding_dim
            )
            iter_ = iter_ + 1
        var_loss = self.args.var_coeff * var_loss / iter_
        cov_loss = self.args.cov_coeff * cov_loss / iter_

        return inv_loss, var_loss, cov_loss

    def compute_metrics(self, outputs):
        def correlation_metric(x):
            x_centered = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-05)
            return torch.mean(
                utils.off_diagonal((x_centered.T @ x_centered) / (x.size(0) - 1))
            )

        def std_metric(x):
            x = F.normalize(x, p=2, dim=1)
            return torch.mean(x.std(dim=0))

        representation = torch.cat(outputs["representation"], dim=0)
        corr = correlation_metric(representation)
        stdrepr = std_metric(representation)

        if self.args.alpha > 0.0:
            embedding = torch.cat(outputs["embedding"], dim=0)
            core = correlation_metric(embedding)
            stdemb = std_metric(embedding)
            return dict(stdr=stdrepr, stde=stdemb, corr=corr, core=core)

        return dict(stdr=stdrepr, corr=corr)

    def forward_networks(self, inputs, is_val):
        outputs = {
            "representation": [],
            "embedding": [],
            "maps_embedding": []
        }
        if double_backbone:
            for x1, x2 in inputs["views"]:
                # maps1, representation1 = self.backbone1(x1)
                # maps2, representation2 = self.backbone2(x2)
                maps1, representation1 = torch_checkpoint(self.backbone1, x1)
                # print(maps1.requires_grad, representation1.requires_grad)
                maps2, representation2 = torch_checkpoint(self.backbone2, x2)
                # print(maps2.requires_grad, representation2.requires_grad)

                fused_maps = self.fuse_features(maps1, maps2)
                fused_representation = self.fuse_features(representation1, representation2)
                outputs["representation"].append(fused_representation)

                if self.args.alpha > 0.0:
                    embedding = self.projector(fused_representation)
                    outputs["embedding"].append(embedding)

                if self.args.alpha < 1.0:
                    batch_size, num_loc, _ = fused_maps.shape
                    maps_embedding = self.maps_projector(fused_maps.flatten(start_dim=0, end_dim=1))
                    maps_embedding = maps_embedding.view(batch_size, num_loc, -1)
                    outputs["maps_embedding"].append(maps_embedding)


            if is_val:
                _, representation1 = self.backbone1(inputs["val_view1"])
                _, representation2 = self.backbone2(inputs["val_view2"])
                fused_representation = self.fuse_features(representation1, representation2)

        else:
            for x in inputs["views"]:
                maps, representation = self.backbone(x)
                outputs["representation"].append(representation)

                if self.args.alpha > 0.0:
                    embedding = self.projector(representation)
                    outputs["embedding"].append(embedding)

                if self.args.alpha < 1.0:
                    batch_size, num_loc, _ = maps.shape
                    maps_embedding = self.maps_projector(maps.flatten(start_dim=0, end_dim=1))
                    maps_embedding = maps_embedding.view(batch_size, num_loc, -1)
                    outputs["maps_embedding"].append(maps_embedding)

            if is_val:
                _, representation = self.backbone(inputs["val_view"])

        return outputs

    def forward(self, inputs, is_val=False, backbone_only=False):
        if backbone_only:
            if double_backbone:
                maps1, _ = self.backbone1(inputs["views"][0])
                maps2, _ = self.backbone2(inputs["views"][1])
                return self.fuse_features(maps1, maps2)
            else:
                maps, _ = self.backbone(inputs)
                return maps

        outputs = self.forward_networks(inputs, is_val)
        with torch.no_grad():
            logs = self.compute_metrics(outputs)
        loss = 0.0

        # Global criterion
        if self.args.alpha > 0.0:
            inv_loss, var_loss, cov_loss = self.global_loss(outputs["embedding"])
            loss = loss + self.args.alpha * (inv_loss + var_loss + cov_loss)
            logs.update(dict(inv_l=inv_loss, var_l=var_loss, cov_l=cov_loss,))

        # Local criterion
        # Maps shape: B, C, H, W
        # With convnext actual maps shape is: B, H * W, C
        if self.args.alpha < 1.0:
            (
                maps_inv_loss,
                maps_var_loss,
                maps_cov_loss,
            ) = self.local_loss(
                outputs["maps_embedding"], inputs["locations"]
            )
            loss = loss + (1 - self.args.alpha) * (
                maps_inv_loss + maps_var_loss + maps_cov_loss
            )
            logs.update(
                dict(minv_l=maps_inv_loss, mvar_l=maps_var_loss, mcov_l=maps_cov_loss,)
            )

        logs.update(dict(l=loss))


        return loss, logs


def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


def neirest_neighbores(input_maps, candidate_maps, distances, num_matches):
    batch_size = input_maps.size(0)

    if num_matches is None or num_matches == -1:
        num_matches = input_maps.size(1)

    topk_values, topk_indices = distances.topk(k=1, largest=False)
    topk_values = topk_values.squeeze(-1)
    topk_indices = topk_indices.squeeze(-1)

    sorted_values, sorted_values_indices = torch.sort(topk_values, dim=1)
    sorted_indices, sorted_indices_indices = torch.sort(sorted_values_indices, dim=1)

    mask = torch.stack(
        [
            torch.where(sorted_indices_indices[i] < num_matches, True, False)
            for i in range(batch_size)
        ]
    )
    topk_indices_selected = topk_indices.masked_select(mask)
    topk_indices_selected = topk_indices_selected.reshape(batch_size, num_matches)

    indices = (
        torch.arange(0, topk_values.size(1))
        .unsqueeze(0)
        .repeat(batch_size, 1)
        .to(topk_values.device)
    )
    indices_selected = indices.masked_select(mask)
    indices_selected = indices_selected.reshape(batch_size, num_matches)

    filtered_input_maps = batched_index_select(input_maps, 1, indices_selected)
    filtered_candidate_maps = batched_index_select(
        candidate_maps, 1, topk_indices_selected
    )

    return filtered_input_maps, filtered_candidate_maps


def neirest_neighbores_on_l2(input_maps, candidate_maps, num_matches):
    """
    input_maps: (B, H * W, C)
    candidate_maps: (B, H * W, C)
    """
    distances = torch.cdist(input_maps, candidate_maps)
    return neirest_neighbores(input_maps, candidate_maps, distances, num_matches)


def neirest_neighbores_on_location(
    input_location, candidate_location, input_maps, candidate_maps, num_matches
):
    """
    input_location: (B, H * W, 2)
    candidate_location: (B, H * W, 2)
    input_maps: (B, H * W, C)
    candidate_maps: (B, H * W, C)
    """
    distances = torch.cdist(input_location, candidate_location)
    return neirest_neighbores(input_maps, candidate_maps, distances, num_matches)


def exclude_bias_and_norm(p):
    return p.ndim == 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pretraining with VICRegL", parents=[get_arguments()])
    args = parser.parse_args()
    main(args)
