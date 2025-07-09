
from __future__ import annotations

import argparse
import importlib
import time
from inspect import signature
from pathlib import Path
from typing import Type

import matplotlib.pyplot as plt  # ← 使用默认后端，便于 plt.show()
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid



def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments."""
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train / Evaluate a GNN on the Cora dataset",
    )

    # ── 通用训练超参数 ───────────────────────────────────────────────────────────
    p.add_argument("--root", type=str, default="data", help="数据缓存目录")
    p.add_argument("--hidden_dim", type=int, default=8, help="隐层维度")
    p.add_argument("--heads", type=int, default=16, help="注意力头数（部分模型需要）")
    p.add_argument("--epochs", type=int, default=200, help="训练轮数")
    p.add_argument("--lr", type=float, default=0.005, help="学习率")
    p.add_argument("--weight_decay", type=float, default=5e-4, help="L2 正则项")
    p.add_argument("--dropout", type=float, default=0.6, help="Dropout 概率")
    p.add_argument("--no_cuda", action="store_true", help="禁用 GPU，仅用 CPU")

    # ── 模型相关 ────────────────────────────────────────────────────────────────
    p.add_argument("--model", type=str, default="gin", help="模型名，如 gat、gcn、graphsage")
    p.add_argument(
        "--impl",
        type=str,
        default="no_pyg",
        choices=["pyg", "no_pyg"],
        help="实现版本（基于 torch_geometric 或纯 PyTorch）",
    )

    # ── 保存路径 ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="模型权重保存路径 (.pt)，留空则自动命名到 logs/",
    )
    return p.parse_args()


def get_model_class(model: str, impl: str) -> Type:
    """根据模型名和实现版本动态导入并返回模型类."""
    module_name = f"models.{impl}_version.{model}_model"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise ValueError(
            f"找不到模块 '{module_name}'，请确认实现是否存在或拼写正确。"
        ) from e

    # 优先查找同名大写类（如 GAT / GCN），否则退回通用类名 `Model`
    for cls_name in (model.upper(), "Model"):
        if hasattr(module, cls_name):
            return getattr(module, cls_name)
    raise ValueError(
        f"模块 '{module_name}' 中未找到期望的模型类 ({model.upper()} / Model)。"
    )

# --------------------------------------------------------------------------- #
#                         Train / Eval & Helper Funcs                         #
# --------------------------------------------------------------------------- #

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy (%)."""
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).sum().item()
    return 100.0 * correct / labels.size(0)


def train(model, data, optimizer) -> float:
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data) -> dict[str, float]:
    model.eval()
    out = model(data.x, data.edge_index)
    return {
        split: accuracy(out[data[f"{split}_mask"]], data.y[data[f"{split}_mask"]])
        for split in ("train", "val", "test")
    }

# --------------------------------------------------------------------------- #
#                                   Runner                                    #
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    device = "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda"
    print(f"🖥  Using device: {device}")

    # ------------------------ 数据 ------------------------ #
    dataset = Planetoid(root=str(Path(args.root).resolve()), name="Cora")
    data = dataset[0].to(device)

    # ----------------------- 模型 ------------------------ #
    ModelClass = get_model_class(args.model, args.impl)
    model_kwargs = {
        "in_dim": dataset.num_node_features,
        "hidden_dim": args.hidden_dim,
        "out_dim": dataset.num_classes,
        "heads": args.heads,
        "dropout": args.dropout,
    }
    filtered_kwargs = {k: v for k, v in model_kwargs.items() if k in signature(ModelClass).parameters}
    model = ModelClass(**filtered_kwargs).to(device)
    print(f"Model: {ModelClass.__name__}\n    params: {filtered_kwargs}")
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --------------------- 优化器 ------------------------ #
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --------------------- 记录列表 ---------------------- #
    train_acc_list: list[float] = []
    val_acc_list: list[float] = []
    test_acc_list: list[float] = []

    # -------------------- 保存路径 ----------------------- #
    if args.save_path is None:
        save_path = Path("logs") / f"best_{args.model}_{args.impl}.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        save_path = Path(args.save_path).expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint path: {save_path}\n")

    # -------------------- 训练循环 ----------------------- #
    best_val_acc = best_test_acc = 0.0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        loss = train(model, data, optimizer)
        accs = evaluate(model, data)
        t_ms = (time.perf_counter() - t0) * 1000  # ms

        train_acc_list.append(accs["train"])
        val_acc_list.append(accs["val"])
        test_acc_list.append(accs["test"])

        if accs["val"] > best_val_acc:
            best_val_acc, best_test_acc, best_epoch = accs["val"], accs["test"], epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": best_val_acc,
                    "test_acc": best_test_acc,
                    "model_class": ModelClass.__name__,
                    "args": vars(args),
                },
                save_path,
            )
            print(f"[Checkpoint] Epoch {epoch:03d} | Val {best_val_acc:.2f}% | Test {best_test_acc:.2f}%")

        if epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d} | Loss {loss:.4f} | Train {accs['train']:.2f}% | "
                f"Val {accs['val']:.2f}% | Test {accs['test']:.2f}% | Time {t_ms:.1f} ms"
            )

    # ------------------------ 绘图 ------------------------ #
    plt.figure(figsize=(8, 5))
    x = range(1, args.epochs + 1)
    plt.plot(x, train_acc_list, label="Train")
    plt.plot(x, val_acc_list, label="Validation")
    plt.plot(x, test_acc_list, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy vs Epoch ({args.model}, {args.impl})")
    plt.legend()
    plt.tight_layout()

    plot_path = Path("logs") / f"acc_curve_{args.model}_{args.impl}.png"
    plt.savefig(plot_path, dpi=300)
    print(f"📈 Saved accuracy curve: {plot_path}")

    # —— 新增：直接展示窗口 —— #
    try:
        plt.show()  # 若在无显示环境则忽略 / 报错
    except Exception as e:
        print(f"Couldn't display the plot (likely a headless environment): {e}")
    finally:
        plt.close()

    # ------------------ 保存最佳结果到 txt ------------------ #
    best_txt_path = Path("logs") / "best_results.txt"
    with best_txt_path.open("a") as f:
        f.write(f"{args.model}\t{args.impl}\t{best_val_acc:.4f}\t{best_test_acc:.4f}\n")
    print(f"\nBest Val Acc: {best_val_acc:.2f}% (epoch {best_epoch}) | Test Acc @ Best Val: {best_test_acc:.2f}%")
    print(f"Best metrics appended to: {best_txt_path}")
    print(f"Saved checkpoint: {save_path}")


# --------------------------------------------------------------------------- #
#                                   Main                                      #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
