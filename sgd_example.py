"""
SGD Optimizer Example from CS336 Assignment 1

This implements a variation of SGD where the learning rate decays over training:
    θ_{t+1} = θ_t - α / sqrt(t+1) * ∇L(θ_t; B_t)
"""

import math
from collections.abc import Callable
from typing import Optional

import torch


class SGD(torch.optim.Optimizer):
    """SGD optimizer with decaying learning rate."""

    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p
                t = state.get(
                    "t", 0
                )  # Get iteration number from the state, or initial value
                grad = p.grad.data  # Get the gradient of loss with respect to p

                # Update weight tensor in-place
                p.data -= lr / math.sqrt(t + 1) * grad

                # Increment iteration number
                state["t"] = t + 1

        return loss


def run_training(lr: float, num_iterations: int = 100):
    """Run training with given learning rate and return loss values."""
    weights = torch.nn.Parameter(5 * torch.rand((10, 10)))
    opt = SGD([weights], lr=lr)

    losses = []
    for t in range(num_iterations):
        opt.zero_grad()  # Reset the gradients for all learnable parameters
        loss = (weights**2).mean()  # Compute a scalar loss value
        losses.append(loss.cpu().item())
        loss.backward()  # Run backward pass, which computes gradients
        opt.step()  # Run optimizer step

    return losses


if __name__ == "__main__":
    # Run the example with default learning rate
    print("=" * 60)
    print("SGD Example with lr=1")
    print("=" * 60)
    losses = run_training(lr=1.0, num_iterations=100)
    print(f"Initial loss: {losses[0]:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")

    # Test different learning rates as per the problem
    print("\n" + "=" * 60)
    print("Learning Rate Tuning (10 iterations each)")
    print("=" * 60)

    learning_rates = [1e1, 1e2, 1e3]  # 10, 100, 1000

    for lr in learning_rates:
        print(f"\nLearning rate: {lr:.0e}")
        losses = run_training(lr=lr, num_iterations=10)
        print(losses)
        print(f"  Initial loss: {losses[0]:.6f}")
        print(f"  Final loss: {losses[-1]:.6f}")
        if losses[-1] > losses[0]:
            print("  Result: Loss INCREASED (diverged)")
        elif losses[-1] < losses[0]:
            print("  Result: Loss decreased")
        else:
            print("  Result: Loss unchanged")
