"""
ProactiveBench 评测框架。

评估记忆系统的对话内主动预判能力。
三个实验条件：C0 (无记忆), C1 (被动), C2 (主动)。
"""

from .adapter import BenchmarkDeps, ProactiveBenchAdapter

__all__ = [
    "BenchmarkDeps",
    "ProactiveBenchAdapter",
]
