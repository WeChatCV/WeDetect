# Copyright (c) Tencent Inc. All rights reserved.
from .batch_task_aligned_assigner import BatchTaskAlignedAssigner
from .task_aligned_assigner import YOLOWorldSegAssigner

__all__ = ['YOLOWorldSegAssigner', 'BatchTaskAlignedAssigner']