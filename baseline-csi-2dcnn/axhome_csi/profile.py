from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np

from .data import SampleRecord


def _sorted_counts(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def summarize_samples(samples: Sequence[SampleRecord]) -> dict[str, object]:
    if not samples:
        raise ValueError("cannot profile an empty sample list")
    human = [sample for sample in samples if sample.person_id != "NONE"]
    packet_source = human or list(samples)
    packets = np.array(
        [sample.csi_packets_written for sample in packet_source], dtype=np.float64
    )
    person_counts = _sorted_counts(
        [sample.person_id for sample in human]
    )
    action_counts = _sorted_counts(
        [sample.action_id for sample in samples]
    )
    human_action_counts = _sorted_counts(
        [sample.action_id for sample in human]
    )
    return {
        "samples_total": len(samples),
        "samples_human": len(human),
        "samples_background": len(samples) - len(human),
        "people": sorted(person_counts),
        "environments": sorted({sample.environment_id for sample in samples}),
        "archive_sessions": sorted(
            {sample.archive_session_id for sample in samples}
        ),
        "person_counts": person_counts,
        "environment_counts": _sorted_counts(
            [sample.environment_id for sample in samples]
        ),
        "human_environment_counts": _sorted_counts(
            [sample.environment_id for sample in human]
        ),
        "action_counts": action_counts,
        "human_action_counts": human_action_counts,
        "archive_session_counts": _sorted_counts(
            [sample.archive_session_id for sample in samples]
        ),
        "packet_summary": {
            "population": "human" if human else "all",
            "min": int(packets.min()),
            "q1": float(np.quantile(packets, 0.25)),
            "median": float(np.median(packets)),
            "q3": float(np.quantile(packets, 0.75)),
            "max": int(packets.max()),
            "mean": float(packets.mean()),
        },
        "human_action_imbalance_ratio": (
            float(max(human_action_counts.values()) / min(human_action_counts.values()))
            if human_action_counts
            else None
        ),
    }

