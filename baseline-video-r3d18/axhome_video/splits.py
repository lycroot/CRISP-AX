from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Sequence

from .data import VideoSampleRecord

HUMAN_ACTIONS = (
    "bed_fall_sim",
    "bend_pick",
    "brush_teeth",
    "fall_like",
    "get_up",
    "lie_down",
    "post_fall_wave",
    "reading",
    "sit_down",
    "stand_up",
    "support_wall",
    "turn_over",
    "walk",
    "wash_hands",
)
ACTION_TO_INDEX = {action: index for index, action in enumerate(HUMAN_ACTIONS)}


@dataclass(frozen=True)
class SubjectFold:
    train: tuple[VideoSampleRecord, ...]
    val: tuple[VideoSampleRecord, ...]
    test: tuple[VideoSampleRecord, ...]
    train_people: tuple[str, ...]
    val_person: str
    test_person: str


@dataclass(frozen=True)
class InDomainSplit:
    train: tuple[VideoSampleRecord, ...]
    val: tuple[VideoSampleRecord, ...]
    test: tuple[VideoSampleRecord, ...]
    seed: int


def human_samples(
    samples: Iterable[VideoSampleRecord],
) -> tuple[VideoSampleRecord, ...]:
    return tuple(
        sample
        for sample in samples
        if sample.person_id != "NONE" and sample.action_id in ACTION_TO_INDEX
    )


def make_subject_fold(
    samples: Sequence[VideoSampleRecord], *, test_person: str, val_person: str
) -> SubjectFold:
    if test_person == val_person:
        raise ValueError("test_person and val_person must be different")
    eligible = human_samples(samples)
    people = sorted({sample.person_id for sample in eligible})
    unknown = {test_person, val_person} - set(people)
    if unknown:
        raise ValueError("unknown fold people: " + ", ".join(sorted(unknown)))
    train_people = tuple(
        person for person in people if person not in {test_person, val_person}
    )
    train = tuple(sample for sample in eligible if sample.person_id in train_people)
    val = tuple(sample for sample in eligible if sample.person_id == val_person)
    test = tuple(sample for sample in eligible if sample.person_id == test_person)
    if not train or not val or not test:
        raise ValueError("subject fold contains an empty split")
    return SubjectFold(
        train=train,
        val=val,
        test=test,
        train_people=train_people,
        val_person=val_person,
        test_person=test_person,
    )


def rotating_loso_pairs(
    samples: Sequence[VideoSampleRecord],
) -> list[tuple[str, str]]:
    people = sorted({sample.person_id for sample in human_samples(samples)})
    if len(people) < 3:
        raise ValueError("LOSO requires at least three people")
    return [
        (test_person, people[(index + 1) % len(people)])
        for index, test_person in enumerate(people)
    ]


def validate_in_domain_split(
    split: InDomainSplit,
    *,
    expected_samples: Sequence[VideoSampleRecord] | None = None,
) -> None:
    partitions = (split.train, split.val, split.test)
    if any(not partition for partition in partitions):
        raise ValueError("in-domain split contains an empty partition")
    id_lists = [
        [sample.sample_id for sample in partition] for partition in partitions
    ]
    if any(len(ids) != len(set(ids)) for ids in id_lists):
        raise ValueError("in-domain split contains duplicate sample IDs")
    id_sets = [set(ids) for ids in id_lists]
    if any(
        left & right
        for index, left in enumerate(id_sets)
        for right in id_sets[index + 1 :]
    ):
        raise ValueError("in-domain train, validation, and test samples overlap")

    assigned = tuple(sample for part in partitions for sample in part)
    if any(
        sample.person_id == "NONE" or sample.action_id not in ACTION_TO_INDEX
        for sample in assigned
    ):
        raise ValueError("in-domain split contains a non-human sample")
    expected = (
        human_samples(expected_samples)
        if expected_samples is not None
        else assigned
    )
    expected_ids = [sample.sample_id for sample in expected]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("eligible samples contain duplicate sample IDs")
    if set(expected_ids) != set().union(*id_sets):
        raise ValueError("in-domain split does not cover every eligible sample")

    expected_strata = {
        (sample.person_id, sample.action_id) for sample in expected
    }
    for name, partition in zip(
        ("train", "validation", "test"), partitions, strict=True
    ):
        present = {(sample.person_id, sample.action_id) for sample in partition}
        missing = expected_strata - present
        if missing:
            rendered = ", ".join(
                f"{person}/{action}" for person, action in sorted(missing)
            )
            raise ValueError(f"in-domain {name} split is missing strata: {rendered}")


def make_in_domain_split(
    samples: Sequence[VideoSampleRecord], *, seed: int
) -> InDomainSplit:
    eligible = human_samples(samples)
    if not eligible:
        raise ValueError("in-domain split contains no human samples")
    groups: dict[tuple[str, str], list[VideoSampleRecord]] = {}
    for sample in eligible:
        groups.setdefault((sample.person_id, sample.action_id), []).append(sample)

    rng = random.Random(seed)
    train: list[VideoSampleRecord] = []
    val: list[VideoSampleRecord] = []
    test: list[VideoSampleRecord] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda sample: sample.sample_id)
        rng.shuffle(group)
        n_train = round(len(group) * 0.65)
        remaining = len(group) - n_train
        n_val = remaining // 2
        n_test = remaining - n_val
        if min(n_train, n_val, n_test) <= 0:
            raise ValueError(
                f"stratum {key} with {len(group)} samples cannot be split "
                "into three non-empty sets"
            )
        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])
    split = InDomainSplit(
        train=tuple(train), val=tuple(val), test=tuple(test), seed=seed
    )
    validate_in_domain_split(split, expected_samples=eligible)
    return split
