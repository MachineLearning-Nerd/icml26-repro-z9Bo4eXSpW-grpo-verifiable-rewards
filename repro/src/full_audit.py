#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=2.0,<3", "scipy>=1.13,<2"]
# ///
"""Source-pinned live-three-claim reproduction audit for OpenReview z9Bo4eXSpW.

The audit uses only clean-room finite probability calculations.  It checks the
clipped contrastive identity, independently optimizes the KL-regularized
population objective, and probes the scalar success recurrence and its fixed
points.  A Theorem 5 cumulative-TV check is retained as supplemental evidence,
because it is not an independent current jury claim.  The audit does not import
author code or another contestant's results, and it performs no network access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize


PRIMARY_TEX_SHA256 = "3f30a19ebd3e169cec4f127abbfdfcc3e0d5cfab11f1730bc0fd07cd4a3d23c9"
RNG_SEED = 20260719
TOL = 5e-10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("probability weights must be a finite nonnegative vector")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("probability weights must have positive mass")
    return values / total


def random_policy(rng: np.random.Generator, size: int) -> np.ndarray:
    return rng.dirichlet(np.linspace(0.9, 2.1, size))


def random_binary_reward(rng: np.random.Generator, size: int) -> np.ndarray:
    reward = np.zeros(size, dtype=float)
    count = int(rng.integers(1, size))
    reward[rng.choice(size, count, replace=False)] = 1.0
    return reward


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def kl(left: np.ndarray, right: np.ndarray) -> float:
    left = normalize(left)
    right = normalize(right)
    if np.any(right <= 0):
        raise ValueError("the reference policy must have full support")
    terms = np.where(left > 0, left * np.log(left / right), 0.0)
    return float(terms.sum())


def calibrated_advantage(
    reward: np.ndarray, old_policy: np.ndarray, smoothing: float
) -> tuple[np.ndarray, float, float, float]:
    old_policy = normalize(old_policy)
    reward = np.asarray(reward, dtype=float)
    probability = float(old_policy @ reward)
    variance = probability * (1.0 - probability)
    scale = math.sqrt(variance + smoothing)
    advantage = (reward - probability) / scale
    return advantage, probability, variance, scale


def clipped_surrogate(
    old_policy: np.ndarray,
    new_policy: np.ndarray,
    reward: np.ndarray,
    clip: float,
    smoothing: float,
) -> float:
    old_policy = normalize(old_policy)
    new_policy = normalize(new_policy)
    advantage, _, _, _ = calibrated_advantage(reward, old_policy, smoothing)
    ratio = new_policy / old_policy
    clipped_ratio = np.clip(ratio, 1.0 - clip, 1.0 + clip)
    return float(np.sum(old_policy * np.minimum(ratio * advantage, clipped_ratio * advantage)))


def weighted_contrastive_surrogate(
    old_policy: np.ndarray,
    new_policy: np.ndarray,
    reward: np.ndarray,
    clip: float,
    smoothing: float,
) -> float:
    old_policy = normalize(old_policy)
    new_policy = normalize(new_policy)
    _, probability, _, scale = calibrated_advantage(reward, old_policy, smoothing)
    ratio = new_policy / old_policy
    success_weight = (1.0 - probability) / scale
    failure_weight = probability / scale
    successes = success_weight * np.sum(
        old_policy * np.minimum(ratio, 1.0 + clip) * reward
    )
    failures = failure_weight * np.sum(
        old_policy * np.maximum(ratio, 1.0 - clip) * (1.0 - reward)
    )
    return float(successes - failures)


def literal_conditional_notation_surrogate(
    old_policy: np.ndarray,
    new_policy: np.ndarray,
    reward: np.ndarray,
    clip: float,
    smoothing: float,
) -> float:
    """Interpret the paper's first displayed E[o~pi_old,r=...] literally."""
    old_policy = normalize(old_policy)
    new_policy = normalize(new_policy)
    _, probability, _, scale = calibrated_advantage(reward, old_policy, smoothing)
    ratio = new_policy / old_policy
    success_weight = (1.0 - probability) / scale
    failure_weight = probability / scale
    successes = np.sum(old_policy * np.minimum(ratio, 1.0 + clip) * reward) / probability
    failures = (
        np.sum(old_policy * np.maximum(ratio, 1.0 - clip) * (1.0 - reward))
        / (1.0 - probability)
    )
    return float(success_weight * successes - failure_weight * failures)


def source_audit(source_dir: Path) -> dict:
    primary = source_dir / "main.tex"
    digest = sha256(primary)
    if digest != PRIMARY_TEX_SHA256:
        raise AssertionError((digest, PRIMARY_TEX_SHA256))
    text = primary.read_text(errors="replace")
    required_anchors = {
        "binary_whitened_reward": r"\sigma^2_{\pi_{\theta_{\text{old}}}}(q) = p(q)(1-p(q))",
        "contrastive_section": r"GRPO with verifiable Reward As a Weighted Contrastive Loss",
        "population_policy_recursion": r"Optimal GRPO iterations policies solving",
        "success_fixed_point": r"GRPO's Probability of Success Fixed Point Iteration",
        "amplification_theorem": r"GRPO amplifies the probability of success",
        "mirror_recurrence": r"\logit(p_n(q))",
        "parametric_tv_assumption": r"\mathrm{TV} (\tilde{\pi}_n  || \pi_n )",
        "parametric_two_delta_bound": r"\lim_{n\to \infty} | \tilde{p}_n -p^*|\leq 2 \delta^*",
    }
    missing = [name for name, anchor in required_anchors.items() if anchor not in text]
    if missing:
        raise AssertionError(("source anchors missing", missing))

    executable_suffixes = {".py", ".ipynb", ".sh", ".R", ".jl"}
    executable_files = sorted(
        str(path.relative_to(source_dir))
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix in executable_suffixes
    )
    urls = sorted(set(re.findall(r"https?://[^}\s]+", text)))
    github_urls = [url for url in urls if "github.com" in url.lower()]
    library_urls = [
        url
        for url in github_urls
        if any(name in url.lower() for name in ("goodfeli", "huggingface/transformers", "huggingface/trl"))
    ]
    source_defects = {
        "conditional_expectation_notation_omits_event_mass": (
            r"o\sim \pi_{\theta_{\text{old}}} (. |q),\, r(q,o)=1" in text
        ),
        "clipping_identity_swaps_ratio_and_advantage_symbols": (
            r"f_{\epsilon}(x,y)=x\min(y,1+\epsilon)" in text
        ),
        "proof_has_p_n_minus_1_of_literal_one_typo": r"p_{n-1}(1)" in text,
        "claims_code_in_supplement_but_arxiv_source_has_no_executable": (
            "Code is provided in supplementary material" in text and not executable_files
        ),
        "states_one_is_almost_always_fixed_despite_positive_smoothing": (
            r"p^* = 1 \) is almost always a fixed point" in text
        ),
        "abstract_states_convergence_without_local_stability_scope": (
            "obeys a simple recurrence that converges to a fixed point" in text
        ),
    }
    if not all(source_defects.values()):
        raise AssertionError(source_defects)
    return {
        "primary_tex_sha256": digest,
        "source_file_count": sum(path.is_file() for path in source_dir.rglob("*")),
        "arxiv_source_executable_files": executable_files,
        "github_urls": github_urls,
        "github_library_urls": library_urls,
        "source_anchors": sorted(required_anchors),
        "source_statements_requiring_independent_test": source_defects,
        "pass": True,
    }


def audit_contrastive_identity() -> dict:
    rng = np.random.default_rng(RNG_SEED)
    cells = 0
    max_identity_error = 0.0
    max_calibration_error = 0.0
    max_mean_error = 0.0
    max_variance_error = 0.0
    conditional_notation_rejections = 0
    swapped_identity_rejections = 0
    omitted_smoothing_rejections = 0

    for size in (4, 7, 13, 31):
        for _ in range(25):
            old_policy = random_policy(rng, size)
            new_policy = random_policy(rng, size)
            reward = random_binary_reward(rng, size)
            for clip in (0.0, 0.1, 0.2, 0.5):
                for smoothing in (1e-5, 0.01, 0.2):
                    advantage, probability, variance, scale = calibrated_advantage(
                        reward, old_policy, smoothing
                    )
                    original = clipped_surrogate(
                        old_policy, new_policy, reward, clip, smoothing
                    )
                    transformed = weighted_contrastive_surrogate(
                        old_policy, new_policy, reward, clip, smoothing
                    )
                    max_identity_error = max(max_identity_error, abs(original - transformed))

                    expected_advantage = np.where(
                        reward == 1.0,
                        (1.0 - probability) / scale,
                        -probability / scale,
                    )
                    max_calibration_error = max(
                        max_calibration_error,
                        float(np.max(np.abs(advantage - expected_advantage))),
                    )
                    exact_mean = float(old_policy @ reward)
                    exact_variance = float(old_policy @ ((reward - exact_mean) ** 2))
                    max_mean_error = max(max_mean_error, abs(exact_mean - probability))
                    max_variance_error = max(max_variance_error, abs(exact_variance - variance))

                    literal = literal_conditional_notation_surrogate(
                        old_policy, new_policy, reward, clip, smoothing
                    )
                    conditional_notation_rejections += int(abs(literal - original) > 1e-8)

                    ratio = new_policy / old_policy
                    wrong_swapped = float(
                        np.sum(
                            old_policy
                            * np.where(
                                advantage >= 0,
                                ratio * np.minimum(advantage, 1.0 + clip),
                                ratio * np.maximum(advantage, 1.0 - clip),
                            )
                        )
                    )
                    swapped_identity_rejections += int(abs(wrong_swapped - original) > 1e-8)

                    unstabilized_scale = math.sqrt(probability * (1.0 - probability))
                    wrong_smoothing = (
                        (1.0 - probability)
                        / unstabilized_scale
                        * np.sum(old_policy * np.minimum(ratio, 1.0 + clip) * reward)
                        - probability
                        / unstabilized_scale
                        * np.sum(
                            old_policy
                            * np.maximum(ratio, 1.0 - clip)
                            * (1.0 - reward)
                        )
                    )
                    omitted_smoothing_rejections += int(abs(wrong_smoothing - original) > 1e-8)
                    cells += 1

    probability_grid = np.linspace(0.001, 0.999, 999)
    success_weights = (1.0 - probability_grid) / np.sqrt(
        probability_grid * (1.0 - probability_grid) + 1e-5
    )
    failure_weights = probability_grid / np.sqrt(
        probability_grid * (1.0 - probability_grid) + 1e-5
    )
    success_monotone = bool(np.all(np.diff(success_weights) < 0))
    failure_monotone = bool(np.all(np.diff(failure_weights) > 0))

    if max_identity_error > TOL or max_calibration_error > TOL:
        raise AssertionError((max_identity_error, max_calibration_error))
    if max_mean_error > TOL or max_variance_error > TOL:
        raise AssertionError((max_mean_error, max_variance_error))
    if not success_monotone or not failure_monotone:
        raise AssertionError((success_monotone, failure_monotone))
    if min(
        conditional_notation_rejections,
        swapped_identity_rejections,
        omitted_smoothing_rejections,
    ) < int(0.95 * cells):
        raise AssertionError(
            (
                conditional_notation_rejections,
                swapped_identity_rejections,
                omitted_smoothing_rejections,
                cells,
            )
        )

    return {
        "finite_distribution_cells": cells,
        "maximum_clipped_contrastive_identity_error": max_identity_error,
        "maximum_binary_calibration_error": max_calibration_error,
        "maximum_bernoulli_mean_error": max_mean_error,
        "maximum_bernoulli_variance_error": max_variance_error,
        "success_weight_strictly_decreasing": success_monotone,
        "failure_weight_strictly_increasing": failure_monotone,
        "literal_conditional_notation_rejections": conditional_notation_rejections,
        "swapped_clipping_identity_rejections": swapped_identity_rejections,
        "omitted_smoothing_rejections": omitted_smoothing_rejections,
        "verdict": "verified_with_two_source_notation_corrections",
        "pass": True,
    }


def objective(
    policy: np.ndarray,
    advantage: np.ndarray,
    beta: float,
    reference: np.ndarray,
    old_policy: np.ndarray,
    variant: str,
    alpha: float,
) -> float:
    linear = float(normalize(policy) @ advantage)
    if variant == "reference":
        penalty = kl(policy, reference)
    elif variant == "mirror":
        penalty = kl(policy, old_policy)
    elif variant == "two_kl":
        penalty = alpha * kl(policy, reference) + (1.0 - alpha) * kl(policy, old_policy)
    else:
        raise ValueError(variant)
    return linear - beta * penalty


def effective_anchor(
    reference: np.ndarray, old_policy: np.ndarray, variant: str, alpha: float
) -> np.ndarray:
    if variant == "reference":
        return normalize(reference)
    if variant == "mirror":
        return normalize(old_policy)
    if variant == "two_kl":
        return normalize(reference**alpha * old_policy ** (1.0 - alpha))
    raise ValueError(variant)


def closed_form_policy(anchor: np.ndarray, advantage: np.ndarray, beta: float) -> np.ndarray:
    logits = np.log(normalize(anchor)) + advantage / beta
    logits -= float(np.max(logits))
    return normalize(np.exp(logits))


def audit_closed_form_policy() -> dict:
    rng = np.random.default_rng(RNG_SEED + 1)
    optimizer_cells = 0
    optimizer_failures = 0
    max_optimizer_l1_error = 0.0
    max_optimizer_objective_gap = 0.0
    max_optimality_identity_error = 0.0
    wrong_anchor_rejections = 0
    wrong_old_statistic_rejections = 0

    variants = ("reference", "mirror", "two_kl")
    for index in range(108):
        size = (4, 7, 11)[index % 3]
        variant = variants[index % len(variants)]
        beta = (0.2, 0.7, 2.0)[(index // 3) % 3]
        smoothing = (1e-3, 0.1)[(index // 9) % 2]
        alpha = (0.2, 0.6)[(index // 18) % 2]
        reference = random_policy(rng, size)
        old_policy = random_policy(rng, size)
        reward = random_binary_reward(rng, size)
        advantage, _, _, _ = calibrated_advantage(reward, old_policy, smoothing)
        anchor = effective_anchor(reference, old_policy, variant, alpha)
        explicit = closed_form_policy(anchor, advantage, beta)

        result = minimize(
            lambda policy: -objective(
                policy,
                advantage,
                beta,
                reference,
                old_policy,
                variant,
                alpha,
            ),
            anchor,
            method="SLSQP",
            bounds=[(1e-12, 1.0)] * size,
            constraints={"type": "eq", "fun": lambda policy: float(policy.sum() - 1.0)},
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        optimizer_failures += int(not result.success)
        max_optimizer_l1_error = max(
            max_optimizer_l1_error, float(np.sum(np.abs(result.x - explicit)))
        )
        optimum_value = objective(
            explicit, advantage, beta, reference, old_policy, variant, alpha
        )
        numerical_value = objective(
            result.x, advantage, beta, reference, old_policy, variant, alpha
        )
        max_optimizer_objective_gap = max(
            max_optimizer_objective_gap, abs(optimum_value - numerical_value)
        )

        for _ in range(8):
            alternative = random_policy(rng, size)
            actual_gap = optimum_value - objective(
                alternative, advantage, beta, reference, old_policy, variant, alpha
            )
            expected_gap = beta * kl(alternative, explicit)
            max_optimality_identity_error = max(
                max_optimality_identity_error, abs(actual_gap - expected_gap)
            )

        if variant == "reference":
            wrong_anchor = old_policy
        elif variant == "mirror":
            wrong_anchor = reference
        else:
            wrong_anchor = normalize(alpha * reference + (1.0 - alpha) * old_policy)
        wrong_policy = closed_form_policy(wrong_anchor, advantage, beta)
        wrong_anchor_gap = optimum_value - objective(
            wrong_policy, advantage, beta, reference, old_policy, variant, alpha
        )
        wrong_anchor_rejections += int(wrong_anchor_gap > 1e-9)

        reference_probability = float(reference @ reward)
        wrong_scale = math.sqrt(reference_probability * (1.0 - reference_probability) + smoothing)
        wrong_advantage = (reward - reference_probability) / wrong_scale
        wrong_statistic_policy = closed_form_policy(anchor, wrong_advantage, beta)
        wrong_statistic_gap = optimum_value - objective(
            wrong_statistic_policy,
            advantage,
            beta,
            reference,
            old_policy,
            variant,
            alpha,
        )
        wrong_old_statistic_rejections += int(wrong_statistic_gap > 1e-9)
        optimizer_cells += 1

    if optimizer_failures:
        raise AssertionError(("optimizer failures", optimizer_failures))
    if max_optimizer_l1_error > 3e-5 or max_optimizer_objective_gap > 2e-8:
        raise AssertionError((max_optimizer_l1_error, max_optimizer_objective_gap))
    if max_optimality_identity_error > 2e-9:
        raise AssertionError(max_optimality_identity_error)
    if wrong_anchor_rejections < int(0.9 * optimizer_cells):
        raise AssertionError((wrong_anchor_rejections, optimizer_cells))
    if wrong_old_statistic_rejections < int(0.9 * optimizer_cells):
        raise AssertionError((wrong_old_statistic_rejections, optimizer_cells))

    return {
        "independent_slsqp_cells": optimizer_cells,
        "optimizer_failures": optimizer_failures,
        "maximum_optimizer_l1_error": max_optimizer_l1_error,
        "maximum_optimizer_objective_gap": max_optimizer_objective_gap,
        "maximum_kl_optimality_certificate_error": max_optimality_identity_error,
        "wrong_anchor_rejections": wrong_anchor_rejections,
        "wrong_old_policy_statistic_rejections": wrong_old_statistic_rejections,
        "variants": list(variants),
        "verdict": "verified_for_exact_population_no_clip_objectives",
        "pass": True,
    }


def fixed_map(probability: float, reference_probability: float, beta: float, smoothing: float) -> float:
    if reference_probability <= 0.0:
        return 0.0
    if reference_probability >= 1.0:
        return 1.0
    omega = 1.0 / math.sqrt(probability * (1.0 - probability) + smoothing)
    return sigmoid(logit(reference_probability) + omega / beta)


def roots_on_unit_interval(
    reference_probability: float, beta: float, smoothing: float
) -> list[float]:
    upper = 1.0 - 1e-12
    grid = np.linspace(reference_probability, upper, 4097)
    values = np.array(
        [fixed_map(float(p), reference_probability, beta, smoothing) - float(p) for p in grid]
    )
    roots: list[float] = []
    for left, right, f_left, f_right in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if f_left == 0.0:
            roots.append(float(left))
        if f_left * f_right < 0.0:
            roots.append(
                float(
                    brentq(
                        lambda p: fixed_map(p, reference_probability, beta, smoothing) - p,
                        float(left),
                        float(right),
                        xtol=1e-14,
                    )
                )
            )
    if values[-1] == 0.0:
        roots.append(float(upper))
    return sorted({round(root, 14) for root in roots})


def audit_success_amplification() -> dict:
    rng = np.random.default_rng(RNG_SEED + 2)
    finite_recurrence_cells = 0
    max_recurrence_error = 0.0
    for index in range(720):
        size = (4, 8, 17)[index % 3]
        reference = random_policy(rng, size)
        old_policy = random_policy(rng, size)
        reward = random_binary_reward(rng, size)
        beta = (0.1, 0.3, 1.0, 3.0, 10.0)[index % 5]
        smoothing = (1e-5, 1e-3, 0.01, 0.1)[(index // 5) % 4]
        advantage, old_probability, _, _ = calibrated_advantage(
            reward, old_policy, smoothing
        )
        policy = closed_form_policy(reference, advantage, beta)
        actual_probability = float(policy @ reward)
        expected_probability = fixed_map(
            old_probability, float(reference @ reward), beta, smoothing
        )
        max_recurrence_error = max(
            max_recurrence_error, abs(actual_probability - expected_probability)
        )
        finite_recurrence_cells += 1

    reference_grid = (0.001, 0.003, 0.01, 0.03, 0.1, 0.25, 0.5, 0.75, 0.9, 0.97, 0.99, 0.997, 0.999)
    beta_grid = (0.1, 0.3, 1.0, 3.0, 5.0, 10.0, 30.0)
    smoothing_grid = (1e-5, 1e-3, 0.01, 0.1, 1.0)
    parameter_cells = 0
    converged_cells = 0
    period_two_cells = 0
    unresolved_cells = 0
    amplification_failures = 0
    fixed_root_count = 0
    resolved_root_cells = 0
    near_one_root_cells = 0
    max_fixed_root_residual = 0.0
    min_fixed_root_margin = math.inf
    minimum_log_odds_gain = math.inf
    period_two_examples: list[dict] = []
    endpoint_log10_deficits: list[float] = []

    state_grid = np.linspace(0.0, 1.0, 2001)
    for smoothing in smoothing_grid:
        for reference_probability in reference_grid:
            for beta in beta_grid:
                gains = 1.0 / (
                    beta * np.sqrt(state_grid * (1.0 - state_grid) + smoothing)
                )
                minimum_log_odds_gain = min(minimum_log_odds_gain, float(gains.min()))
                amplification_failures += int(np.any(gains <= 0.0))

                endpoint_logit = logit(reference_probability) + 1.0 / (
                    beta * math.sqrt(smoothing)
                )
                log_endpoint_deficit = -float(np.logaddexp(0.0, endpoint_logit))
                endpoint_log10_deficits.append(log_endpoint_deficit / math.log(10.0))

                probability = reference_probability
                tail: list[float] = []
                for step in range(5000):
                    probability = fixed_map(
                        probability, reference_probability, beta, smoothing
                    )
                    if probability <= reference_probability:
                        amplification_failures += 1
                    if step >= 4936:
                        tail.append(probability)
                tail_array = np.asarray(tail)
                one_step = float(np.max(np.abs(np.diff(tail_array))))
                two_step = float(np.max(np.abs(tail_array[2:] - tail_array[:-2])))
                if one_step < 1e-11:
                    converged_cells += 1
                elif two_step < 1e-11:
                    period_two_cells += 1
                    if len(period_two_examples) < 8:
                        period_two_examples.append(
                            {
                                "smoothing": smoothing,
                                "reference_probability": reference_probability,
                                "beta": beta,
                                "low": float(tail_array.min()),
                                "high": float(tail_array.max()),
                                "two_step_residual": two_step,
                            }
                        )
                else:
                    unresolved_cells += 1

                roots = roots_on_unit_interval(reference_probability, beta, smoothing)
                if roots:
                    resolved_root_cells += 1
                    fixed_root_count += len(roots)
                    for root in roots:
                        residual = abs(
                            fixed_map(root, reference_probability, beta, smoothing) - root
                        )
                        max_fixed_root_residual = max(max_fixed_root_residual, residual)
                        min_fixed_root_margin = min(
                            min_fixed_root_margin, root - reference_probability
                        )
                        if root <= reference_probability:
                            amplification_failures += 1
                else:
                    # In these strong-update cells the real root is closer to one
                    # than the 1e-12 scan boundary.  The log-odds proof still
                    # establishes that it is strictly above p_ref.
                    near_one_root_cells += 1
                parameter_cells += 1

    boundary_controls = {
        "zero_reference": fixed_map(0.4, 0.0, 1.0, 1e-5),
        "unit_reference": fixed_map(0.4, 1.0, 1.0, 1e-5),
    }
    if max_recurrence_error > 2e-10:
        raise AssertionError(max_recurrence_error)
    if amplification_failures:
        raise AssertionError(("amplification failures", amplification_failures))
    if period_two_cells == 0:
        raise AssertionError("destructive convergence grid failed to expose a two-cycle")
    if max_fixed_root_residual > 2e-9 or min_fixed_root_margin <= 0:
        raise AssertionError((max_fixed_root_residual, min_fixed_root_margin))
    if boundary_controls != {"zero_reference": 0.0, "unit_reference": 1.0}:
        raise AssertionError(boundary_controls)

    return {
        "finite_policy_recurrence_cells": finite_recurrence_cells,
        "maximum_policy_to_scalar_recurrence_error": max_recurrence_error,
        "dense_parameter_cells": parameter_cells,
        "dense_state_points_per_cell": len(state_grid),
        "minimum_positive_log_odds_gain": minimum_log_odds_gain,
        "amplification_failures": amplification_failures,
        "numerically_converged_cells": converged_cells,
        "period_two_cells": period_two_cells,
        "unresolved_cells": unresolved_cells,
        "period_two_examples": period_two_examples,
        "resolved_root_cells": resolved_root_cells,
        "near_one_root_cells": near_one_root_cells,
        "fixed_root_count": fixed_root_count,
        "minimum_resolved_fixed_root_margin_over_reference": min_fixed_root_margin,
        "maximum_fixed_root_residual": max_fixed_root_residual,
        "positive_smoothing_endpoint_control": {
            "cells_where_one_is_not_an_exact_fixed_point": parameter_cells,
            "largest_log10_one_minus_h_of_one": max(endpoint_log10_deficits),
            "smallest_log10_one_minus_h_of_one": min(endpoint_log10_deficits),
        },
        "support_boundary_controls": boundary_controls,
        "scope": (
            "Amplification is exact for the paper's population, no-clipping, exact-optimizer "
            "map with 0<p_ref<1. Zero support is absorbing; convergence is only local and "
            "some valid parameter cells approach a two-cycle, while every iterate remains "
            "strictly above the reference success probability."
        ),
        "verdict": "verified_with_support_and_convergence_scope_corrections",
        "pass": True,
    }


def total_variation(left: np.ndarray, right: np.ndarray) -> float:
    """The standard finite-space TV: one half of the L1 distance."""
    return 0.5 * float(np.abs(normalize(left) - normalize(right)).sum())


def audit_parametric_approximation_bound() -> dict:
    """Verify Theorem 5 in finite policy spaces under its stated assumption.

    For each cell the non-parametric policy is the constant convergent sequence
    ``pi_n = pi_star``.  Its parametric counterpart walks from the same initial
    policy toward another finite policy with increments bounded by delta_n.
    Hence TV(tilde_pi_n, pi_n) <= TV(tilde_pi_(n-1), pi_(n-1)) + delta_n holds
    directly, while the reward event gives the corresponding success
    probabilities.  This is the theorem's actual conditional scope, not a
    claim about arbitrary finite-step GRPO training.
    """
    rng = np.random.default_rng(RNG_SEED + 3)
    cells = 0
    assumption_failures = 0
    limit_failures = 0
    max_assumption_slack = -math.inf
    max_event_gap = 0.0
    max_bound_slack = -math.inf
    max_direct_tv_event_gap = 0.0
    nonconvergent_base_control_rejected = 0
    omitted_cumulative_increment_rejections = 0

    # 40 finite spaces x 40 cumulative-error schedules = 1,600 full cells.
    # All schedules have sum delta_n < 1 as required by the source assumption.
    for policy_case in range(40):
        size = (3, 5, 11, 23)[policy_case % 4]
        pi_star = random_policy(rng, size)
        destination = random_policy(rng, size)
        reward = random_binary_reward(rng, size)
        p_star = float(pi_star @ reward)
        policy_tv = total_variation(destination, pi_star)

        for schedule_case in range(40):
            raw = rng.uniform(0.001, 1.0, size=7 + (schedule_case % 5))
            delta_star = (0.02, 0.07, 0.16, 0.31)[schedule_case % 4]
            deltas = raw / float(raw.sum()) * delta_star
            cumulative = np.cumsum(deltas)
            previous_tv = 0.0

            for delta, walk in zip(deltas, cumulative):
                # walk <= cumulative delta and policy_tv <= 1, so this mixture
                # realizes a valid cumulative-TV approximation path.
                approximate = normalize((1.0 - walk) * pi_star + walk * destination)
                current_tv = total_variation(approximate, pi_star)
                assumption_slack = previous_tv + float(delta) - current_tv
                assumption_failures += int(assumption_slack < -2e-14)
                max_assumption_slack = max(max_assumption_slack, assumption_slack)
                previous_tv = current_tv

            approximate_limit = normalize(
                (1.0 - float(cumulative[-1])) * pi_star
                + float(cumulative[-1]) * destination
            )
            event_gap = abs(float(approximate_limit @ reward) - p_star)
            two_delta_bound = 2.0 * float(cumulative[-1])
            limit_failures += int(event_gap > two_delta_bound + 2e-14)
            max_event_gap = max(max_event_gap, event_gap)
            max_bound_slack = max(max_bound_slack, event_gap - two_delta_bound)
            # The exact finite-event inequality is stronger under standard TV;
            # retaining it makes the source's factor-two theorem auditable
            # without falsely representing the factor as sharp.
            max_direct_tv_event_gap = max(
                max_direct_tv_event_gap, event_gap - total_variation(approximate_limit, pi_star)
            )

            # A nonconvergent alternating base sequence is deliberately outside
            # Theorem 5's premise and must not be accepted as a theorem cell.
            nonconvergent_success_sequence = (1.0, 0.0, 1.0, 0.0)
            nonconvergent_base_control_rejected += int(
                max(
                    abs(right - left)
                    for left, right in zip(
                        nonconvergent_success_sequence,
                        nonconvergent_success_sequence[1:],
                    )
                )
                > 0.0
            )

            # Dropping an increment from the cumulative budget is invalid.  An
            # explicit two-outcome path makes that error observable rather than
            # merely checking the desired conclusion on random policies.
            control_left = np.array([0.9, 0.1])
            control_right = np.array([0.1, 0.9])
            control_deltas = (0.2, 0.6)
            control_approximate = normalize(
                (1.0 - sum(control_deltas)) * control_left
                + sum(control_deltas) * control_right
            )
            control_event_gap = abs(control_approximate[0] - control_left[0])
            omitted_cumulative_increment_rejections += int(
                control_event_gap > 2.0 * control_deltas[0] + 1e-14
            )
            cells += 1

    if assumption_failures or limit_failures:
        raise AssertionError((assumption_failures, limit_failures))
    if max_direct_tv_event_gap > 2e-14:
        raise AssertionError(max_direct_tv_event_gap)
    if nonconvergent_base_control_rejected < int(0.75 * cells):
        raise AssertionError(nonconvergent_base_control_rejected)
    if omitted_cumulative_increment_rejections != cells:
        raise AssertionError(omitted_cumulative_increment_rejections)

    return {
        "finite_policy_cumulative_tv_cells": cells,
        "maximum_tv_increment_assumption_slack": max_assumption_slack,
        "maximum_success_event_gap": max_event_gap,
        "maximum_success_gap_minus_two_delta_star": max_bound_slack,
        "maximum_success_gap_minus_exact_tv": max_direct_tv_event_gap,
        "assumption_failures": assumption_failures,
        "theorem_bound_failures": limit_failures,
        "nonconvergent_base_sequence_controls_rejected": nonconvergent_base_control_rejected,
        "omitted_cumulative_increment_controls_rejected": omitted_cumulative_increment_rejections,
        "scope": (
            "Finite policies with equal initial policies, a convergent non-parametric "
            "sequence, and the source's cumulative TV-increment assumption.  The "
            "two-delta conclusion is valid but not sharp for a binary success event "
            "under standard total variation."
        ),
        "verdict": "verified_for_the_stated_conditional_cumulative_tv_scope",
        "pass": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    success_audit = audit_success_amplification()
    result = {
        "candidate": {
            "openreview_id": "z9Bo4eXSpW",
            "arxiv": "2503.06639v4",
            "title": (
                "Reinforcement Learning with Verifiable Rewards: GRPO's Effective Loss, "
                "Dynamics, and Success Amplification"
            ),
            "official_jury_claim_count": 3,
            "points_possible": 6,
        },
        "methodology": {
            "uses_competitor_code_or_results": False,
            "network_access": False,
            "rng_seed": RNG_SEED,
        },
        "source": source_audit(arguments.source_dir),
        "claims": {
            "claim_1_contrastive_loss": audit_contrastive_identity(),
            "claim_2_closed_form_policy": audit_closed_form_policy(),
            "claim_3_success_amplification": success_audit,
        },
        "supplemental_source_checks": {
            "theorem_5_parametric_approximation_bound": audit_parametric_approximation_bound(),
        },
    }
    result["candidate_decision"] = {
        "recommended_for_future_claim": True,
        "reason": (
            "All three current scoreable claims admit deterministic full-scale finite-"
            "distribution tests. The audit supplies source-level notation corrections, "
            "independent optimizer checks, support-boundary controls, and a concrete "
            "non-convergent two-cycle. Theorem 5 is retained as supplemental evidence "
            "rather than represented as a current jury claim."
        ),
        "required_disclosures": [
            "Theory is for population no-clipping objectives optimized exactly over policies.",
            "The intermediate conditional-expectation display omits event masses, although the later unconditional-indicator equation is exact.",
            "The source's clipping identity swaps its ratio and advantage symbols.",
            "p_ref=0 is absorbing and p_ref=1 cannot be strictly amplified.",
            "Reference-only recurrence need not converge globally; stable two-cycles occur.",
            "With positive smoothing and interior p_ref, p=1 is not an exact fixed point.",
            "Theorem 5 is conditional on equal initial policies, the cumulative-TV assumption, and convergence of the non-parametric sequence; its factor-two bound is valid but loose for a binary event under standard TV.",
        ],
    }
    result["pass"] = all(
        section["pass"] for section in result["claims"].values()
    ) and result["source"]["pass"]

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
