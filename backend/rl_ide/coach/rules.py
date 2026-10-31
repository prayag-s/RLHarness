"""Deterministic coach rules.

Each rule is a pure function of the facts. No rule calls a model, touches the network, or reads
mutable state, so the whole catalogue is testable without a provider and costs nothing to run on
every checkpoint.
"""

from __future__ import annotations

from rl_ide.coach.facts import (
    DatasetFacts,
    ExportFacts,
    ObjectiveFacts,
    ParameterFacts,
    PreviewFacts,
    ResultFacts,
    episodes_for_half_width,
)
from rl_ide.coach.models import (
    CoachStage,
    GuidanceAction,
    GuidanceItem,
    Severity,
)


def _item(
    rule_id: str,
    stage: CoachStage,
    severity: Severity,
    title: str,
    why: str,
    *,
    evidence: dict | None = None,
    actions: list[GuidanceAction] | None = None,
) -> GuidanceItem:
    return GuidanceItem(
        rule_id=rule_id,
        stage=stage,
        severity=severity,
        title=title,
        why=why,
        evidence=evidence or {},
        actions=actions or [],
    )


def _guide(section: str, label: str | None = None) -> GuidanceAction:
    return GuidanceAction(
        kind="open_guide",
        label=label or f"Read guide §{section}",
        payload={"section": section},
    )


def _say(message: str, label: str) -> GuidanceAction:
    return GuidanceAction(kind="send_message", label=label, payload={"message": message})


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def objective_items(facts: ObjectiveFacts) -> list[GuidanceItem]:
    items: list[GuidanceItem] = []

    if facts.word_count < 8:
        items.append(
            _item(
                "objective.too_vague",
                CoachStage.OBJECTIVE,
                Severity.WARNING,
                "The objective is too short to constrain the build",
                f"It is {facts.word_count} words. The architect fills every gap with a default, "
                "and a generic reward is the usual result.",
                evidence={"word_count": facts.word_count},
                actions=[
                    _guide("5.1", "Open the six-part formula"),
                    _say(
                        "Rewrite the objective so it answers: who decides, in what situation, "
                        "what may it choose, what success looks like, what it must never do, "
                        "and what the human must see.",
                        "Ask for a rewrite",
                    ),
                ],
            )
        )

    if not facts.has_metric_cue and not facts.dataset_has_target:
        items.append(
            _item(
                "objective.no_success_metric",
                CoachStage.OBJECTIVE,
                Severity.WARNING,
                "No measurable success criterion was stated",
                "Without a metric the training loop optimises whatever the reward happens to "
                "reward, and the scorecard cannot tell you whether it worked.",
                evidence={
                    "has_metric_cue": False,
                    "dataset_has_target": facts.dataset_has_target,
                },
                actions=[
                    _guide("5.1", "See how to state success"),
                    _say(
                        "Add a measurable success criterion to the objective, for example "
                        "'at least 90% agreement with the recorded outcome' or 'average steps "
                        "within 10% of the shortest path'.",
                        "Add a success metric",
                    ),
                ],
            )
        )

    if (
        facts.has_prohibition_cue
        and facts.hard_constraint_count == 0
        and not facts.reward_has_penalty
    ):
        items.append(
            _item(
                "objective.unencoded_prohibition",
                CoachStage.OBJECTIVE,
                Severity.WARNING,
                "A 'never' rule was stated but not encoded",
                "The objective contains a prohibition, but it produced neither a hard constraint "
                "nor a penalty term. The policy is free to violate it and the scorecard will "
                "still call the run a success.",
                evidence={
                    "hard_constraint_count": 0,
                    "reward_terms": facts.reward_terms,
                    "reward_has_penalty": facts.reward_has_penalty,
                },
                actions=[
                    _guide("17", "Choose a verified guard"),
                    _say(
                        "Turn the prohibition into an executable rule with a threshold, for "
                        "example 'never approve when vendor_risk_score >= 85', or add an explicit "
                        "penalty for violating it.",
                        "Make the rule executable",
                    ),
                ],
            )
        )

    if not facts.action_labels and not facts.blueprint_actions and not facts.dataset_has_target:
        items.append(
            _item(
                "objective.no_action_space",
                CoachStage.OBJECTIVE,
                Severity.BLOCKER,
                "The agent has no actions to choose between",
                "No action set could be read from the objective and the blueprint exposes none. "
                "An environment cannot be built without a decision to make.",
                evidence={
                    "action_labels": facts.action_labels,
                    "blueprint_actions": facts.blueprint_actions,
                },
                actions=[
                    _say(
                        "Give the agent an explicit set of actions, for example "
                        "'choose between approve, review, and block'.",
                        "Define the actions",
                    )
                ],
            )
        )

    if facts.multiple_jobs:
        items.append(
            _item(
                "objective.multiple_jobs",
                CoachStage.OBJECTIVE,
                Severity.WARNING,
                "This objective asks for more than one job",
                "A phrase such as 'and also' is a reliable sign that two agents are being "
                "described. One agent, one job keeps the reward and the scorecard meaningful.",
                evidence={"prompt_excerpt": facts.prompt[:200]},
                actions=[
                    _guide("5.3", "Read the rules of thumb"),
                    _say(
                        "Split this into separate agents, one job each, and tell me which one "
                        "to build first.",
                        "Split into one job",
                    ),
                ],
            )
        )

    if not facts.observation_features and not facts.dataset_has_target:
        items.append(
            _item(
                "objective.no_observation",
                CoachStage.OBJECTIVE,
                Severity.TIP,
                "The state the agent observes is implicit",
                "No observation features were named. The environment will invent a state, and "
                "the arena may not show what actually drove the decision.",
                evidence={"observation_features": []},
                actions=[_guide("5.1", "Name what the agent sees")],
            )
        )

    return items


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def dataset_items(facts: DatasetFacts, *, supervised_objective: bool = False) -> list[GuidanceItem]:
    items: list[GuidanceItem] = []
    stage = CoachStage.DATA

    if facts.row_count <= 0:
        items.append(
            _item(
                "data.empty",
                stage,
                Severity.BLOCKER,
                "The attached dataset has no rows",
                "Nothing can be trained or evaluated on an empty table.",
                evidence={"row_count": facts.row_count},
                actions=[_guide("11.1", "What can be attached")],
            )
        )
        return items

    if facts.target_column and len(facts.target_distribution) == 1:
        label = next(iter(facts.target_distribution))
        items.append(
            _item(
                "data.single_class_target",
                stage,
                Severity.BLOCKER,
                "The label column contains only one class",
                f"Every row is labelled '{label}'. A classifier cannot learn a boundary when "
                "there is nothing to separate, and any score is meaningless.",
                evidence={
                    "target_column": facts.target_column,
                    "target_distribution": facts.target_distribution,
                },
                actions=[
                    _say(
                        f"Replace the attached data or choose a different label column; "
                        f"'{facts.target_column}' has only one class.",
                        "Replace the data",
                    )
                ],
            )
        )

    if supervised_objective and not facts.target_column:
        items.append(
            _item(
                "data.missing_label_for_supervised_objective",
                stage,
                Severity.BLOCKER,
                "The objective is supervised but the data has no label column",
                "The success metric is accuracy-like, yet no column was detected as the target. "
                "Training would have no ground truth to score against.",
                evidence={
                    "target_column": None,
                    "columns": facts.column_count,
                    "row_count": facts.row_count,
                },
                actions=[
                    _say(
                        "Name the column that holds the recorded outcome, for example "
                        "'use the decision column as the label'.",
                        "Name the label column",
                    ),
                    _guide("11.2", "How the app uses a dataset"),
                ],
            )
        )

    if facts.minority_ratio is not None and facts.minority_ratio < 0.05:
        items.append(
            _item(
                "data.class_imbalance",
                stage,
                Severity.WARNING,
                "The classes are severely imbalanced",
                f"The minority class is {facts.minority_ratio * 100:.1f}% of the data. Plain "
                "accuracy will look excellent while the rare, costly class is missed.",
                evidence={
                    "target_column": facts.target_column,
                    "minority_ratio": facts.minority_ratio,
                    "target_distribution": facts.target_distribution,
                },
                actions=[
                    _say(
                        "Optimise recall on the minority class at a fixed false-positive rate "
                        "instead of plain accuracy, and say which error costs more.",
                        "Optimise the costly class",
                    ),
                    _guide("11.3", "Data hygiene that changes results"),
                ],
            )
        )

    if facts.max_missing_ratio > 0.30:
        worst = facts.missing_columns[0] if facts.missing_columns else {}
        items.append(
            _item(
                "data.high_missing",
                stage,
                Severity.WARNING,
                "A feature is mostly missing",
                f"{worst.get('column')} is {facts.max_missing_ratio * 100:.0f}% empty. The "
                "encoder will impute a value the policy then treats as real signal.",
                evidence={
                    "max_missing_ratio": facts.max_missing_ratio,
                    "missing_columns": facts.missing_columns,
                },
                actions=[_guide("11.3", "Handle missing values")],
            )
        )

    if facts.duplicate_ratio > 0.10:
        items.append(
            _item(
                "data.duplicates",
                stage,
                Severity.WARNING,
                "The dataset contains many duplicate rows",
                f"{facts.duplicate_rows} rows ({facts.duplicate_ratio * 100:.0f}%) are exact "
                "duplicates. If they span the train/test split, the held-out score is inflated.",
                evidence={
                    "duplicate_rows": facts.duplicate_rows,
                    "duplicate_ratio": facts.duplicate_ratio,
                },
                actions=[_guide("11.4", "Run the leakage test")],
            )
        )

    if facts.id_like_columns:
        items.append(
            _item(
                "data.id_leakage",
                stage,
                Severity.WARNING,
                "A row identifier is being used as a feature",
                "An ID-like column with one unique value per row lets the policy memorise the "
                "split instead of learning a rule.",
                evidence={"id_like_columns": facts.id_like_columns},
                actions=[
                    _say(
                        "Exclude the identifier columns from the observations: "
                        + ", ".join(facts.id_like_columns)
                        + ".",
                        "Exclude the identifiers",
                    )
                ],
            )
        )

    if facts.separation_columns:
        items.append(
            _item(
                "data.target_leakage",
                stage,
                Severity.WARNING,
                "A single feature separates the label perfectly",
                "Perfect separation is almost always the target leaking into a feature. The "
                "held-out score will collapse in production.",
                evidence={"separation_columns": facts.separation_columns},
                actions=[
                    _guide("11.4", "Run the leakage test"),
                    _say(
                        "Remove or lag the leaking feature(s): "
                        + ", ".join(facts.separation_columns)
                        + ".",
                        "Remove the leaking features",
                    ),
                ],
            )
        )

    if facts.row_count < 200:
        items.append(
            _item(
                "data.too_small",
                stage,
                Severity.WARNING,
                "The dataset is too small for a stable held-out score",
                f"{facts.row_count} rows cannot support a test split with a useful confidence "
                "interval. Treat any result as a smoke test, not evidence.",
                evidence={"row_count": facts.row_count},
                actions=[
                    _guide("13", "Read the held-out scorecard"),
                    _say(
                        "Attach more rows, or lower the target and treat this run as a smoke test.",
                        "Adjust expectations",
                    ),
                ],
            )
        )

    return items


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

def recommended_parameters(facts: ParameterFacts) -> dict[str, float | int]:
    episodes = max(int(facts.episodes_per_round or 0), 200)
    target = float(facts.target or 85.0)
    rounds = max(int(facts.rounds or 0), 1)

    if target >= 99.999:
        target = 90.0
    if target <= 1.0:
        target = 85.0
    if facts.baseline_metric is not None and target <= facts.baseline_metric + 1.0:
        target = min(95.0, round(facts.baseline_metric + 10.0, 1))

    return {
        "max_auto_tune_rounds": rounds,
        "episodes_per_round": episodes,
        "target_metric_threshold": target,
    }


def parameter_items(facts: ParameterFacts) -> list[GuidanceItem]:
    items: list[GuidanceItem] = []
    stage = CoachStage.PARAMETERS
    recommended = recommended_parameters(facts)

    if facts.rounds <= 0 or facts.episodes_per_round <= 0:
        items.append(
            _item(
                "params.nonpositive_budget",
                stage,
                Severity.BLOCKER,
                "The training budget is zero",
                "Training needs at least one round and one episode. With a zero budget the "
                "loop exits before the policy can learn anything.",
                evidence={
                    "rounds": facts.rounds,
                    "episodes_per_round": facts.episodes_per_round,
                },
                actions=[
                    GuidanceAction(
                        kind="apply_params",
                        label="Use 3 rounds × 200 episodes",
                        payload=recommended,
                    )
                ],
            )
        )

    if facts.target <= 0 or facts.target > 100:
        # Not a blocker: an unreachable target is also how you deliberately ask for the full
        # round budget, and the run is still valid. It is a value that needs explaining.
        items.append(
            _item(
                "params.target_out_of_range",
                stage,
                Severity.WARNING,
                "The target metric is outside 0–100%",
                f"A target of {facts.target}% cannot be reached by a percentage metric. The run "
                "will exhaust its rounds rather than stop early.",
                evidence={"target": facts.target, "metric_name": facts.metric_name},
                actions=[
                    GuidanceAction(
                        kind="apply_params",
                        label="Set a reachable target",
                        payload=recommended,
                    )
                ],
            )
        )

    if facts.target >= 99.999:
        items.append(
            _item(
                "params.target_100",
                stage,
                Severity.WARNING,
                "A 100% target turns measurement noise into failure",
                f"At {facts.episodes_per_round} episodes the 95% interval is about "
                f"±{facts.ci_half_width_pct or 0:.1f} points. The policy can be good and still "
                "never print 100%. Raising episodes tightens the estimate; raising the target "
                "usually just burns rounds.",
                evidence={
                    "target": facts.target,
                    "episodes_per_round": facts.episodes_per_round,
                    "ci_half_width_pct": facts.ci_half_width_pct,
                },
                actions=[
                    GuidanceAction(
                        kind="apply_params",
                        label="Use a 90% target instead",
                        payload=recommended,
                    ),
                    _guide("9.3", "Read the 100% trap"),
                ],
            )
        )

    if facts.episodes_per_round < 20:
        needed = episodes_for_half_width(5.0)
        items.append(
            _item(
                "params.episodes_too_few",
                stage,
                Severity.WARNING,
                "Too few episodes to trust the score",
                f"At {facts.episodes_per_round} episodes the 95% interval is about "
                f"±{facts.ci_half_width_pct or 0:.1f} points. About {needed} episodes are "
                "needed for ±5 points.",
                evidence={
                    "episodes_per_round": facts.episodes_per_round,
                    "ci_half_width_pct": facts.ci_half_width_pct,
                    "episodes_for_5pt": needed,
                },
                actions=[
                    GuidanceAction(
                        kind="run_episodes",
                        label=f"Run {needed} episodes per round",
                        payload={"episodes_per_round": needed},
                    ),
                    _guide("9.2", "What more episodes buys"),
                ],
            )
        )

    if facts.baseline_metric is not None and facts.target <= facts.baseline_metric + 1.0:
        items.append(
            _item(
                "params.target_below_baseline",
                stage,
                Severity.WARNING,
                "The target is already met by doing nothing",
                f"The majority-class baseline is {facts.baseline_metric:.1f}%, so a target of "
                f"{facts.target:.1f}% can be reached without learning anything.",
                evidence={
                    "target": facts.target,
                    "baseline_metric": facts.baseline_metric,
                },
                actions=[
                    GuidanceAction(
                        kind="apply_params",
                        label=f"Target {recommended['target_metric_threshold']}%",
                        payload=recommended,
                    )
                ],
            )
        )

    if (
        facts.uniform_chance is not None
        and facts.action_count > 1
        and facts.target <= facts.uniform_chance + 1.0
    ):
        items.append(
            _item(
                "params.target_below_chance",
                stage,
                Severity.WARNING,
                "The target is at or below random choice",
                f"With {facts.action_count} actions, uniform random choice scores about "
                f"{facts.uniform_chance:.1f}%. A target of {facts.target:.1f}% proves nothing.",
                evidence={
                    "target": facts.target,
                    "uniform_chance": facts.uniform_chance,
                    "action_count": facts.action_count,
                },
                actions=[
                    GuidanceAction(
                        kind="apply_params",
                        label="Raise the target",
                        payload=recommended,
                    )
                ],
            )
        )

    if facts.projected_episodes > 20000:
        items.append(
            _item(
                "params.run_too_large",
                stage,
                Severity.WARNING,
                "This run is very large",
                f"{facts.rounds} rounds × {facts.episodes_per_round} episodes is "
                f"{facts.projected_episodes} episodes. Expect a long wall-clock run and a "
                "bigger bill before you have any evidence.",
                evidence={
                    "rounds": facts.rounds,
                    "episodes_per_round": facts.episodes_per_round,
                    "projected_episodes": facts.projected_episodes,
                },
                actions=[
                    GuidanceAction(
                        kind="apply_params",
                        label="Start smaller",
                        payload={
                            "max_auto_tune_rounds": 3,
                            "episodes_per_round": 200,
                            "target_metric_threshold": recommended["target_metric_threshold"],
                        },
                    ),
                    _guide("18", "Read the cost playbook"),
                ],
            )
        )

    if facts.prompt_has_prohibition and not facts.reward_has_penalty:
        items.append(
            _item(
                "params.prohibition_without_penalty",
                stage,
                Severity.WARNING,
                "The reward has no term for the thing you forbade",
                "The objective contains a prohibition, but the reward terms are all positive. "
                "The policy has no incentive to avoid the forbidden behaviour.",
                evidence={
                    "reward_terms": facts.reward_terms,
                    "reward_has_penalty": False,
                },
                actions=[
                    _guide("10.2", "How to phrase a reward change"),
                ],
            )
        )

    return items


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

def preview_items(facts: PreviewFacts, *, interactive_expected: bool = False) -> list[GuidanceItem]:
    items: list[GuidanceItem] = []
    stage = CoachStage.PREVIEW

    if facts.html_length == 0:
        items.append(
            _item(
                "preview.empty",
                stage,
                Severity.WARNING,
                "No arena was produced",
                "The centre panel is the only place a human can judge behaviour. Without it, "
                "the first honest look happens after training.",
                evidence={"html_length": 0},
                actions=[_guide("6", "What the centre panel must show")],
            )
        )
        return items

    if not facts.has_metric_signal:
        items.append(
            _item(
                "preview.no_metric",
                stage,
                Severity.WARNING,
                "The arena does not show the success metric",
                f"The score you will be asked to trust is '{facts.metric_name}', but the arena "
                "never displays it. You will be unable to tell success from luck by looking.",
                evidence={"metric_name": facts.metric_name},
                actions=[
                    _say(
                        "Add a visible readout for "
                        f"'{facts.metric_name}' to the arena, alongside the current decision.",
                        "Add the metric to the arena",
                    )
                ],
            )
        )

    if not facts.has_live_update:
        items.append(
            _item(
                "preview.static",
                stage,
                Severity.WARNING,
                "The arena looks static",
                "No animation or message hook was found. A static picture cannot show a policy "
                "changing its mind as the state changes.",
                evidence={"html_length": facts.html_length},
                actions=[_say("Make the arena update live as each step is taken.", "Make it live")],
            )
        )

    if interactive_expected and not facts.has_controls:
        items.append(
            _item(
                "preview.no_controls",
                stage,
                Severity.TIP,
                "The arena has no interactive controls",
                "You asked to inspect or override decisions, but the arena exposes no buttons.",
                evidence={"interactive_expected": True},
                actions=[_guide("8.1", "Play / Interact")],
            )
        )

    return items


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def result_items(facts: ResultFacts) -> list[GuidanceItem]:
    items: list[GuidanceItem] = []
    stage = CoachStage.RESULTS

    if not facts.has_scorecard:
        items.append(
            _item(
                "results.no_scorecard",
                stage,
                Severity.TIP,
                "No held-out scorecard yet",
                "Run at least one training round to produce the evidence this screen interprets.",
                evidence={"rounds": len(facts.rounds)},
                actions=[_guide("13", "What the scorecard contains")],
            )
        )
        return items

    if not facts.held_out:
        items.append(
            _item(
                "results.not_held_out",
                stage,
                Severity.WARNING,
                "This score is in-sample",
                "The evaluation could not use an unseen split, so the number describes fit, "
                "not generalisation. Do not quote it as evidence.",
                evidence={"held_out": False, "notes": facts.notes[:2]},
                actions=[_guide("13.2", "In-sample is a warning")],
            )
        )

    if facts.ci and (facts.ci[1] - facts.ci[0]) > 10:
        width = facts.ci[1] - facts.ci[0]
        p = (facts.policy_value or 85.0) / 100.0
        needed = max(facts.episodes + 1, episodes_for_half_width(5.0, p=p))
        items.append(
            _item(
                "results.wide_ci",
                stage,
                Severity.WARNING,
                "The confidence interval is too wide to decide",
                f"The score is {facts.policy_value:.1f}% with a {width:.1f}-point interval "
                f"({facts.ci[0]:.1f}–{facts.ci[1]:.1f}). That range contains both 'good' and "
                "'not good enough'.",
                evidence={
                    "policy_value": facts.policy_value,
                    "ci95_pct": facts.ci,
                    "episodes": facts.episodes,
                },
                actions=[
                    GuidanceAction(
                        kind="run_episodes",
                        label=f"Run {needed} episodes",
                        payload={"episodes_per_round": needed},
                    ),
                    _guide("19.4", "Read the interval playbook"),
                ],
            )
        )

    if facts.best_baseline is not None and facts.policy_value is not None:
        if facts.policy_value <= facts.best_baseline + 1.0:
            items.append(
                _item(
                    "results.worse_than_baseline",
                    stage,
                    Severity.WARNING,
                    "The policy is no better than the trivial baseline",
                    f"The policy scores {facts.policy_value:.1f}% against a best baseline of "
                    f"{facts.best_baseline:.1f}%. Training has not bought you anything yet.",
                    evidence={
                        "policy_value": facts.policy_value,
                        "best_baseline": facts.best_baseline,
                        "uplift": facts.uplift,
                    },
                    actions=[_guide("19.3", "It scores well but behaves badly")],
                )
            )

    metrics = [
        (r.get("metric"), r.get("status"))
        for r in facts.rounds
        if isinstance(r.get("metric"), (int, float))
    ]
    if len(metrics) >= 3:
        last_three = [float(m[0]) for m in metrics[-3:]]
        deltas = [abs(last_three[i + 1] - last_three[i]) for i in range(len(last_three) - 1)]
        below_target = facts.target is None or (facts.policy_value or 0) < facts.target
        if all(d < 1.5 for d in deltas) and below_target:
            items.append(
                _item(
                    "results.plateau",
                    stage,
                    Severity.WARNING,
                    "The last rounds stopped improving",
                    f"The last three rounds moved by {', '.join(f'{d:.1f}' for d in deltas)} "
                    "points. More of the same is unlikely to change the answer.",
                    evidence={"recent_metrics": last_three, "deltas": deltas},
                    actions=[
                        _guide("19.1", "Read the plateau playbook"),
                        _say(
                            "The run has plateaued. Change one thing at a time: the reward "
                            "term, the observation set, or the episodes per round.",
                            "Diagnose the plateau",
                        ),
                    ],
                )
            )

    if facts.corpus_regressions:
        items.append(
            _item(
                "results.corpus_regressions",
                stage,
                Severity.WARNING,
                "Previously approved behaviour regressed",
                f"{len(facts.corpus_regressions)} corpus case(s) that used to pass now fail. "
                "The new score is hiding a trade you did not ask for.",
                evidence={
                    "regressed_case_ids": facts.corpus_regressions[:10],
                    "corpus_pass_rate_pct": facts.corpus_pass_rate_pct,
                },
                actions=[_guide("15.3", "Why a change was kept or rolled back")],
            )
        )

    if facts.crossfire_reasons:
        items.append(
            _item(
                "results.crossfire_blockers",
                stage,
                Severity.WARNING,
                "The result cannot be called proven yet",
                "; ".join(facts.crossfire_reasons[:2]),
                evidence={
                    "crossfire_verdict": facts.crossfire_verdict,
                    "reasons": facts.crossfire_reasons[:4],
                },
                actions=[_guide("15.1", "What earns a win")],
            )
        )

    if facts.illegal_actions_pct > 0:
        items.append(
            _item(
                "results.illegal_actions",
                stage,
                Severity.WARNING,
                "The policy chose illegal actions",
                f"{facts.illegal_actions_pct:.1f}% of decisions were masked as illegal. The "
                "action space and the validator disagree about what is allowed.",
                evidence={"illegal_actions_pct": facts.illegal_actions_pct},
                actions=[_guide("21", "Troubleshooting")],
            )
        )

    return items


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_items(facts: ExportFacts) -> list[GuidanceItem]:
    items: list[GuidanceItem] = []
    stage = CoachStage.EXPORT

    if not facts.has_scorecard:
        items.append(
            _item(
                "export.no_scorecard",
                stage,
                Severity.WARNING,
                "The bundle has no scorecard",
                "An exported policy without its held-out evidence is a binary nobody can audit.",
                evidence={"has_scorecard": False},
                actions=[_guide("13", "What the scorecard contains")],
            )
        )

    if not facts.crossfire_closed:
        items.append(
            _item(
                "export.crossfire_open",
                stage,
                Severity.TIP,
                "Crossfire has not closed a verdict",
                "The policy can be exported, but the claim 'this works' is still unearned.",
                evidence={"crossfire_closed": False},
                actions=[_guide("15.1", "What earns a win")],
            )
        )

    if facts.deployment_agreement_pct is not None and facts.deployment_agreement_pct < 99.0:
        items.append(
            _item(
                "export.deployment_mismatch",
                stage,
                Severity.WARNING,
                "The deployed policy disagrees with the evaluated one",
                f"Action agreement is {facts.deployment_agreement_pct:.1f}%. The scorecard "
                "describes a policy that is not the one in the bundle.",
                evidence={"action_agreement_pct": facts.deployment_agreement_pct},
                actions=[_guide("19.6", "It behaves differently after export")],
            )
        )

    if facts.onnx_parity_diff is not None and facts.onnx_parity_diff > 1e-4:
        items.append(
            _item(
                "export.onnx_parity",
                stage,
                Severity.WARNING,
                "ONNX output diverges from the trained policy",
                f"The maximum numeric difference is {facts.onnx_parity_diff:g}, above the "
                "parity threshold. Do not ship this bundle.",
                evidence={"onnx_parity_diff": facts.onnx_parity_diff},
                actions=[_guide("21", "Troubleshooting")],
            )
        )

    return items
