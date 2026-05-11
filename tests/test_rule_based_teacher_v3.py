from tsp_hh.hyper_env_v3 import HyperHeuristicStateV3
from tsp_hh.rule_based_teacher_v3 import (
    RuleBasedTeacherV3,
    TeacherConfigV3,
    CATEGORY_CONSTRUCTION,
    CATEGORY_IMPROVEMENT,
    CATEGORY_PERTURBATION,
    teacher_probability,
)


def make_state(
    stagnation: int,
    last_category: int = -1,
) -> HyperHeuristicStateV3:
    return HyperHeuristicStateV3(
        current_length=100.0,
        best_length=100.0,
        initial_length=120.0,
        last_improvement=0.0,
        steps_without_improvement=stagnation,
        step_count=10,
        n_cities=50,
        last_category=last_category,
    )


def test_teacher_selects_improvement_when_not_stuck():
    teacher = RuleBasedTeacherV3()

    state = make_state(stagnation=0)

    assert teacher.select_category(state) == CATEGORY_IMPROVEMENT


def test_teacher_selects_perturbation_when_strongly_stuck():
    teacher = RuleBasedTeacherV3(
        TeacherConfigV3(
            strong_stagnation=20,
            restart_stagnation=60,
        )
    )

    state = make_state(stagnation=25)

    assert teacher.select_category(state) == CATEGORY_PERTURBATION


def test_teacher_selects_construction_when_very_stuck():
    teacher = RuleBasedTeacherV3(
        TeacherConfigV3(
            restart_stagnation=40,
        )
    )

    state = make_state(stagnation=45)

    assert teacher.select_category(state) == CATEGORY_CONSTRUCTION


def test_teacher_repairs_after_perturbation():
    teacher = RuleBasedTeacherV3()

    state = make_state(
        stagnation=40,
        last_category=CATEGORY_PERTURBATION,
    )

    assert teacher.select_category(state) == CATEGORY_IMPROVEMENT


def test_teacher_probability_starts_high_and_ends_low():
    p0 = teacher_probability(
        episode=0,
        total_episodes=100,
        start_prob=1.0,
        end_prob=0.0,
        decay_fraction=0.8,
    )

    p_end = teacher_probability(
        episode=99,
        total_episodes=100,
        start_prob=1.0,
        end_prob=0.0,
        decay_fraction=0.8,
    )

    assert p0 == 1.0
    assert p_end == 0.0