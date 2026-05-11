from dataclasses import dataclass

from tsp_hh.hyper_env_v3 import HyperHeuristicStateV3


CATEGORY_CONSTRUCTION = 0
CATEGORY_IMPROVEMENT = 1
CATEGORY_PERTURBATION = 2


@dataclass
class TeacherConfigV3:
    """
    Configuration for the V3 rule-based teacher.

    The teacher chooses high-level categories:
    - construction
    - improvement
    - perturbation
    """

    mild_stagnation: int = 5
    medium_stagnation: int = 15
    strong_stagnation: int = 35
    restart_stagnation: int = 60

    force_improve_after_perturbation: bool = True


class RuleBasedTeacherV3:
    """
    Rule-based teacher for hierarchical V3 hyper-heuristic.

    Main idea:
    - If search is still improving, intensify with improvement.
    - If mildly stuck, perturb.
    - After perturbation, repair with improvement.
    - If strongly stuck, restart using construction.
    """

    def __init__(self, config: TeacherConfigV3 | None = None):
        self.config = config or TeacherConfigV3()

    def select_category(self, state: HyperHeuristicStateV3) -> int:
        """
        Select a high-level category based on the search state.
        """
        stagnation = state.steps_without_improvement

        # If the previous action was perturbation, repair immediately.
        if (
            self.config.force_improve_after_perturbation
            and state.last_category == CATEGORY_PERTURBATION
        ):
            return CATEGORY_IMPROVEMENT

        # Very strong stagnation: restart proposal.
        if stagnation >= self.config.restart_stagnation:
            return CATEGORY_CONSTRUCTION

        # Strong stagnation: perturb to escape local minimum.
        if stagnation >= self.config.strong_stagnation:
            return CATEGORY_PERTURBATION

        # Medium stagnation: alternate perturbation and improvement.
        if stagnation >= self.config.medium_stagnation:
            return CATEGORY_PERTURBATION

        # Mild stagnation: still try improvement.
        if stagnation >= self.config.mild_stagnation:
            return CATEGORY_IMPROVEMENT

        # Early or improving phase: intensify.
        return CATEGORY_IMPROVEMENT


def teacher_probability(
    episode: int,
    total_episodes: int,
    start_prob: float = 1.0,
    end_prob: float = 0.0,
    decay_fraction: float = 0.8,
) -> float:
    """
    Anneal teacher dependence over training.

    During early training, the teacher strongly guides the agent.
    Later, control is gradually transferred to the Q-learning agent.
    """
    if total_episodes <= 1:
        return end_prob

    decay_episodes = max(1, int(total_episodes * decay_fraction))

    if episode >= decay_episodes:
        return end_prob

    progress = episode / decay_episodes

    return start_prob + progress * (end_prob - start_prob)