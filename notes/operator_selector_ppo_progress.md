# PPO Operator Selector Progress

Implemented a DRL hyper-heuristic environment for TSP.

Current action space:
- STOP
- 2-opt limited
- relocate best-of-k
- swap best-of-k
- perturb + 2-opt

Current best quick run:
results/operator_selector_ppo_quick_quality_best_memory

Observations:
- PPO improves over NN+2opt on training instances berlin52, eil51, st70.
- PPO matches NN+2opt on ch130 and kroA100 in the current quick test.
- PPO still mostly uses 2-opt and relocate.
- Perturbation is not yet being used.
- Added best-so-far memory so risky operators can be explored safely.

Next steps:
- Encourage perturbation usage.
- Train longer or with mixed start mode.
- Possibly add stronger final-gap reward.
- Evaluate on more TSPLIB instances.
