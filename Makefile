.PHONY: test baselines random-hh train-q train-q-v2 eval-q compare compare-q compare-v2 plots train-plots analyze stats stats-v2 clean-results

test:
	pytest

baselines:
	python experiments/run_baselines.py \
		--n-cities 20 50 100 \
		--n-instances 10 \
		--two-opt-max-iterations 200

random-hh:
	python experiments/run_random_hh.py \
		--n-cities 20 50 100 \
		--n-instances 10 \
		--max-steps 50 \
		--initial-method nearest_neighbor

train-q:
	python experiments/train_q_learning.py \
		--episodes 500 \
		--n-cities 20 50 100 \
		--max-steps 50 \
		--initial-method nearest_neighbor \
		--learning-rate 0.1 \
		--discount-factor 0.95 \
		--epsilon 1.0 \
		--epsilon-min 0.05 \
		--epsilon-decay 0.995 \
		--out-dir results/q_learning

train-q-v2:
	python experiments/train_q_learning_v2.py \
		--episodes 3000 \
		--n-cities 20 50 100 \
		--max-steps 100 \
		--initial-method nearest_neighbor \
		--k-neighbors 10 \
		--perturbation-moves 3 \
		--reward-scale initial_length \
		--learning-rate 0.1 \
		--discount-factor 0.95 \
		--epsilon 1.0 \
		--epsilon-min 0.02 \
		--epsilon-decay 0.998 \
		--out-dir results/q_learning_v2

compare-v2:
	python experiments/run_comparison_v2.py \
		--q-table-path results/q_learning_v2/q_table_v2.pkl \
		--n-cities 20 50 100 \
		--n-instances 30 \
		--max-steps 100 \
		--initial-method nearest_neighbor \
		--k-neighbors 10 \
		--perturbation-moves 3 \
		--reward-scale initial_length \
		--out-dir results/comparison_v2

stats-v2:
	python experiments/statistical_tests.py \
		--raw-path results/comparison_v2/comparison_v2_raw.csv \
		--out-dir results/comparison_v2/stats \
		--compare \
			q_learning_hh_v2 random_hh_v2 \
			q_learning_hh_v2 cycle_hh_v2 \
			q_learning_hh_v2 always_first_2opt_v2 \
			q_learning_hh_v2 always_random_2opt_v2 \
			q_learning_hh_v2 always_bounded_2opt_v2 \
			q_learning_hh_v2 always_bounded_swap_v2 \
			q_learning_hh_v2 always_bounded_insertion_v2 \
			q_learning_hh_v2 always_perturb_bounded_2opt_v2

train-plots:
	python experiments/plot_training.py \
		--log-path results/q_learning/q_learning_train_log.csv \
		--out-dir results/q_learning/plots \
		--window 25

eval-q:
	python experiments/evaluate_q_learning.py \
		--q-table-path results/q_learning/q_table.pkl \
		--n-cities 20 50 100 \
		--n-instances 30 \
		--max-steps 50 \
		--initial-method nearest_neighbor \
		--out-dir results/q_learning_eval

compare:
	python experiments/run_comparison.py \
		--n-cities 20 50 100 \
		--n-instances 30 \
		--two-opt-max-iterations 200 \
		--two-opt-long-max-iterations 1000 \
		--hh-max-steps 50 \
		--out-dir results/comparison

compare-q:
	python experiments/run_comparison.py \
		--n-cities 20 50 100 \
		--n-instances 30 \
		--two-opt-max-iterations 200 \
		--two-opt-long-max-iterations 1000 \
		--hh-max-steps 50 \
		--q-table-path results/q_learning/q_table.pkl \
		--out-dir results/comparison_with_q

plots:
	python experiments/plot_comparison.py \
		--summary-path results/comparison_with_q/comparison_summary.csv \
		--raw-path results/comparison_with_q/comparison_raw.csv \
		--out-dir results/comparison_with_q/plots

analyze:
	python experiments/analyze_results.py \
		--train-log results/q_learning/q_learning_train_log.csv \
		--summary results/comparison_with_q/comparison_summary.csv \
		--raw results/comparison_with_q/comparison_raw.csv

stats:
	python experiments/statistical_tests.py \
		--raw-path results/comparison_with_q/comparison_raw.csv \
		--out-dir results/comparison_with_q/stats

clean-results:
	rm -rf results/baselines
	rm -rf results/random_hh
	rm -rf results/q_learning
	rm -rf results/q_learning_eval
	rm -rf results/comparison
	rm -rf results/comparison_with_q