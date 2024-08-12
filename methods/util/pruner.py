from optuna.pruners import BasePruner
from optuna.trial._state import TrialState

# Defines a pruner for optuna that prunes trials that are not promissing
class LastPlacePruner(BasePruner):
    def __init__(self, warmup_steps, warmup_trials):
        self._warmup_steps = warmup_steps
        self._warmup_trials = warmup_trials

    def prune(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> bool:
        # Get the latest score reported from this trial
        step = trial.last_step

        if step:  # trial.last_step == None when no scores have been reported yet
            this_score = trial.intermediate_values[step]

            # Get scores from other trials in the study reported at the same step
            completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
            other_scores = [
                t.intermediate_values[step]
                for t in completed_trials
                if step in t.intermediate_values
            ]
            other_scores = sorted(other_scores)

            if step >= self._warmup_steps and len(other_scores) > self._warmup_trials:
              if this_score + 0.2 < other_scores[-1]:
                print(f"prune() True: Trial {trial.number}, Step {step}, Score {this_score}")
                return True
              elif this_score < 0.0:
                print(f"prune() True: Trial {trial.number}, Step {step}, Score {this_score}")
                return True


        return False

