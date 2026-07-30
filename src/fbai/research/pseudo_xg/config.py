"""Frozen source configuration for the historical pseudo-xG experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class PseudoXGConfig:
    """Exact effective configuration of the authoritative primary experiment."""

    estimator: str = "PoissonRegressor"
    estimator_target: str = "goals"
    estimator_predictors: tuple[str, str] = (
        "shots_on_target",
        "shots_off_target",
    )
    poisson_alpha: float = 1e-6
    poisson_max_iter: int = 1000
    poisson_tolerance: float = 1e-4
    poisson_fit_intercept: bool = True
    poisson_solver: str = "lbfgs"
    rolling_windows: tuple[int, int] = (5, 10)
    estimator_scope: str = "outer_training_fold_all_divisions"
    history_season_scope: str = "SeasonStartYear"
    history_division_scope: str = "Division"
    candidate: str = "LogisticRegression"
    candidate_base_feature_count: int = 52
    candidate_pseudo_xg_feature_count: int = 12
    imputation: str = "median"
    scaling: str = "standard"
    logistic_solver: str = "lbfgs"
    logistic_penalty: str = "l2"
    logistic_c: float = 1.0
    logistic_max_iter: int = 2000
    logistic_fit_intercept: bool = True
    logistic_class_weight: str | None = None
    logistic_random_state: int = 42
    logistic_tolerance: float = 1e-4

    def __post_init__(self) -> None:
        if self.estimator != "PoissonRegressor":
            raise ValueError("pseudo-xG estimator is fixed to PoissonRegressor")
        if self.estimator_target != "goals":
            raise ValueError("pseudo-xG target is fixed to goals")
        if self.estimator_predictors != ("shots_on_target", "shots_off_target"):
            raise ValueError("pseudo-xG predictors are fixed to target and off-target shots")
        if self.poisson_alpha != 1e-6 or self.poisson_max_iter != 1000:
            raise ValueError("pseudo-xG Poisson regularization and iteration limit are frozen")
        if self.poisson_tolerance <= 0.0 or not self.poisson_fit_intercept:
            raise ValueError("pseudo-xG Poisson tolerance/intercept settings are invalid")
        if self.poisson_solver != "lbfgs":
            raise ValueError("pseudo-xG Poisson solver is fixed to lbfgs")
        if self.rolling_windows != (5, 10):
            raise ValueError("pseudo-xG rolling windows are fixed to 5 and 10")
        if self.estimator_scope != "outer_training_fold_all_divisions":
            raise ValueError("pseudo-xG Poisson fitting is pooled across training divisions")
        if (
            self.history_season_scope != "SeasonStartYear"
            or self.history_division_scope != "Division"
        ):
            raise ValueError("pseudo-xG history is isolated by public season and division")
        if self.candidate != "LogisticRegression":
            raise ValueError("pseudo-xG downstream candidate is fixed to LogisticRegression")
        if self.candidate_base_feature_count != 52 or self.candidate_pseudo_xg_feature_count != 12:
            raise ValueError("pseudo-xG candidate is fixed to 52 plus 12 features")
        if self.imputation != "median" or self.scaling != "standard":
            raise ValueError("pseudo-xG preprocessing is fixed to median then standard scaling")
        if self.logistic_solver != "lbfgs" or self.logistic_penalty != "l2":
            raise ValueError("pseudo-xG Logistic Regression solver/penalty are frozen")
        if self.logistic_c != 1.0 or self.logistic_max_iter != 2000:
            raise ValueError("pseudo-xG Logistic Regression C/iteration limit are frozen")
        if not self.logistic_fit_intercept or self.logistic_class_weight is not None:
            raise ValueError("pseudo-xG Logistic Regression intercept/class weight are frozen")
        if self.logistic_random_state != 42 or self.logistic_tolerance <= 0.0:
            raise ValueError("pseudo-xG Logistic Regression seed/tolerance are frozen")

    @property
    def identifier(self) -> str:
        """Return the stable source-configuration identifier."""

        return "pxg_poisson_sot_off_alpha1e6_w5w10_lr64_seed42"

    @property
    def formula(self) -> str:
        """Return the recorded pseudo-xG conditional-mean formula."""

        return (
            "E[goals | shots_on_target, shots_off_target] = "
            "exp(intercept + beta_target * shots_on_target + "
            "beta_off * shots_off_target)"
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe configuration metadata."""

        values: dict[str, object] = asdict(self)
        values["estimator_predictors"] = list(self.estimator_predictors)
        values["rolling_windows"] = list(self.rolling_windows)
        values["identifier"] = self.identifier
        values["formula"] = self.formula
        return values
