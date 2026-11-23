"""Settings of the hidden demand shock that confounds the logged margins.

In the confounded scenario a hidden shock raises the attractiveness of some
packages and, at the same time, the margin the historical pricing rule charges for
them. The weighted experiment is the main model trained with stabilized inverse
propensity weights of the logged margins.
"""

from configs.modeling import LABELED, POINTWISE, experiment
from src.settings.confounder_models import ConfounderSettings


CONFOUNDER = ConfounderSettings(
    enabled=False,
    shock_share=0.2,
    utility_shift=2.0,
    margin_shift=0.08,
    seed=97,
    data_folder="data_confounded",
    shock_column="hid_demand_shock",
    weighted_experiment=experiment("cosines_and_labels_weighted", LABELED, POINTWISE),
)
