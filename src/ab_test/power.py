"""Plans the number of sessions an experiment needs to detect an effect."""


def required_sessions(
    standard_deviation: float,
    effect: float,
    alpha: float,
    power: float,
) -> int:
    """Computes the sessions per arm for a two-sided test of a mean difference."""
    raise NotImplementedError
