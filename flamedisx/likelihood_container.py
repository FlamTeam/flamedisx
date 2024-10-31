import flamedisx as fd

export, __all__ = fd.exporter()


@export
class LikelihoodContainer():
    """
    """
    def __init__(self, sources, arguments, bactch_size=1000, log_constraint=None,
                 expected_signal_counts=dict(), expected_background_counts=dict()):
        self.sources = sources
        self.arguments = arguments
        self.bactch_size = bactch_size
        self.log_constraint = log_constraint
        self.expected_signal_counts = expected_signal_counts
        self.expected_background_counts = expected_background_counts