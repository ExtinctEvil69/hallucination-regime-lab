"""Bayesian Hallucination Regime Lab.

Detection, forecasting, factor attribution and intervention for LLM
hallucination onsets using (covariate-dependent) Bayesian hidden Markov models.
Core modules depend only on NumPy/SciPy; the Ollama collector uses the
standard library HTTP client.
"""

SCHEMA_VERSION = "2.0"
STATES = ("G", "U", "H")
__version__ = "2.0.0"
