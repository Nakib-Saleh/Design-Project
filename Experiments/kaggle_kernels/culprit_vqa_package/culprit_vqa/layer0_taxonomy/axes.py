"""The five axes of the factor space (proposal §4.1)."""

from enum import Enum


class Modality(str, Enum):
    """Where the interfering signal lives."""

    VISUAL = "visual"
    TEXTUAL = "textual"
    CROSS_MODAL = "cross_modal"


class Relevance(str, Enum):
    """Whether the signal bears on the answer."""

    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"


class Conflict(str, Enum):
    """Relation to the decisive visual evidence."""

    CONSISTENT = "consistent"
    CONTRADICTORY = "contradictory"


class Culturality(str, Enum):
    """Whether cultural priors are engaged."""

    NEUTRAL = "neutral"
    CULTURALLY_LOADED = "culturally_loaded"


class Knowledge(str, Enum):
    """Whether long-tail knowledge is needed."""

    PERCEPTUAL = "perceptual"
    KNOWLEDGE_REQUIRED = "knowledge_required"
