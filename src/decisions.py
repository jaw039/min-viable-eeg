"""Protocol decisions outside the original locked list, with their sign-off status.

Two choices the runner depends on were not in the pipeline specification:
how unselected channels are removed, and which electrodes form the
sensorimotor pool. They are recorded here with their status, so Methods can
cite them and a config edit cannot silently change one.
"""

from src.channels import SENSORIMOTOR_POOL


class ProtocolDecisionError(RuntimeError):
    """The configuration disagrees with a recorded protocol decision."""


DECISIONS = {
    "reduction_mode": {
        "value": "reduce",
        "status": "proposed",
        "owner": "Kiran",
        "since": "2026-08-17",
        "note": "unselected channels are removed from the model input rather than "
                "zero-masked; config key `reduction_mode`",
    },
    "sensorimotor_pool": {
        "value": list(SENSORIMOTOR_POOL),
        "status": "proposed",
        "owner": "Kiran",
        "since": "2026-09-02",
        "note": "17-electrode motor strip for the restricted arm; declared in src/channels.py",
    },
}


def decision_value(cfg: dict, name: str):
    """The value in force for a recorded decision, checked against the config.

    Raises ProtocolDecisionError if the config drifted from the recorded
    decision: changing one is a protocol change and goes through this file.
    """
    if name not in DECISIONS:
        raise KeyError("no recorded decision named {!r}".format(name))
    recorded = DECISIONS[name]["value"]
    configured = cfg.get(name, recorded)
    if configured != recorded:
        raise ProtocolDecisionError(
            "config sets {}={!r} but the recorded decision is {!r}; update "
            "src/decisions.py (with sign-off) before running".format(name, configured, recorded)
        )
    return configured


def decision_status(name: str) -> str:
    return DECISIONS[name]["status"]


def unresolved():
    """Decisions still awaiting sign-off, for Methods and the README."""
    return sorted(n for n, d in DECISIONS.items() if d["status"] != "approved")
