"""Task identity is a controller input, independent of dataset provenance."""

from collections.abc import Mapping


def scenario_family(scenario):
    if not isinstance(scenario, Mapping):
        return ""
    task = scenario.get("task", {})
    if isinstance(task, Mapping) and task.get("family"):
        return str(task["family"])
    metadata = scenario.get("metadata", {})
    metadata = metadata.get("block_smb_monte_carlo", {}) if isinstance(metadata, Mapping) else {}
    if isinstance(metadata, Mapping) and metadata.get("family"):
        return str(metadata["family"])
    # Unlabelled traversal requests use geometry-based local goals. Special
    # success rules (stomping, boarding) require an explicit task contract;
    # the presence of a moving object alone does not imply mandatory boarding.
    return "mixed_section"
