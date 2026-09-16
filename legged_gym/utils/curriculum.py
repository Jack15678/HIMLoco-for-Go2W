"""Command coverage and task-aware terrain progression (torch only)."""
import torch


def stratified_extended_mask(terrain_columns, fraction=.2):
    """Random, fixed membership; round to the nearest count in each terrain column."""
    if not 0 <= fraction <= 1:
        raise ValueError('extended speed fraction must be in [0, 1]')
    mask = torch.zeros_like(terrain_columns, dtype=torch.bool)
    for column in terrain_columns.unique():
        ids = (terrain_columns == column).nonzero(as_tuple=False).flatten()
        order = torch.randperm(len(ids), device=ids.device)
        mask[ids[order[:round(len(ids) * fraction)]]] = True
    return mask
