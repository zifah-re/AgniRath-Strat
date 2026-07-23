"""
analysis/degraded_playbook.py — STUB (block 8.1, owner: Junior B+C).

Plan v3 §7: batch L1+L2 over capability-drop grid (array -10/-25/-50%,
pack -1 module, P_max -20%) x remaining-days -> lookup playbook document.
One-command re-plan path: replan --set <field>=<value> --from-position <x>.
"""

CAPABILITY_GRID = dict(
    array_scale=(0.9, 0.75, 0.5),
    packs_lost=(1,),
    p_max_scale=(0.8,),
)

def build_playbook(routes, base_car):
    raise NotImplementedError("block 8.1 — Junior B+C")
