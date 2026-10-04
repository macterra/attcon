"""Neutral task views: modeled state, actual observations and Controller actions.

Simulator truth and condition labels are not parameters of this interface.
"""
import copy
import torch
from .bound_content import COLORS, SHAPES
from .functional_interface import record, remap, without_relation


def task_record(visual, current, predicted_recovery, observed, row, query,
                command, response=None, executed=False, order=(0, 1, 2)):
    """Attach a current task decision to the existing neutral modeled-state view.

    A pending command has no emitted response. An executed decision must have
    either two valid attribute labels or the Controller's joint (-1,-1) abstention.
    No claim about physical correctness or unobserved acquisition is inferred.
    """
    batch = len(visual)
    for values in (query, command):
        if values.shape != (batch,) or values.dtype != torch.long or not bool(((values >= 0) & (values < 4)).all()):
            raise ValueError('query and command must be batch-long indices 0..3')
    if executed:
        if response is None or response.shape != (batch, 2) or response.dtype != torch.long:
            raise ValueError('executed decision requires batch-long attribute-pair responses')
        valid = ((response >= 0) & (response < 4)).all(-1) | (response == -1).all(-1)
        if not bool(valid.all()):
            raise ValueError('responses must be two labels or a joint abstention')
        if observed.shape[1] == 0 or not torch.equal(observed[:, -1, -4:].argmax(-1), command):
            raise ValueError('executed command must match the latest actual observation')
    elif response is not None:
        raise ValueError('pending command cannot have an emitted response')
    payload = record(visual, current, predicted_recovery, observed, row, order=order)
    decision = {'query': {'node': payload['output_node'], 'position': f'p{int(query[row])}'},
                'selected_command': f'k{int(command[row])}', 'command_executed': bool(executed)}
    if not executed:
        decision['response'] = {'status': 'pending', 'color': None, 'shape': None}
    elif bool((response[row] == -1).all()):
        decision['response'] = {'status': 'abstained', 'color': None, 'shape': None}
    else:
        decision['response'] = {'status': 'answered', 'color': COLORS[int(response[row, 0])],
                                'shape': SHAPES[int(response[row, 1])]}
    payload['task_decision'] = decision
    return payload


def remap_task(payload, node_names=None, command_names=None):
    nodes = node_names or {'n0': 'q7', 'n1': 'q2', 'n2': 'q9'}
    commands = command_names or {'k0': 'm7', 'k1': 'm2', 'k2': 'm9', 'k3': 'm4'}
    changed = remap(payload, nodes, commands)
    changed['task_decision']['query']['node'] = nodes[payload['task_decision']['query']['node']]
    changed['task_decision']['selected_command'] = commands[payload['task_decision']['selected_command']]
    return changed


def without_task_relation(payload):
    """Remove histories/forecasts, retaining supplied current state and task action."""
    return without_relation(copy.deepcopy(payload))
