"""Neutral factual views of three-way learned states; no physical owner input."""
import copy
import torch
from .bound_content import COLORS, SHAPES
from .functional_model import buffer_contents


def selection(values):
    values = [round(float(v), 5) for v in values]
    return {'selection_distribution':dict(zip(('p0','p1','p2','p3'),values)),
            'selected_position':f'p{values.index(max(values))}' if max(values)>=.6 else None}


def record(visual, current, predicted_recovery, observed, row, order=(0,1,2),
           command_names=('k0','k1','k2','k3'), anticipated=None, modeled_step=None):
    """Round model predictions and observations, keeping anticipated events separate.

    ``observed`` and ``anticipated`` are [batch,time,20] event sequences. No
    simulator state, ownership label or scoring outcome is accepted here.
    """
    if tuple(sorted(order))!=(0,1,2) or len(set(command_names))!=4:
        raise ValueError('invalid identifier mapping')
    names = {physical:f'n{shown}' for shown,physical in enumerate(order)}
    now = buffer_contents(visual,current.access[:,0])
    future = buffer_contents(visual[:,None],predicted_recovery)
    def nodes(values, allocation=None, access=None):
        out=[]
        for p in order:
            buffer=p if p<2 else 0
            objects=[]
            for pos, dist in enumerate(values[buffer]):
                color=[round(float(v),5) for v in dist[:4]]
                shape=[round(float(v),5) for v in dist[4:]]
                obj={'position':f'p{pos}',
                    'color_distribution':dict(zip(COLORS,color)),
                    'shape_distribution':dict(zip(SHAPES,shape)),
                    'identified_color':COLORS[color.index(max(color))] if max(color)>=.6 else None,
                    'identified_shape':SHAPES[shape.index(max(shape))] if max(shape)>=.6 else None}
                if p<2 and access is not None:
                    q=[round(float(v),5) for v in access[:,p,pos]]
                    obj['recovery_by_delay']=dict(zip(('0','1','2'),q))
                    d=q[2]-q[0]
                    obj['unattended_trend']='declining' if d<-.02 else 'rising' if d>.02 else 'steady'
                objects.append(obj)
            node={'node':names[p],'objects':objects}
            if p<2 and allocation is not None:node.update(selection(allocation[p]))
            out.append(node)
        return out
    def history(xs):
        if xs is None:return []
        out=[]
        for obs in xs[row]:
            out.append({'command':command_names[int(obs[-4:].argmax())],
                'events':[{'node':names[p],
                    'allocation':[round(float(v),5) for v in obs[:8].reshape(2,4)[p]],
                    'acquisition':[round(float(v),5) for v in obs[8:16].reshape(2,4)[p]]}
                    for p in order if p<2]})
        return out
    return {'output_node':names[2],
        'modeled_step':observed.shape[1] if modeled_step is None else modeled_step,
        'observed_history':history(observed),'anticipated_history':history(anticipated),
        'predicted_current':nodes(now[row],current.allocation[row],current.access[row]),
        'predicted_by_command':[{'command':command_names[c],
            'nodes':nodes(future[row,c],current.effects[row,c])} for c in range(4)]}


def remap(payload, node_names=None, command_names=None):
    """Reversible presentation mapping, not a novel physical command mapping."""
    nodes=node_names or {'n0':'q7','n1':'q2','n2':'q9'}
    commands=command_names or {'k0':'m7','k1':'m2','k2':'m9','k3':'m4'}
    if len(set(nodes.values()))!=len(nodes) or len(set(commands.values()))!=len(commands):
        raise ValueError('identifier mapping must be injective')
    out=copy.deepcopy(payload)
    out['output_node']=nodes[out['output_node']]
    for key in ('observed_history','anticipated_history'):
        for event in out[key] or []:
            event['command']=commands[event['command']]
            for node in event['events']:node['node']=nodes[node['node']]
    for node in out['predicted_current']:node['node']=nodes[node['node']]
    for trial in out['predicted_by_command'] or []:
        trial['command']=commands[trial['command']]
        for node in trial['nodes']:node['node']=nodes[node['node']]
    return out


def without_relation(payload):
    out=copy.deepcopy(payload)
    out['observed_history']=None;out['anticipated_history']=None
    out['predicted_by_command']=None
    return out
