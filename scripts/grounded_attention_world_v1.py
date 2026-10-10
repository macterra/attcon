"""Unfrozen outcome-grounded world primitives; no attention-model qualification."""
from dataclasses import dataclass
from typing import Callable
import torch
from isolated_orientation_task_v1 import stimuli, render


@dataclass(frozen=True)
class OrientationWorld:
    labels:torch.Tensor
    contrasts:torch.Tensor
    mask:torch.Tensor
    phase:torch.Tensor
    command_mapping:torch.Tensor

    @classmethod
    def sample(cls,seed:int):
        _,_,nuisance=stimuli(seed,4)
        labels=torch.randint(2,(4,),generator=torch.Generator().manual_seed(seed+3000000))
        generator=torch.Generator().manual_seed(seed+2000000)
        contrasts=torch.randint(6,(4,),generator=generator)
        mapping=torch.randperm(4,generator=generator)
        return cls(labels.clone(),contrasts,nuisance['mask'].clone(),nuisance['phase'].clone(),mapping)

    def remap(self,mapping):
        mapping=torch.as_tensor(mapping,dtype=torch.long)
        assert torch.equal(mapping.sort().values,torch.arange(4))
        return OrientationWorld(self.labels.clone(),self.contrasts.clone(),self.mask.clone(),self.phase.clone(),mapping.clone())

    def image(self,location:int,noise:torch.Tensor):
        assert 0<=location<4 and noise.shape==(3,16,16) and torch.isfinite(noise).all()
        # Generate the same finite sensory conversion at the location's contrast.
        images=render(self.labels[location:location+1],self.mask[location:location+1],
            self.phase[location:location+1],noise[None,None].expand(1,6,3,16,16))
        return images[0,int(self.contrasts[location])].clone()


@dataclass(frozen=True)
class OrientationMemory:
    logits:torch.Tensor
    observed:torch.Tensor
    last_update:torch.Tensor
    time:int=0

    @classmethod
    def empty(cls):
        return cls(torch.zeros(4,dtype=torch.float64),torch.zeros(4,dtype=torch.bool),torch.full((4,),-1,dtype=torch.long))

    def advance(self):
        # Static world/latest-glimpse rule: time does not alter stored evidence.
        return OrientationMemory(self.logits.clone(),self.observed.clone(),self.last_update.clone(),self.time+1)

    def observe(self,location:int,logit:float):
        assert 0<=location<4 and torch.isfinite(torch.tensor(logit))
        logits=self.logits.clone();observed=self.observed.clone();last=self.last_update.clone()
        logits[location]=logit;observed[location]=True;last[location]=self.time
        return OrientationMemory(logits,observed,last,self.time)

    def answer(self,location:int):
        assert 0<=location<4
        return int(self.logits[location]>0)  # horizontal tie/unseen guess

    def uncalibrated_orientation_coordinates(self):
        # Normalized raw-logit transform, NOT a calibrated probability/posterior.
        # It is never multiplied by attention success.
        vertical=self.logits.sigmoid()
        return torch.stack((1-vertical,vertical),1)


@dataclass(frozen=True)
class DeliveredObservation:
    requested_command:int
    delivered_location:int
    sensory_logit:float
    time:int

    def model_input(self):
        # No world orientation, contrast, task correctness or condition flag.
        return {'command':self.requested_command,'delivered_location':self.delivered_location,
            'sensory_logit':self.sensory_logit,'time':self.time}


def execute(world:OrientationWorld,memory:OrientationMemory,command:int,
            noise:torch.Tensor,sensor:Callable[[torch.Tensor],float]):
    assert 0<=command<4
    location=int(world.command_mapping[command])
    image=world.image(location,noise)
    logit=float(sensor(image))
    updated=memory.advance().observe(location,logit)
    return updated,DeliveredObservation(command,location,logit,updated.time)


def evaluator_correctness(world:OrientationWorld,memory:OrientationMemory):
    """Training/evaluation target only; never a historical observation field."""
    return (memory.logits>0).long()==world.labels


def counterfactual_branches(world,memory,noises,sensor):
    """Execute all commands on immutable pre-command states; no forecast input."""
    assert noises.ndim==5 and noises.shape[1:]==(4,3,16,16)
    branches=[]
    for repetition in range(len(noises)):
        group=[]
        for command in range(4):
            post,observation=execute(world,memory,command,noises[repetition,command],sensor)
            group.append({'memory':post,'observation':observation,
                'evaluator_correctness':evaluator_correctness(world,post)})
        branches.append(group)
    return branches
