# Copyright (С) ABBYY Production LLC, 2019. All rights reserved.
from catalyst.dl import registry

from runner import CustomRunner as Runner
from experiment import SimpleExperiment as Experiment

from . import callbacks
from . import models

registry.CALLBACKS.add_from_module(callbacks)
registry.MODELS.add_from_module(models)
