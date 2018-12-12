#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Interfaces in the backend used for modifier and objects.

The `backend` module is hosting the interfaces for each kind of
object and modifier used during the compression process. The character `_` is
being used as an indicator for the file being an interface description.
Objects use the indicator as a prefix and modifier as a suffix.
"""

# Objects
from pasc.backend._data import BaseData
from pasc.backend._array import BaseArray
from pasc.backend._sequence import BaseSequence
from pasc.backend._informationspace import BaseInformationSpace
from pasc.backend._informationcontext import BaseInformationContext
# from pasc.backend._coded import CodedInterface
# from pasc.backend._pasc import PascInterface

# Modifiers
from pasc.backend.mapper_ import BaseMapper
from pasc.backend.sequencer_ import BaseSequencer
from pasc.backend.builder_ import BaseBuilder
from pasc.backend.predictor_ import CorePredictor, MixedPredictor, NDPredictor
from pasc.backend.subtractor_ import BaseSubstractor
# from pasc.backend.encoder_ import EncoderInterface
# from pasc.backend.writer_ import WriterInterface
