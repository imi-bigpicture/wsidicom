#    Copyright 2023 SECTRA AB
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import datetime
from dataclasses import dataclass
from typing import (
    Optional,
    Sequence,
    Union,
)

from wsidicom.conceptcode import SpecimenSamplingProcedureCode
from wsidicom.metadata.sample import SamplingLocation, SpecimenIdentifier


@dataclass
class SerializedSamplingChainConstraint:
    """Simplified representation of a sampling chain constraint, replacing the sampling
    with the identifier of the sampled specimen and the index of the sampling step
    within the step sequence of the specimen."""

    identifier: Union[str, SpecimenIdentifier]
    sampling_step_index: int


@dataclass
class SerializedSampling:
    """Simplified representation of a `Sampling`, replacing the sampled specimen with
    the idententifier and sampling constratins with simplified sampling constraints."""

    method: SpecimenSamplingProcedureCode
    sampling_chain_constraints: Optional[
        Sequence[SerializedSamplingChainConstraint]
    ] = None
    date_time: Optional[datetime.datetime] = None
    location: Optional[SamplingLocation] = None
    description: Optional[str] = None
