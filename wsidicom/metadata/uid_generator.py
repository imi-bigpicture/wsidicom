#    Copyright 2026 SECTRA AB
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

"""Per-role UID generation contract.

`UidGenerator` is an ABC with one method per UID role. Each per-entity role
receives the entity (or the parent `WsiMetadata` for WSI-wide UIDs), so
implementations can derive UIDs from content. `sop_uid()` receives the
already-built `pydicom.Dataset`. The `concatenation_*()` roles receive the level
dataset and are called once per concatenation, for grouping identities of a
notional source instance that is never itself written (so they are separate from
`sop_uid()`, which is per real written instance). `annotation_group_uid()` is
zero-arg.

`CallableUidGenerator` is the default implementation: every role delegates
to a single zero-arg callable, ignoring any entity argument.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable

from pydicom.dataset import Dataset
from pydicom.uid import UID, generate_uid

from wsidicom.metadata.pyramid import Pyramid
from wsidicom.metadata.sample import SlideSample
from wsidicom.metadata.series import Series
from wsidicom.metadata.study import Study
from wsidicom.metadata.wsi import WsiMetadata


class UidGenerator(ABC):
    """Per-role UID generation contract.

    Subclass directly when your generator has its own state (a DB connection,
    a filepath, a counter).
    """

    @abstractmethod
    def study_uid(self, study: Study) -> UID: ...

    @abstractmethod
    def series_uid(self, series: Series) -> UID: ...

    @abstractmethod
    def frame_of_reference_uid(self, metadata: WsiMetadata) -> UID: ...

    @abstractmethod
    def dimension_organization_uid(self, metadata: WsiMetadata) -> UID: ...

    @abstractmethod
    def sample_uid(self, sample: SlideSample) -> UID: ...

    @abstractmethod
    def pyramid_uid(self, pyramid: Pyramid) -> UID: ...

    @abstractmethod
    def sop_uid(self, dataset: Dataset) -> UID: ...

    @abstractmethod
    def concatenation_uid(self, dataset: Dataset) -> UID: ...

    @abstractmethod
    def concatenation_source_uid(self, dataset: Dataset) -> UID: ...

    @abstractmethod
    def annotation_group_uid(self) -> UID: ...


class CallableUidGenerator(UidGenerator):
    """`UidGenerator` that delegates every role to a single zero-arg callable.

    Examples
    --------
    Prefix every UID with a single root::

        gen = CallableUidGenerator(lambda: generate_uid(prefix=f"{ROOT}."))

    Specialize one role only::

        class LoggingGen(CallableUidGenerator):
            def study_uid(self, study):
                uid = super().study_uid(study)
                log.info("study UID %s for %s", uid, study.identifier)
                return uid
    """

    def __init__(self, generate: Callable[[], UID] = generate_uid):
        self._generate = generate

    def study_uid(self, study: Study) -> UID:
        return self._generate()

    def series_uid(self, series: Series) -> UID:
        return self._generate()

    def frame_of_reference_uid(self, metadata: WsiMetadata) -> UID:
        return self._generate()

    def dimension_organization_uid(self, metadata: WsiMetadata) -> UID:
        return self._generate()

    def sample_uid(self, sample: SlideSample) -> UID:
        return self._generate()

    def pyramid_uid(self, pyramid: Pyramid) -> UID:
        return self._generate()

    def sop_uid(self, dataset: Dataset) -> UID:
        return self._generate()

    def concatenation_uid(self, dataset: Dataset) -> UID:
        return self._generate()

    def concatenation_source_uid(self, dataset: Dataset) -> UID:
        return self._generate()

    def annotation_group_uid(self) -> UID:
        return self._generate()
