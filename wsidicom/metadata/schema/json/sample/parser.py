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

from collections import UserDict
from typing import (
    Callable,
    Iterable,
    List,
    Union,
)

from wsidicom.metadata.schema.json.sample.model import (
    BaseSpecimenJsonModel,
)
from wsidicom.metadata.sample import (
    SampledSpecimen,
    BaseSpecimen,
    SpecimenIdentifier,
)


class SpecimenDictionary(UserDict[Union[str, SpecimenIdentifier], BaseSpecimen]):
    """Dictionary for specimens that creates missing specimens when accessed."""

    def __init__(
        self,
        create_missing: Callable[[Union[str, SpecimenIdentifier]], BaseSpecimen],
        *args,
        **kwargs,
    ):
        """Initialize dictionary.

        Parameters
        ----------
        create_missing: Callable[[Union[str, SpecimenIdentifier]], Specimen]
            Function that creates a specimen for a given identifier.
        """
        super().__init__(*args, **kwargs)
        self._create_missing = create_missing

    def __missing__(self, key: Union[str, SpecimenIdentifier]) -> BaseSpecimen:
        """Create missing specimen."""
        specimen = self._create_missing(key)
        self[key] = specimen
        return specimen


class SpecimenJsonParser:
    """Factory for creating specimens from json models."""

    def __init__(self, specimen_models: Iterable[BaseSpecimenJsonModel]):
        """Initiate factory.

        Parameters
        ----------
        specimen_models: Iterable[BaseSpecimenJsonModel]
            Json models to create specimens from.
        """
        self._specimens = SpecimenDictionary(self._make_specimen)
        self._specimen_models_by_identifier = {
            specimen.identifier: specimen for specimen in specimen_models
        }

    def create_specimens(self) -> List[BaseSpecimen]:
        """Create specimens

        Returns
        -------
        List[Specimen]
            List of created specimens.
        """
        for identifier in self._specimen_models_by_identifier:
            self._specimens[identifier] = self._make_specimen(identifier)
        sampled_specimens = [
            sampled_from.specimen.identifier
            for specimen in self._specimens.values()
            if isinstance(specimen, SampledSpecimen)
            for sampled_from in specimen.sampled_from_list
        ]
        return [
            specimen
            for specimen in self._specimens.values()
            if specimen.identifier not in sampled_specimens
        ]

    def _make_specimen(
        self,
        identifier: Union[str, SpecimenIdentifier],
    ) -> BaseSpecimen:
        """Create specimen by identifier from json model.

        Create nested specimens that the specimen is sampled from if needed.

        Parameters
        ----------
        identifier: Union[str, SpecimenIdentifier]
            Identifier of specimen to create

        Returns
        -------
        Specimen
            Specimen created from json model.

        """
        specimen_model = self._specimen_models_by_identifier[identifier]
        return specimen_model.from_json_model(self._specimens)
