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

from typing import Optional, Tuple, Union

import pytest

from wsidicom.metadata.sample import (
    IssuerOfIdentifier,
    LocalIssuerOfIdentifier,
    SpecimenIdentifier,
    UniversalIssuerOfIdentifier,
    UniversalIssuerType,
)


class TestIssuerOfIdentifier:
    @pytest.mark.parametrize(
        ["value", "expected"],
        [
            ["issuer", LocalIssuerOfIdentifier("issuer")],
            [
                "^issuer^UUID",
                UniversalIssuerOfIdentifier("issuer", UniversalIssuerType.UUID),
            ],
            [
                "local^issuer^UUID",
                UniversalIssuerOfIdentifier(
                    "issuer", UniversalIssuerType.UUID, "local"
                ),
            ],
        ],
    )
    def test_from_hl7v2(self, value: str, expected: IssuerOfIdentifier):
        # Arrange

        # Act
        actual = IssuerOfIdentifier.from_hl7v2(value)

        # Assert
        assert actual == expected

    @pytest.mark.parametrize(
        ["issuer", "expected"],
        [
            [LocalIssuerOfIdentifier("issuer"), "issuer"],
            [
                UniversalIssuerOfIdentifier("issuer", UniversalIssuerType.UUID),
                "^issuer^UUID",
            ],
            [
                UniversalIssuerOfIdentifier(
                    "issuer", UniversalIssuerType.UUID, "local"
                ),
                "local^issuer^UUID",
            ],
        ],
    )
    def test_to_hlv2(self, issuer: IssuerOfIdentifier, expected: str):
        # Arrange

        # Act
        actual = issuer.to_hl7v2()

        # Assert
        assert actual == expected


class TestSpecimenIdentifier:
    @pytest.mark.parametrize(
        ["first", "second", "expected"],
        [
            ["identifier", "identifier", True],
            ["identifier", "other", False],
            ["identifier", SpecimenIdentifier("identifier"), True],
            ["identifier", SpecimenIdentifier("other"), False],
            [SpecimenIdentifier("identifier"), "identifier", True],
            [SpecimenIdentifier("identifier"), "other", False],
            [SpecimenIdentifier("identifier"), SpecimenIdentifier("identifier"), True],
            [SpecimenIdentifier("identifier"), SpecimenIdentifier("other"), False],
            [
                "identifier",
                SpecimenIdentifier("identifier", LocalIssuerOfIdentifier("local")),
                False,
            ],
            [
                SpecimenIdentifier("identifier", LocalIssuerOfIdentifier("local")),
                "identifier",
                False,
            ],
            [
                SpecimenIdentifier("identifier", LocalIssuerOfIdentifier("local")),
                SpecimenIdentifier("identifier"),
                False,
            ],
            [
                SpecimenIdentifier("identifier", LocalIssuerOfIdentifier("local")),
                SpecimenIdentifier("identifier", LocalIssuerOfIdentifier("local")),
                True,
            ],
            [
                SpecimenIdentifier("identifier", LocalIssuerOfIdentifier("local")),
                SpecimenIdentifier("identifier", LocalIssuerOfIdentifier("other")),
                False,
            ],
        ],
    )
    def test_equal(
        self,
        first: Union[str, SpecimenIdentifier],
        second: Union[str, SpecimenIdentifier],
        expected: bool,
    ):
        # Arrange

        # Act
        actual = first == second

        # Assert
        assert actual == expected

    @pytest.mark.parametrize(
        ["identifier", "expected"],
        [
            [SpecimenIdentifier("identifier"), ("identifier", None)],
            [
                SpecimenIdentifier("identifier", LocalIssuerOfIdentifier("local")),
                ("identifier", "local"),
            ],
            [
                SpecimenIdentifier(
                    "identifier",
                    UniversalIssuerOfIdentifier("issuer", UniversalIssuerType.UUID),
                ),
                ("identifier", "^issuer^UUID"),
            ],
            [
                SpecimenIdentifier(
                    "identifier",
                    UniversalIssuerOfIdentifier(
                        "issuer", UniversalIssuerType.UUID, "local"
                    ),
                ),
                ("identifier", "local^issuer^UUID"),
            ],
        ],
    )
    def test_to_string_identifier_and_issuer(
        self, identifier: SpecimenIdentifier, expected: Tuple[str, Optional[str]]
    ):
        # Arrange

        # Act
        actual = identifier.to_string_identifier_and_issuer()

        # Assert
        assert actual == expected

    @pytest.mark.parametrize(
        ["identifier", "expected"],
        [["identifier", ("identifier", None)]],
    )
    def test_get_string_identifier_and_issuer(
        self,
        identifier: Union[str, SpecimenIdentifier],
        expected: Tuple[str, Optional[str]],
    ):
        # Arrange

        # Act
        actual = SpecimenIdentifier.get_string_identifier_and_issuer(identifier)

        # Assert
        assert actual == expected
