#    Copyright 2021, 2023 SECTRA AB
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

from typing import Optional, Type

import pytest

from wsidicom.conceptcode import CidConceptCode, Code


@pytest.mark.unittest
class TestWsiDicomCode:
    @pytest.mark.parametrize(
        ["code_class", "code"],
        (
            (
                code_class,
                code,
            )
            for code_class in CidConceptCode.__subclasses__()
            for code in code_class.cid.values()
        ),
    )
    @pytest.mark.parametrize("case", [None, "lower", "upper"])
    def test_create_code_from_meaning(
        self, code_class: Type[CidConceptCode], code: Code, case: Optional[str]
    ):
        # Arrange
        if case == "lower":
            meaning = code.meaning.lower()
        elif case == "upper":
            meaning = code.meaning.upper()
        else:
            meaning = code.meaning

        # Act
        created_code = code_class(meaning)

        # Assert
        assert code == created_code
