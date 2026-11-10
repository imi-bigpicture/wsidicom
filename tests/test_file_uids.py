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

import pytest
from pydicom.uid import UID

from wsidicom.uid import FileUids, SlideUids

SLIDE_UIDS = SlideUids(UID("1.2.1"), UID("1.2.2"), UID("1.2.3"))


@pytest.mark.unittest
class TestFileUids:
    def test_equal_uids_are_equal(self):
        # Arrange
        first = FileUids(UID("1.2.4"), None, SLIDE_UIDS)
        second = FileUids(UID("1.2.4"), None, SLIDE_UIDS)

        # Act
        equal = first == second

        # Assert
        assert equal
        assert hash(first) == hash(second)

    def test_parts_of_a_concatenation_are_not_equal(self):
        # Arrange
        first = FileUids(UID("1.2.4"), UID("1.2.6"), SLIDE_UIDS)
        second = FileUids(UID("1.2.5"), UID("1.2.6"), SLIDE_UIDS)

        # Act
        equal = first == second

        # Assert
        assert not equal

    def test_parts_of_a_concatenation_match_concatenation(self):
        # Arrange
        first = FileUids(UID("1.2.4"), UID("1.2.6"), SLIDE_UIDS)
        second = FileUids(UID("1.2.5"), UID("1.2.6"), SLIDE_UIDS)

        # Act
        matches = first.matches_concatenation(second)

        # Assert
        assert matches

    def test_files_without_concatenation_do_not_match_concatenation(self):
        # Arrange
        uids = FileUids(UID("1.2.4"), None, SLIDE_UIDS)

        # Act
        matches = uids.matches_concatenation(uids)

        # Assert
        assert not matches

    def test_different_concatenations_do_not_match_concatenation(self):
        # Arrange
        first = FileUids(UID("1.2.4"), UID("1.2.6"), SLIDE_UIDS)
        second = FileUids(UID("1.2.5"), UID("1.2.7"), SLIDE_UIDS)

        # Act
        matches = first.matches_concatenation(second)

        # Assert
        assert not matches
