#    Copyright 2021 SECTRA AB
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

from typing import Any, Dict, Optional, Sequence


def str_indent(indent: int = 0) -> str:
    """Returns a string for indentation

    Parameters
    ----------
    indent: int
        Number of double whitespaces to indent

    Returns
    ----------
    str
        string with whitespaces

    """
    return ' ' * 2 * indent


def list_pretty_str(
    items: Sequence[Any],
    indent: int = 0,
    depth: Optional[int] = None,
    pre_new_lines: int = 0,
    list_new_lines: int = 1,
    space: bool = False
) -> str:
    """Returns a pretty-printed string items
    String is indented and new lines are added before the list
    and/or between each item in the list.
    Each item is pretty-printed with increased indent.

    Parameters
    ----------
    items: sequence
        List of items to pretty-print
    indent: int
        Indentation for the string and list
    depth: Optional[int] = None
        Depth to print, if item contains list
    pre_new_lines: int
        Number of new lines at beginning of string
    list_new_lines: int
        Number of new lines between list items
    space: bool
        Use space instead of newline

    Returns
    ----------
    str
        Pretty-printed string of items

    """
    delimiter = '\n'
    if space:
        delimiter = ' '
    return (
        delimiter * pre_new_lines
        + (delimiter * list_new_lines).join(
            [str_indent(indent) + f'[{i}]: '
             + item.pretty_str(indent+1, depth)
             for i, item in enumerate(items)]
         )
    )


def dict_pretty_str(
    items: Dict[Any, Any],
    indent: int = 0,
    depth: Optional[int] = None,
    pre_new_lines: int = 0,
    list_new_lines: int = 1,
    space: bool = False
) -> str:
    """Returns a pretty-printed string items
    String is indented and new lines are added before the list
    and/or between each item in the list.
    Each item is pretty-printed with increased indent.

    Parameters
    ----------
    items: sequence
        List of items to pretty-print
    indent: int
        Indentation for the string and list
    depth: Optional[int] = None
        Depth to print, if item contains list
    pre_new_lines: int
        Number of new lines at beginning of string
    list_new_lines: int
        Number of new lines between list items
    space: bool
        Use space instead of newline

    Returns
    ----------
    str
        Pretty-printed string of items

    """
    delimiter = '\n'
    if space:
        delimiter = ' '
    return (
        delimiter * pre_new_lines
        + (delimiter * list_new_lines).join(
            [str_indent(indent) + f'[{i}]: '
             + item.pretty_str(indent+1, depth)
             for i, item in enumerate(items.values())]
         )
    )
