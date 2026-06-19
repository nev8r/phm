"""
Training history container.
"""

from typing import Dict, List


class History:
    def __init__(self):
        self.rows: List[Dict] = []

    def append(self, row: Dict) -> None:
        self.rows.append(dict(row))

    def to_list(self) -> List[Dict]:
        return list(self.rows)
