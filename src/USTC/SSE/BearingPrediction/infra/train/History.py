"""
Training history container.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: zdh
Program date: 2026-06
Copyright: USTC

2026
"""

from typing import Dict, List


class History:
    def __init__(self):
        self.rows: List[Dict] = []

    def append(self, row: Dict) -> None:
        self.rows.append(dict(row))

    def to_list(self) -> List[Dict]:
        return list(self.rows)
