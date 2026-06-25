"""
Registry module.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: zdh
Program date: 2026-06
Copyright: USTC

2026
"""

from typing import Any, Dict, Tuple


class Registry:
    """
    Store objects by string key with duplicate and missing-key checks.
    """

    def __init__(self, name: str):
        self.name = name
        self._items: Dict[str, Any] = {}

    def register(self, key: str, obj: Any = None):
        if obj is None:
            def decorator(item: Any) -> Any:
                self.register(key, item)
                return item

            return decorator

        if key in self._items:
            raise KeyError(f"{key!r} is already registered in {self.name!r}")
        self._items[key] = obj
        return obj

    def get(self, key: str) -> Any:
        if key not in self._items:
            raise KeyError(f"{key!r} is not registered in {self.name!r}")
        return self._items[key]

    def keys(self) -> Tuple[str, ...]:
        return tuple(self._items.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._items
