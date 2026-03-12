"""
Component registry module

this file is for registering extensible framework components

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ComponentRegistry:
    """
    Lightweight component registry.

    Parameters
    ----------
    name : str
        registry name
    entries : dict[str, type]
        registered component types
    """

    name: str
    entries: dict[str, type] = field(default_factory=dict)

    def register(self, key: str) -> Callable[[type], type]:
        """
        register a class under a string key

        Parameters
        ----------
        key : str
            registry key

        Returns
        -------
        Callable[[type], type]
            class decorator
        """

        def decorator(component_class: type) -> type:
            normalized_key = key.lower()
            if normalized_key in self.entries:
                raise KeyError(f"{normalized_key} is already registered in {self.name}")
            self.entries[normalized_key] = component_class
            return component_class

        return decorator

    def create(self, key: str, *args: Any, **kwargs: Any) -> Any:
        """
        instantiate a registered component

        Parameters
        ----------
        key : str
            registry key

        Returns
        -------
        Any
            component instance
        """

        component_class = self.get(key)
        return component_class(*args, **kwargs)

    def get(self, key: str) -> type:
        """
        retrieve a registered component class

        Parameters
        ----------
        key : str
            registry key

        Returns
        -------
        type
            registered class
        """

        normalized_key = key.lower()
        if normalized_key not in self.entries:
            available = ", ".join(sorted(self.entries))
            raise KeyError(f"{normalized_key} is not registered in {self.name}. available: {available}")
        return self.entries[normalized_key]

    def keys(self) -> list[str]:
        """
        list registered keys

        Returns
        -------
        list[str]
            sorted registry keys
        """

        return sorted(self.entries)

