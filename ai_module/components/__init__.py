"""Framework-agnostic component registry (the seam for Langflow). See ``registry.py``."""
from ai_module.components.registry import COMPONENTS, ComponentSpec, describe_components, get_component

__all__ = ["COMPONENTS", "ComponentSpec", "describe_components", "get_component"]
