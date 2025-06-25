#!/usr/bin/env python3
"""
Advanced reward component registry for RLlama.
Provides dynamic registration, discovery, and management of reward components.
"""

import inspect
import importlib
import pkgutil
from typing import Dict, Type, List, Any, Optional, Union, Callable, Set
from pathlib import Path
import logging
import warnings

from .base import BaseReward

class RewardRegistry:
    """
    Advanced registry for reward components with dynamic discovery and validation.
    Supports automatic component discovery, dependency tracking, and validation.
    """
    
    def __init__(self):
        """Initialize the reward registry."""
        self._registry: Dict[str, Type[BaseReward]] = {}
        self._aliases: Dict[str, str] = {}
        self._categories: Dict[str, Set[str]] = {}
        self._dependencies: Dict[str, Set[str]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Auto-discovery settings
        self._auto_discovery_enabled = True
        self._discovered_modules: Set[str] = set()
        
    def register(self, 
                reward_class_or_name: Union[Type[BaseReward], str] = None,
                aliases: Optional[List[str]] = None,
                category: Optional[str] = None,
                dependencies: Optional[List[str]] = None,
                metadata: Optional[Dict[str, Any]] = None) -> Callable:
        """
        Register a reward component class with advanced features.
        
        Args:
            reward_class_or_name: Reward class to register, or name to use
            aliases: Alternative names for the component
            category: Category for organization (e.g., "basic", "advanced", "llm")
            dependencies: List of required packages/modules
            metadata: Additional metadata about the component
            
        Returns:
            Decorator function that registers the class
        """
        def decorator(cls):
            # Determine the registration name
            if reward_class_or_name is None:
                name = cls.__name__
            elif isinstance(reward_class_or_name, str):
                name = reward_class_or_name
            else:
                name = reward_class_or_name.__name__
                cls = reward_class_or_name
                
            # Validate class
            if not issubclass(cls, BaseReward):
                raise TypeError(f"Reward class {name} must inherit from BaseReward")
                
            # Check for conflicts
            if name in self._registry:
                warnings.warn(f"Overriding existing reward component: {name}")
                
            # Register the class
            self._registry[name] = cls
            
            # Register aliases
            if aliases:
                for alias in aliases:
                    if alias in self._aliases:
                        warnings.warn(f"Overriding existing alias: {alias}")
                    self._aliases[alias] = name
                    
            # Set category
            if category:
                if category not in self._categories:
                    self._categories[category] = set()
                self._categories[category].add(name)
                
            # Set dependencies
            if dependencies:
                self._dependencies[name] = set(dependencies)
                
            # Store metadata
            component_metadata = metadata or {}
            component_metadata.update({
                'module': cls.__module__,
                'qualname': cls.__qualname__,
                'docstring': inspect.getdoc(cls),
                'file': inspect.getfile(cls) if hasattr(cls, '__file__') else None
            })
            self._metadata[name] = component_metadata
            
            self.logger.debug(f"Registered reward component: {name}")
            return cls
            
        # Handle direct class registration
        if isinstance(reward_class_or_name, type) and issubclass(reward_class_or_name, BaseReward):
            return decorator(reward_class_or_name)
            
        return decorator
    
    def get(self, name: str) -> Optional[Type[BaseReward]]:
        """
        Get a reward component class by name or alias.
        
        Args:
            name: Name or alias of the reward component
            
        Returns:
            The reward component class if found, None otherwise
        """
        # Check direct name first
        if name in self._registry:
            return self._registry[name]
            
        # Check aliases
        if name in self._aliases:
            actual_name = self._aliases[name]
            return self._registry.get(actual_name)
            
        # Try auto-discovery if enabled
        if self._auto_discovery_enabled:
            self._auto_discover_components()
            
            # Try again after discovery
            if name in self._registry:
                return self._registry[name]
            if name in self._aliases:
                actual_name = self._aliases[name]
                return self._registry.get(actual_name)
                
        return None
    
    def create(self, 
               name: str, 
               validate_dependencies: bool = True,
               **kwargs) -> Optional[BaseReward]:
        """
        Create an instance of a reward component with validation.
        
        Args:
            name: Name of the reward component
            validate_dependencies: Whether to validate dependencies
            **kwargs: Arguments to pass to the constructor
            
        Returns:
            An instance of the reward component if successful, None otherwise
        """
        reward_class = self.get(name)
        if reward_class is None:
            self.logger.error(f"Reward component not found: {name}")
            return None
            
        # Validate dependencies
        if validate_dependencies and name in self._dependencies:
            missing_deps = self._check_dependencies(name)
            if missing_deps:
                self.logger.error(f"Missing dependencies for {name}: {missing_deps}")
                return None
                
        try:
            # Filter kwargs to only include valid parameters
            valid_params = self._get_valid_parameters(reward_class)
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
            
            # Warn about unused parameters
            unused_params = set(kwargs.keys()) - set(filtered_kwargs.keys())
            if unused_params:
                self.logger.warning(f"Unused parameters for {name}: {unused_params}")
                
            return reward_class(**filtered_kwargs)
            
        except Exception as e:
            self.logger.error(f"Error creating instance of {name}: {e}")
            return None
    
    def list_components(self, 
                       category: Optional[str] = None,
                       include_aliases: bool = False) -> List[str]:
        """
        List all registered reward components.
        
        Args:
            category: Filter by category (optional)
            include_aliases: Whether to include aliases in the list
            
        Returns:
            List of component names
        """
        if category:
            components = list(self._categories.get(category, set()))
        else:
            components = list(self._registry.keys())
            
        if include_aliases:
            components.extend(list(self._aliases.keys()))
            
        return sorted(components)
    
    def list_categories(self) -> List[str]:
        """
        List all available categories.
        
        Returns:
            List of category names
        """
        return sorted(self._categories.keys())
    
    def describe_component(self, name: str) -> Dict[str, Any]:
        """
        Get detailed information about a reward component.
        
        Args:
            name: Name of the reward component
            
        Returns:
            Dictionary with comprehensive component information
        """
        # Resolve alias to actual name
        actual_name = name
        if name in self._aliases:
            actual_name = self._aliases[name]
            
        reward_class = self._registry.get(actual_name)
        if reward_class is None:
            return {"error": f"Reward component {name} not found"}
            
        # Get constructor signature
        signature = inspect.signature(reward_class.__init__)
        parameters = {}
        
        for param_name, param in signature.parameters.items():
            if param_name in ("self", "kwargs"):
                continue
                
            param_info = {
                "required": param.default is inspect.Parameter.empty,
                "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any"
            }
            
            if param.default is not inspect.Parameter.empty:
                param_info["default"] = param.default
                
            parameters[param_name] = param_info
        
        # Get metadata
        metadata = self._metadata.get(actual_name, {})
        
        # Find category
        component_category = None
        for cat, components in self._categories.items():
            if actual_name in components:
                component_category = cat
                break
                
        # Get aliases
        component_aliases = [alias for alias, target in self._aliases.items() if target == actual_name]
        
        return {
            "name": actual_name,
            "aliases": component_aliases,
            "category": component_category,
            "description": metadata.get("docstring", ""),
            "parameters": parameters,
            "dependencies": list(self._dependencies.get(actual_name, set())),
            "module": metadata.get("module"),
            "file": metadata.get("file"),
            "metadata": {k: v for k, v in metadata.items() 
                        if k not in ["docstring", "module", "file"]}
        }
    
    def validate_component(self, name: str) -> Dict[str, Any]:
        """
        Validate a reward component and its dependencies.
        
        Args:
            name: Name of the reward component
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "dependencies_met": True,
            "missing_dependencies": []
        }
        
        # Check if component exists
        reward_class = self.get(name)
        if reward_class is None:
            validation_result["errors"].append(f"Component {name} not found")
            return validation_result
            
        # Validate class structure
        try:
            # Check if it properly inherits from BaseReward
            if not issubclass(reward_class, BaseReward):
                validation_result["errors"].append("Class does not inherit from BaseReward")
                
            # Check if calculate method is implemented
            if not hasattr(reward_class, 'calculate'):
                validation_result["errors"].append("Missing calculate method")
            elif not callable(getattr(reward_class, 'calculate')):
                validation_result["errors"].append("calculate is not callable")
                
            # Check constructor signature
            try:
                signature = inspect.signature(reward_class.__init__)
                # Should accept **kwargs for flexibility
                has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD 
                               for p in signature.parameters.values())
                if not has_kwargs:
                    validation_result["warnings"].append("Constructor doesn't accept **kwargs")
                    
            except Exception as e:
                validation_result["warnings"].append(f"Could not inspect constructor: {e}")
                
        except Exception as e:
            validation_result["errors"].append(f"Error validating class structure: {e}")
            
        # Check dependencies
        if name in self._dependencies:
            missing_deps = self._check_dependencies(name)
            if missing_deps:
                validation_result["dependencies_met"] = False
                validation_result["missing_dependencies"] = missing_deps
                validation_result["errors"].append(f"Missing dependencies: {missing_deps}")
                
        # Set overall validity
        validation_result["valid"] = len(validation_result["errors"]) == 0
        
        return validation_result
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a reward component.
        
        Args:
            name: Name of the component to unregister
            
        Returns:
            True if successfully unregistered, False otherwise
        """
        # Check if it exists
        if name not in self._registry:
            return False
            
        # Remove from registry
        del self._registry[name]
        
        # Remove from categories
        for category, components in self._categories.items():
            components.discard(name)
            
        # Remove aliases
        aliases_to_remove = [alias for alias, target in self._aliases.items() if target == name]
        for alias in aliases_to_remove:
            del self._aliases[alias]
            
        # Remove dependencies and metadata
        self._dependencies.pop(name, None)
        self._metadata.pop(name, None)
        
        self.logger.info(f"Unregistered reward component: {name}")
        return True
    
    def clear(self) -> None:
        """Clear all registered components."""
        self._registry.clear()
        self._aliases.clear()
        self._categories.clear()
        self._dependencies.clear()
        self._metadata.clear()
        self.logger.info("Cleared all registered components")
    
    def export_registry(self) -> Dict[str, Any]:
        """
        Export the current registry state.
        
        Returns:
            Dictionary containing the complete registry state
        """
        return {
            "components": {name: cls.__module__ + "." + cls.__qualname__ 
                          for name, cls in self._registry.items()},
            "aliases": dict(self._aliases),
            "categories": {cat: list(components) for cat, components in self._categories.items()},
            "dependencies": {name: list(deps) for name, deps in self._dependencies.items()},
            "metadata": dict(self._metadata)
        }
    
    def _auto_discover_components(self) -> None:
        """Automatically discover and register reward components."""
        try:
            # Import components from the components package
            from . import components
            
            # Walk through all modules in the components package
            components_path = Path(components.__file__).parent
            
            for module_info in pkgutil.walk_packages([str(components_path)], 
                                                   prefix=f"{components.__name__}."):
                module_name = module_info.name
                
                if module_name in self._discovered_modules:
                    continue
                    
                try:
                    module = importlib.import_module(module_name)
                    self._discovered_modules.add(module_name)
                    
                    # Look for classes that inherit from BaseReward
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (issubclass(obj, BaseReward) and 
                            obj != BaseReward and 
                            obj.__module__ == module_name):
                            
                            # Auto-register if not already registered
                            if name not in self._registry:
                                self.register(obj)
                                
                except Exception as e:
                    self.logger.debug(f"Could not import module {module_name}: {e}")
                    
        except Exception as e:
            self.logger.debug(f"Auto-discovery failed: {e}")
    
    def _check_dependencies(self, name: str) -> List[str]:
        """
        Check if dependencies for a component are satisfied.
        
        Args:
            name: Name of the component
            
        Returns:
            List of missing dependencies
        """
        if name not in self._dependencies:
            return []
            
        missing = []
        for dep in self._dependencies[name]:
            try:
                importlib.import_module(dep)
            except ImportError:
                missing.append(dep)
                
        return missing
    
    def _get_valid_parameters(self, reward_class: Type[BaseReward]) -> Set[str]:
        """
        Get valid parameter names for a reward class constructor.
        
        Args:
            reward_class: The reward class
            
        Returns:
            Set of valid parameter names
        """
        try:
            signature = inspect.signature(reward_class.__init__)
            valid_params = set()
            
            for param_name, param in signature.parameters.items():
                if param_name == "self":
                    continue
                if param.kind == inspect.Parameter.VAR_KEYWORD:  # **kwargs
                    # Accept any parameter if **kwargs is present
                    return set()  # Empty set means accept all
                valid_params.add(param_name)
                
            return valid_params
            
        except Exception:
            # If we can't inspect, accept all parameters
            return set()

# Create global registry instance
reward_registry = RewardRegistry()

# Decorator and function aliases for backward compatibility
def register_reward_component(reward_class_or_name=None, **kwargs):
    """Alias for reward_registry.register with backward compatibility."""
    return reward_registry.register(reward_class_or_name, **kwargs)
    
def get_reward_component_class(name: str) -> Optional[Type[BaseReward]]:
    """Alias for reward_registry.get"""
    return reward_registry.get(name)
    
def get_reward_component(name: str, **kwargs) -> Optional[BaseReward]:
    """Alias for reward_registry.create"""
    return reward_registry.create(name, **kwargs)

# Legacy REWARD_REGISTRY dict for backward compatibility
class LegacyRegistryDict(dict):
    """Legacy registry dict that delegates to the new registry."""
    
    def __getitem__(self, key):
        component = reward_registry.get(key)
        if component is None:
            raise KeyError(f"Reward component not found: {key}")
        return component
        
    def __setitem__(self, key, value):
        reward_registry.register(value)
        
    def __contains__(self, key):
        return reward_registry.get(key) is not None
        
    def get(self, key, default=None):
        component = reward_registry.get(key)
        return component if component is not None else default
        
    def keys(self):
        return reward_registry.list_components()
        
    def values(self):
        return [reward_registry.get(name) for name in reward_registry.list_components()]
        
    def items(self):
        return [(name, reward_registry.get(name)) for name in reward_registry.list_components()]

REWARD_REGISTRY = LegacyRegistryDict()

# Auto-register basic components
try:
    from .components.common import LengthReward, ConstantReward
    reward_registry.register(LengthReward, category="basic")
    reward_registry.register(ConstantReward, category="basic")
except ImportError:
    pass  # Components not available yet

# Enable auto-discovery by default
reward_registry._auto_discovery_enabled = True
