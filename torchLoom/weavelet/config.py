"""
Configuration validation and type conversion utilities.
"""

from typing import Any, Type


class TypeConverter:
    """Handles type validation and conversion for configuration values."""

    @staticmethod
    def validate_and_convert_value(
        config_key: str, value: Any, expected_type: Type
    ) -> Any:
        """Validate and convert a configuration value to the expected type.
        
        Args:
            config_key: The configuration parameter name
            value: The value to convert
            expected_type: The expected type for the value
            
        Returns:
            The converted value
            
        Raises:
            TypeError: If the value cannot be converted to the expected type
        """
        # If value is already the correct type, return as-is
        if isinstance(value, expected_type):
            return value

        # Try to convert the value
        try:
            if expected_type == bool:
                # Handle boolean conversion from strings
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes", "on")
                return bool(value)
            elif expected_type == int:
                return int(float(value))  # Handle "1.0" -> 1
            elif expected_type == float:
                return float(value)
            elif expected_type == str:
                return str(value)
            else:
                # For other types, try direct conversion
                return expected_type(value)
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"Cannot convert value '{value}' to expected type {expected_type.__name__} "
                f"for config parameter '{config_key}': {e}"
            ) 