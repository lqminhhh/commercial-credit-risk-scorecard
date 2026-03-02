"""
Formatting and small helper utilities for the underwriting app.
"""

from __future__ import annotations


def format_currency(value: float) -> str:
    """
    Format numeric value as USD currency string.
    """
    return f"${value:,.2f}"


def format_pct(value: float, decimals: int = 1) -> str:
    """
    Format decimal as percentage string.
    Example: 0.182 -> 18.2%
    """
    return f"{value * 100:.{decimals}f}%"


def format_ratio(value: float, decimals: int = 2) -> str:
    """
    Format a ratio value.
    """
    return f"{value:.{decimals}f}x"


def title_case_decision(decision: str) -> str:
    """
    Normalize decision text for display.
    """
    return str(decision).title()