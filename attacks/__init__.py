"""
Multi-Shield Attack Module

This module contains modified versions of AutoAttack components adapted for
evaluating the Multi-Shield defense mechanism. The modifications add support
for rejection-aware adversarial attacks.
"""

from .modified_autoattack import AutoAttack

__all__ = ["AutoAttack"]
