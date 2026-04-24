"""Base state class and CancelMixin for service state management.

All service state classes inherit from BaseState, which provides:
- Common fields (running, enabled)
- reset() that clears fields in-place (preserves object reference — critical for DI singletons)
- model_dump() override that excludes private/internal fields

CancelMixin adds simple flag-based cancellation (NOT asyncio.Task cancellation integration).

Design decisions (from CONTEXT.md):
- D-01: All state classes are Pydantic BaseModel subclasses
- D-02: BaseState lives in app/services/base_state.py
- D-03: asyncio.Lock is per-subclass, NOT in BaseState
- D-04: All state classes are Dishka-managed APP-scoped singletons
- D-05: reset() clears fields in-place (preserves object reference)
- D-09: Method-based cancellation (request_cancel, is_cancelled, clear_cancel)
- D-13: BaseState overrides model_dump() to exclude private/internal fields
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar, Set

from pydantic import BaseModel, ConfigDict, PrivateAttr


class BaseState(BaseModel):
    """Base class for all service state objects.

    Provides common fields (running, enabled), reset-in-place semantics,
    and model_dump() that excludes private/internal fields automatically.

    Subclasses should override _reset_fields() to clear their own fields
    when reset() is called.

    IMPORTANT: reset() must only be called when the background loop is
    stopped or in a quiet period — not while updates are in-flight.
    Calling reset() while a background task is actively mutating state
    fields can lead to race conditions.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    running: bool = False
    enabled: bool = False

    # Shared asyncio.Lock for subclass synchronization.
    # This is a ClassVar: all instances of the same class reference
    # the same lock object. Per D-03, subclasses that need per-instance
    # locks add their own. This provides a default synchronization
    # point for reset() + background updates when subclasses use it.
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    def reset(self) -> None:
        """Reset state to initial values in-place.

        Sets running=False and calls _reset_fields() for subclass cleanup.

        IMPORTANT: This method must only be called when the background
        loop is stopped or in a quiet period — not while updates are
        in-flight.
        """
        self.running = False
        self._reset_fields()

    def _reset_fields(self) -> None:
        """Hook for subclasses to clear their own fields on reset().

        Override this in subclasses to reset service-specific fields.
        Called by reset() after setting running=False.
        """
        pass

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Override to exclude private/internal fields from serialization.

        Automatically excludes any field with exclude=True in its FieldInfo.
        Private attributes (PrivateAttr) are already excluded by Pydantic.
        ClassVar attributes are not model fields and are never included.

        This prevents internal implementation details from leaking to
        API consumers (e.g., locks, cancellation flags).
        """
        # Collect field names to exclude based on FieldInfo.exclude
        auto_exclude: Set[str] = set()

        for field_name, field_info in self.__class__.model_fields.items():
            if getattr(field_info, "exclude", None) is True:
                auto_exclude.add(field_name)

        # Merge with any caller-provided exclude set
        caller_exclude = kwargs.pop("exclude", None) or set()
        merged_exclude = auto_exclude | set(caller_exclude)

        return super().model_dump(exclude=merged_exclude, **kwargs)

    def is_running(self) -> bool:
        """Check if the service background loop is running."""
        return self.running

    def is_enabled(self) -> bool:
        """Check if the service is enabled."""
        return self.enabled


class CancelMixin(BaseModel):
    """Mixin that adds simple flag-based cancellation to a state class.

    Provides request_cancel(), is_cancelled(), and clear_cancel() methods
    for cooperative cancellation of background tasks.

    NOTE: This is simple flag-checking only. It does NOT integrate with
    asyncio.Task.cancel(). Callers are responsible for periodically
    checking is_cancelled() and responding appropriately (e.g., breaking
    out of loops, skipping remaining work).
    """

    _cancelled: bool = PrivateAttr(default=False)

    def request_cancel(self) -> None:
        """Request cancellation of the current operation.

        Sets the internal cancellation flag. Background tasks should
        periodically check is_cancelled() and stop gracefully.
        """
        self._cancelled = True

    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested.

        This is simple flag-checking only — it does NOT integrate with
        asyncio.Task.cancel(). Callers are responsible for checking this
        flag and responding appropriately.
        """
        return self._cancelled

    def clear_cancel(self) -> None:
        """Clear the cancellation flag.

        Should be called before starting a new operation to ensure
        a stale cancellation request doesn't prevent work.
        """
        self._cancelled = False
