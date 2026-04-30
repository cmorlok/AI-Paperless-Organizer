"""Type stub for dishka FromDishka to satisfy pyright.

Dishka's FromDishka[X] is used with `= None` as a default in FastAPI routes.
The @inject decorator replaces None with the actual service at runtime.
This stub tells pyright that FromDishka[X] accepts None as a default.
"""

from typing import TypeVar, Generic

T = TypeVar("T")

class FromDishka(Generic[T]):
    """Marker type for Dishka dependency injection in FastAPI routes.

    Usage:
        @router.get("/")
        @inject
        async def handler(service: FromDishka[MyService] = None):
            ...

    The @inject decorator replaces None with the actual service at runtime.
    """
    def __class_getitem__(cls, item: type) -> type: ...
    # Allow None as default
    @classmethod
    def __or__(cls, other: type) -> type: ...
