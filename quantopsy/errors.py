"""Finite caller-visible failures; input bytes are never executed."""

from typing import NoReturn


class QuantopsyError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def fail(code: str, message: str) -> NoReturn:
    raise QuantopsyError(code, message)
