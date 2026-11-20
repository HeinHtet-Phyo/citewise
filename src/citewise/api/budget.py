"""Daily spend cap for the public demo (in memory; resets at UTC midnight and on restart)."""

from __future__ import annotations

from datetime import UTC, datetime


class DailyBudget:
    """Track generation spend per UTC day and refuse new work once the cap is reached."""

    def __init__(self, cap_gbp: float) -> None:
        self.cap_gbp = cap_gbp
        self._day = self._today()
        self._spent = 0.0

    @staticmethod
    def _today() -> str:
        return datetime.now(UTC).date().isoformat()

    def _roll(self) -> None:
        if self._today() != self._day:
            self._day, self._spent = self._today(), 0.0

    def exhausted(self) -> bool:
        """True when today's spend has reached the cap."""
        self._roll()
        return self._spent >= self.cap_gbp

    def add(self, cost_gbp: float) -> None:
        """Record the cost of one answer."""
        self._roll()
        self._spent += cost_gbp
