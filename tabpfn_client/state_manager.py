#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0
"""State management for resuming interrupted registration flows."""

import json
from pathlib import Path
from typing import Optional


class RegistrationState:
    """Manages state persistence for interrupted registration flows."""

    def __init__(self):
        self.state_file = Path.home() / ".tabpfn" / "registration_state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    def save(self, data: dict) -> None:
        """Save registration state to disk."""
        try:
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            # Silently fail if we can't save state
            pass

    def load(self) -> Optional[dict]:
        """Load registration state from disk."""
        try:
            if self.state_file.exists():
                with open(self.state_file, "r") as f:
                    return json.load(f)
        except Exception:
            pass
        return None

    def clear(self) -> None:
        """Clear saved state."""
        try:
            if self.state_file.exists():
                self.state_file.unlink()
        except Exception:
            pass


def check_internet_connection() -> bool:
    """Check if internet connection is available."""
    import socket

    try:
        # Try to connect to a reliable host
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except (socket.timeout, socket.error):
        return False
