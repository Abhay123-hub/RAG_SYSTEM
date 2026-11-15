import re

from config.settings import get_settings


class SafetyChecker:
    """Apply bounded input checks and flag common instruction-override attempts."""

    _injection = re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions|reveal\s+(the\s+)?system prompt", re.IGNORECASE)

    def check(self, text: str) -> tuple[bool, str]:
        if len(text) > get_settings().max_input_length:
            return False, "Message is too long. Please shorten it and try again."
        if self._injection.search(text):
            return False, "I can't help with requests to override system instructions."
        return True, ""