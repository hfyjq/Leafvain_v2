"""
Security kernel: path whitelist, command blacklist, audit logging.

Executed BEFORE any file I/O or command execution. Any violation
raises SecurityError immediately — no recovery path that bypasses
the guard.
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path


class SecurityError(Exception):
    """Raised when a security policy is violated."""
    pass


class SecurityGuard:
    """
    Validates file paths against allowed directories and commands
    against a blacklist. Logs all actions to an audit log.
    """

    def __init__(
        self,
        allowed_dirs: list[str],
        command_blacklist: list[str],
        audit_log_path: str,
    ) -> None:
        # Resolve allowed directories to absolute paths upfront
        self._allowed_dirs = [Path(d).resolve() for d in allowed_dirs]
        self._command_blacklist = set(cmd.lower() for cmd in command_blacklist)
        self._audit_log_path = Path(audit_log_path)

        # Ensure audit log directory exists
        self._audit_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Set up dedicated audit logger
        self._logger = logging.getLogger("audit")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False  # don't leak to root logger

        if not self._logger.handlers:
            handler = logging.FileHandler(
                self._audit_log_path, encoding="utf-8", delay=True
            )
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            self._logger.addHandler(handler)

    def validate_path(self, file_path: str) -> Path:
        """
        Resolve *file_path*, verify it lies within an allowed directory.

        Returns the resolved Path on success.  Raises SecurityError if
        the resolved path escapes all allowed directories.
        """
        resolved = Path(file_path).resolve()

        for allowed in self._allowed_dirs:
            try:
                resolved.relative_to(allowed)
            except ValueError:
                continue  # not under this allowed dir
            else:
                self._audit("PATH_ALLOW", str(resolved))
                return resolved

        # Escaped all allowed directories
        self._audit("PATH_DENY", str(resolved))
        raise SecurityError(
            f"Access denied: '{file_path}' (resolved: '{resolved}') "
            f"is outside allowed directories: {self._allowed_dirs}"
        )

    def validate_command(self, command: str) -> bool:
        """
        Return True if *command* does not contain any blacklisted substring
        (case-insensitive).  Raises SecurityError otherwise.
        """
        lower_cmd = command.lower()
        for banned in self._command_blacklist:
            if banned in lower_cmd:
                self._audit("CMD_DENY", command)
                raise SecurityError(
                    f"Command blocked: '{command}' matches blacklisted pattern '{banned}'"
                )
        self._audit("CMD_ALLOW", command)
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _audit(self, action: str, target: str) -> None:
        """Write a structured audit entry."""
        self._logger.info("%s | %s", action, target)
