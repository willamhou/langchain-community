"""Callback Handler that creates cryptographic receipts with Signet.

Signs every tool call with Ed25519 and appends to a hash-chained audit log.
Receipts are tamper-evident and offline-verifiable — no external service required.

Usage:
    .. code-block:: python

        from langchain_community.callbacks.signet_callback import (
            SignetCallbackHandler,
        )

        handler = SignetCallbackHandler(key_name="my-agent")
        chain.invoke(input, config={"callbacks": [handler]})

        # Verify audit trail
        # signet audit --since 1h
        # signet verify --chain

Requirements:
    pip install signet-auth
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)


class SignetCallbackHandler(BaseCallbackHandler):
    """Callback Handler that signs tool calls with Ed25519 via Signet.

    Each tool invocation produces a signed receipt appended to a
    hash-chained audit log at ``~/.signet/audit/``. Receipts can be
    verified offline with ``signet verify --chain``.

    Parameters:
        key_name: Name of the Ed25519 signing identity.
            Auto-created on first use if it doesn't exist.
        owner: Optional owner label for the identity.
        audit: Whether to write to the audit log. Default True.

    Example:
        .. code-block:: python

            from langchain_community.callbacks.signet_callback import (
                SignetCallbackHandler,
            )

            handler = SignetCallbackHandler(key_name="langchain-bot")
            agent_executor.invoke(
                {"input": "search for AI news"},
                config={"callbacks": [handler]},
            )
    """

    def __init__(
        self,
        key_name: str = "langchain-agent",
        owner: str = "",
        audit: bool = True,
    ) -> None:
        super().__init__()
        try:
            import signet_auth
        except ImportError:
            raise ImportError(
                "signet-auth is required for SignetCallbackHandler. "
                "Install it with: pip install signet-auth"
            )

        try:
            self._agent = signet_auth.SigningAgent(key_name)
        except signet_auth.KeyNotFoundError:
            self._agent = signet_auth.SigningAgent.create(key_name, owner=owner)

        self._audit = audit
        self._receipts: List[Any] = []

    @property
    def receipts(self) -> List[Any]:
        """Return all receipts generated during this session."""
        return list(self._receipts)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Sign the tool call when it starts."""
        tool_name = serialized.get("name", serialized.get("id", ["unknown"])[-1])

        params: Dict[str, Any] = {}
        if inputs is not None:
            params = inputs
        elif input_str:
            try:
                params = json.loads(input_str)
            except (json.JSONDecodeError, TypeError):
                params = {"input": input_str}

        try:
            receipt = self._agent.sign(
                tool_name,
                params=params,
                audit=self._audit,
            )
            self._receipts.append(receipt)
            logger.debug("Signet: signed %s (%s)", tool_name, receipt.id)
        except Exception:
            logger.warning("Signet: failed to sign %s", tool_name, exc_info=True)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Log tool completion (no additional signing needed)."""
        pass

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Sign a failure receipt when a tool errors."""
        try:
            receipt = self._agent.sign(
                "_tool_error",
                params={"error": str(error)[:500]},
                audit=self._audit,
            )
            self._receipts.append(receipt)
            logger.debug("Signet: signed tool_error (%s)", receipt.id)
        except Exception:
            logger.warning("Signet: failed to sign tool_error", exc_info=True)
