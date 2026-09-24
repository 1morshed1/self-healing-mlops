"""Notifications: get a diagnosis + remediation summary in front of a human.

Auto-remediation must ALWAYS be accompanied by a notification. "The system
quietly retrained itself and nobody knew" is how you lose trust in automation.
So every cycle ends with a message — to Slack/Discord if a webhook is set, else
to the log.

Kept deliberately tiny and dependency-light (uses ``requests``). The webhook
payload uses the common ``{"text": ...}`` shape that both Slack and Discord
incoming webhooks accept.
"""

from __future__ import annotations

import logging

import requests

from mlops.config import get_settings

logger = logging.getLogger(__name__)


def send_alert(title: str, body: str) -> bool:
    """Send an alert to the configured webhook, or log it if none is set.

    Returns True if a webhook actually received the message, False if we only
    logged it (or the webhook call failed). Never raises — a broken notification
    channel must not take down the healing loop.
    """
    settings = get_settings()
    message = f"*{title}*\n{body}"

    if not settings.alert_webhook_url:
        logger.info("ALERT (no webhook configured):\n%s", message)
        return False

    try:
        resp = requests.post(
            settings.alert_webhook_url,
            json={"text": message},
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except requests.RequestException as exc:
        # Log and carry on — notification failure is not fatal.
        logger.warning("Failed to post alert to webhook: %s", exc)
        logger.info("ALERT (webhook failed):\n%s", message)
        return False
