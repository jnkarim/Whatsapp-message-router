import json
import re
from typing import Optional


def parse_llm_json(text: str) -> dict:
    """
    LLM sometimes wraps JSON in ```json ... ``` blocks.
    Sometimes it adds extra text before/after.
    This function handles all those cases safely.

    If parsing fails completely, returns a safe fallback dict
    so the pipeline never crashes mid-run.
    """
    if not text:
        return _fallback()

    # Strip markdown code fences if present
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try finding JSON object inside the text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Nothing worked — return safe fallback
    return _fallback()


def _fallback() -> dict:
    """
    Safe default when LLM output cannot be parsed.
    Pipeline continues instead of crashing.
    """
    return {
        "action": "digest",
        "message_type": "unknown",
        "reason": "Could not parse LLM response",
        "confidence": 0.3,
        "evidence_message_ids": "none"
    }


def format_context_for_prompt(context: dict) -> str:
    """
    The LLM cannot read a Python dict directly.
    This converts the context into clean labeled sections
    that the LLM can reason over easily.

    Example output:
        [USER]
        user_id: u_002
        do_not_disturb_window: 23:00-08:00
        messages_opened_30d: 52
        ...

        [BUSINESS]
        display_name: HDFC Bank
        verified: True
        ...
    """
    sections = []

    # User profile
    if context.get("user"):
        u = context["user"]
        sections.append(
            f"[USER]\n"
            f"user_id: {u.get('user_id', 'N/A')}\n"
            f"do_not_disturb_window: {u.get('do_not_disturb_window', 'N/A')}\n"
            f"messages_opened_30d: {u.get('messages_opened_30d', 0)}\n"
            f"messages_replied_30d: {u.get('messages_replied_30d', 0)}\n"
            f"notifications_dismissed_30d: {u.get('notifications_dismissed_30d', 0)}\n"
            f"messages_reported_30d: {u.get('messages_reported_30d', 0)}"
        )

    # Group info
    if context.get("group"):
        g = context["group"]
        sections.append(
            f"[GROUP]\n"
            f"group_type: {g.get('group_type', 'N/A')}\n"
            f"member_count: {g.get('member_count', 'N/A')}\n"
            f"admin_count: {g.get('admin_count', 'N/A')}\n"
            f"messages_sent_30d: {g.get('messages_sent_30d', 'N/A')}"
        )

    # User's relationship to the group
    if context.get("group_member"):
        gm = context["group_member"]
        sections.append(
            f"[USER-GROUP RELATIONSHIP]\n"
            f"role: {gm.get('role', 'N/A')}\n"
            f"group_muted_by_user: {gm.get('group_muted_by_user', 'N/A')}\n"
            f"notifications_dismissed_30d: {gm.get('notifications_dismissed_30d', 0)}\n"
            f"messages_read_30d: {gm.get('messages_read_30d', 0)}\n"
            f"replies_sent_30d: {gm.get('replies_sent_30d', 0)}"
        )

    # Business sender info
    if context.get("business"):
        b = context["business"]
        sections.append(
            f"[BUSINESS]\n"
            f"display_name: {b.get('display_name', 'N/A')}\n"
            f"category: {b.get('category', 'N/A')}\n"
            f"verified: {bool(b.get('verified', 0))}\n"
            f"official_domain: {b.get('official_domain', 'N/A')}\n"
            f"domain_used_by_sender: {b.get('domain_used_by_sender', 'N/A')}\n"
            f"domain_match: {b.get('official_domain') == b.get('domain_used_by_sender')}\n"
            f"account_age_days: {b.get('account_age_days', 'N/A')}\n"
            f"user_reports_30d: {b.get('user_reports_30d', 0)}"
        )

    # User's history with this business
    if context.get("user_business_history"):
        ubh = context["user_business_history"]
        sections.append(
            f"[USER-BUSINESS RELATIONSHIP]\n"
            f"why_user_knows_account: {ubh.get('why_user_knows_account', 'N/A')}\n"
            f"allows_promotions: {bool(ubh.get('allows_promotions', 0))}\n"
            f"promotions_opted_out_at: {ubh.get('promotions_opted_out_at', 'N/A')}\n"
            f"messages_opened_30d: {ubh.get('messages_opened_30d', 0)}\n"
            f"messages_dismissed_30d: {ubh.get('messages_dismissed_30d', 0)}\n"
            f"last_activity_at: {ubh.get('last_activity_at', 'N/A')}"
        )

    # Recent message history (for evidence)
    if context.get("message_history"):
        history_lines = []
        for m in context["message_history"][:5]:  # Top 5 only to keep prompt short
            history_lines.append(
                f"  - [{m.get('message_id')}] {m.get('conversation_type')} | "
                f"{m.get('message_text', '')[:80]} | forwarded: {m.get('forwarded_count', 0)}"
            )
        sections.append("[MESSAGE HISTORY (recent)]\n" + "\n".join(history_lines))

    # User reactions to past messages
    if context.get("message_events"):
        event_lines = []
        for e in context["message_events"][:5]:
            event_lines.append(
                f"  - [{e.get('message_id')}] "
                f"opened:{e.get('message_opened')} "
                f"replied:{e.get('message_replied')} "
                f"dismissed:{e.get('notification_dismissed')} "
                f"reported:{e.get('message_reported')}"
            )
        sections.append("[USER REACTIONS TO PAST MESSAGES]\n" + "\n".join(event_lines))

    # Media transcription or description (added later by media nodes)
    if context.get("media_content"):
        sections.append(f"[MEDIA CONTENT]\n{context['media_content']}")

    return "\n\n".join(sections)



def extract_evidence_ids(message_history: list[dict], max_ids: int = 3) -> str:
    """
    Takes the message history list and returns
    a semicolon-separated string of message IDs.
    These become the evidence_message_ids in output.csv.

    Returns 'none' if no history exists.
    """
    if not message_history:
        return "none"

    ids = [m["message_id"] for m in message_history[:max_ids] if "message_id" in m]
    if not ids:
        return "none"

    return ";".join(ids)