import pandas as pd
from pathlib import Path
from typing import Optional


DATASET_DIR = Path(__file__).parent.parent / "dataset"


users_df                 = pd.read_csv(DATASET_DIR / "users.csv")
groups_df                = pd.read_csv(DATASET_DIR / "groups.csv")
group_members_df         = pd.read_csv(DATASET_DIR / "group_members.csv")
business_accounts_df     = pd.read_csv(DATASET_DIR / "business_accounts.csv")
user_business_history_df = pd.read_csv(DATASET_DIR / "user_business_history.csv")
message_history_df       = pd.read_csv(DATASET_DIR / "message_history.csv")
message_events_df        = pd.read_csv(DATASET_DIR / "message_events.csv")
images_df                = pd.read_csv(DATASET_DIR / "images.csv")
voice_notes_df           = pd.read_csv(DATASET_DIR / "voice_notes.csv")



def get_user(user_id: str) -> dict:
    """
    Returns the user's notification behavior profile.
    Tells LLM: is this user active? do they report spam?
    """
    row = users_df[users_df["user_id"] == user_id]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def get_group(group_id: Optional[str]) -> dict:
    """
    Returns group metadata: type, member count, admin count.
    Society group with 184 members is very different from a 5-person family group.
    """
    if not group_id or pd.isna(group_id):
        return {}
    row = groups_df[groups_df["group_id"] == group_id]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def get_group_member(user_id: str, group_id: Optional[str]) -> dict:
    """
    Returns how THIS user relates to THIS group.
    Key signals: is user admin? did they mute this group?
    how many messages did they dismiss vs read?
    """
    if not group_id or pd.isna(group_id):
        return {}
    row = group_members_df[
        (group_members_df["user_id"] == user_id) &
        (group_members_df["group_id"] == group_id)
    ]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def get_business(business_id: Optional[str]) -> dict:
    """
    Returns business sender info: verified, domain match, reports.
    Verified + matching domain = trustworthy.
    Unverified + mismatched domain + many reports = likely scam.
    """
    if not business_id or pd.isna(business_id):
        return {}
    row = business_accounts_df[business_accounts_df["business_id"] == business_id]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def get_user_business_history(user_id: str, business_id: Optional[str]) -> dict:
    """
    Returns whether this user has a relationship with this business.
    Recent orders / opt-ins → message is expected → notify or digest.
    Opted out → mute regardless of message content.
    """
    if not business_id or pd.isna(business_id):
        return {}
    row = user_business_history_df[
        (user_business_history_df["user_id"] == user_id) &
        (user_business_history_df["business_id"] == business_id)
    ]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def get_message_history(user_id: str, limit: int = 10) -> list[dict]:
    """
    Returns the last N messages this user received historically.
    Used to detect repeated patterns and find evidence_message_ids.
    """
    rows = message_history_df[message_history_df["user_id"] == user_id]
    rows = rows.sort_values("created_at", ascending=False).head(limit)
    return rows.to_dict(orient="records")


def get_message_events(user_id: str, limit: int = 10) -> list[dict]:
    """
    Returns how the user reacted to past messages.
    Dismissed 8 out of 10 similar → mute this one.
    Replied quickly → engaged → lean toward notify.
    """
    user_history_ids = message_history_df[
        message_history_df["user_id"] == user_id
    ]["message_id"].tolist()

    rows = message_events_df[
        (message_events_df["user_id"] == user_id) &
        (message_events_df["message_id"].isin(user_history_ids))
    ]
    rows = rows.head(limit)
    return rows.to_dict(orient="records")


def get_media_path(media_type: Optional[str], media_id: Optional[str]) -> Optional[str]:
    """
    Returns the absolute file path for an image or voice note.
    Returns None if no media is attached.
    This path is used by the media analysis nodes later.
    """
    if not media_type or not media_id or pd.isna(media_id):
        return None

    if media_type == "image":
        row = images_df[images_df["image_id"] == media_id]
        if row.empty:
            return None
        return str(DATASET_DIR / row.iloc[0]["file_path"])

    if media_type == "voice":
        row = voice_notes_df[voice_notes_df["voice_note_id"] == media_id]
        if row.empty:
            return None
        return str(DATASET_DIR / row.iloc[0]["file_path"])

    return None



def retrieve_context(message: dict) -> dict:
    """
    Single function that pulls ALL context for one message.
    nodes.py calls this once and gets everything back in one dict.
    """
    user_id     = message.get("user_id")
    group_id    = message.get("group_id")
    business_id = message.get("business_id")
    media_type  = message.get("media_type")
    media_id    = message.get("media_id")

    return {
        "user":                  get_user(user_id),
        "group":                 get_group(group_id),
        "group_member":          get_group_member(user_id, group_id),
        "business":              get_business(business_id),
        "user_business_history": get_user_business_history(user_id, business_id),
        "message_history":       get_message_history(user_id, limit=10),
        "message_events":        get_message_events(user_id, limit=10),
        "media_path":            get_media_path(media_type, media_id),
    }