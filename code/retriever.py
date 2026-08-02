import pandas as pd
from pathlib import Path

DATASET_DIR = Path(__file__).parent.parent / "dataset"

messages_df = pd.read_csv(DATASET_DIR/ "messages.csv")
users_df = pd.read_csv(DATASET_DIR/"users.csv")
groups_df = pd.read_csv(DATASET_DIR/"groups.csv")
group_members_df = pd.read_csv(DATASET_DIR/"group_members.csv")
business_df = pd.read_csv(DATASET_DIR/"business_accounts.csv")
history_df = pd.read_csv(DATASET_DIR/ "message_history.csv")
events_df = pd.read_csv(DATASET_DIR/ "message_events.csv")

def get_user(user_id: str):
    
    user = users_df[
        users_df["user_id"] == user_id
    ]
    
    if user.empty:
        return None
    
    return user.iloc[0].to_dict()