from typing import TypedDict, Optional

class Message(TypedDict):
    message_id: str
    
    user_id: str
    
    conversation_type: str
    
    group_id: Optional[str]
    
    business_id: Optional[str]
    
    sender_user_id: Optional[str]
    
    created_at: str
    
    message_text: str
    
    media_type: Optional[str]
    
    media_id: Optional[str]
    
    forwarded_count: int
    
class Prediction(TypedDict):
    action: str
    
    message_type: str
    
    reason: str
    
    confidence: float
    
    evidence_message_ids: str

class RouterState(TypedDict):
    
    # current incoming text
    message: Message
    
    # user info
    user: dict
    
    # grp info
    group: Optional[dict]
    
    # business info
    business: Optional[dict]
    
    # prev related msg
    history: list
    
    # image analysis res
    image_context: Optional[str]
    
    # voice transcription
    voice_context: Optional[str]
    
    # final prediction
    prediction: Optional[Prediction]