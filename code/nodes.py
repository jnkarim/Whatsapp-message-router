import base64
from pathlib import Path
import os

import whisper
from langchain_core.messages import HumanMessage, SystemMessage

from retriever import retrieve_context
from state import RouterState
from llm import llm
from prompts import SYSTEM_PROMPT
from utils import parse_llm_json, format_context_for_prompt

# Load Whisper once at startup
whisper_model = whisper.load_model("tiny")


def retrieve_node(state: RouterState) -> dict:
    message = state["message"]
    context = retrieve_context(message)
    return {
        "user":     context.get("user", {}),
        "group":    context.get("group", {}),
        "business": context.get("business", {}),
        "history":  {
            "group_member":          context.get("group_member", {}),
            "user_business_history": context.get("user_business_history", {}),
            "message_history":       context.get("message_history", []),
            "message_events":        context.get("message_events", []),
            "media_path":            context.get("media_path"),
        }
    }


def image_node(state: RouterState) -> dict:
    media_path = state["history"].get("media_path")
    if not media_path or not Path(media_path).exists():
        return {"image_context": "Image file not found."}

    filename = Path(media_path).name
    return {"image_context": f"Image attached: {filename}. Use message text and context for routing."}


def voice_node(state: RouterState) -> dict:
    from groq import Groq

    media_path = state["history"].get("media_path")

    if not media_path or not Path(media_path).exists():
        return {"voice_context": "Voice note file not found."}

    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        with open(media_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=f,
                response_format="text"
            )
        return {"voice_context": f"Voice note transcript: {transcription}"}
    except Exception as e:
        return {"voice_context": f"Transcription failed: {str(e)[:80]}"}


def route_node(state: RouterState) -> dict:
    message = state["message"]
    history = state.get("history", {})

    context = {
        "user":                  state.get("user", {}),
        "group":                 state.get("group", {}),
        "group_member":          history.get("group_member", {}),
        "business":              state.get("business", {}),
        "user_business_history": history.get("user_business_history", {}),
        "message_history":       history.get("message_history", []),
        "message_events":        history.get("message_events", []),
        "media_content":         state.get("image_context") or state.get("voice_context"),
    }

    context_text = format_context_for_prompt(context)

    user_prompt = f"""
Incoming message details:
- message_id: {message.get('message_id')}
- conversation_type: {message.get('conversation_type')}
- sender_user_id: {message.get('sender_user_id', 'N/A')}
- message_text: {message.get('message_text', '[no text]')}
- media_type: {message.get('media_type', 'none')}
- forwarded_count: {message.get('forwarded_count', 0)}
- created_at: {message.get('created_at')}

Context:
{context_text}

Based on all of the above, decide the routing for this message.
Respond ONLY with a valid JSON object in this exact format:
{{
  "action": "notify" | "digest" | "mute",
  "message_type": "personal" | "urgent" | "event" | "payment" | "business_update" | "promotion" | "greeting" | "forward" | "spam" | "scam" | "unknown",
  "reason": "short explanation under 20 words",
  "confidence": 0.0 to 1.0,
  "evidence_message_ids": "msg_id1;msg_id2" or "none"
}}
"""

    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ])
        prediction = parse_llm_json(response.content)
    except Exception as e:
        prediction = {
            "action": "digest",
            "message_type": "unknown",
            "reason": f"LLM error: {str(e)[:50]}",
            "confidence": 0.2,
            "evidence_message_ids": "none"
        }

    return {"prediction": prediction}


def output_node(state: RouterState) -> dict:
    message    = state["message"]
    prediction = state.get("prediction")

    if not isinstance(prediction, dict):
        prediction = {
            "action": "digest",
            "message_type": "unknown",
            "reason": "Invalid prediction format",
            "confidence": 0.3,
            "evidence_message_ids": "none"
        }

    result = {
        "message_id":           message.get("message_id"),
        "action":               prediction.get("action", "digest"),
        "message_type":         prediction.get("message_type", "unknown"),
        "reason":               prediction.get("reason", "No reason provided"),
        "confidence":           prediction.get("confidence", 0.3),
        "evidence_message_ids": prediction.get("evidence_message_ids", "none")
    }

    return {"history": {**state.get("history", {}), "result": result}}