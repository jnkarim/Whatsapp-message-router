import base64
import json
import re
from pathlib import Path
from typing import Any

import whisper
from langchain_core.messages import HumanMessage, SystemMessage

from retriever import retrieve_context
from state import RouterState
from llm import llm
from prompts import SYSTEM_PROMPT
from utils import parse_llm_json, format_context_for_prompt, extract_evidence_ids

# Load Whisper once at startup (not per message)
# tiny model = fast, good enough for short voice notes
whisper_model = whisper.load_model("tiny")




def retrieve_node(state: RouterState) -> RouterState:
    """
    First node in the pipeline.
    Pulls all CSV context for this message using retrieve_context().
    Stores it in state so all later nodes can use it.
    """
    message = state["message"]
    context = retrieve_context(message)
    return {**state, "context": context}



def image_node(state: RouterState) -> RouterState:
    """
    Runs only if message has an image (checked by router in graph.py).
    Reads the image file, converts to base64, sends to Groq vision model.
    Adds the image description to context as 'media_content'.
    """
    context = state["context"]
    media_path = context.get("media_path")

    if not media_path or not Path(media_path).exists():
        return {**state, "context": {**context, "media_content": "Image file not found."}}

    # Read and encode image
    with open(media_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    # Determine media type
    suffix = Path(media_path).suffix.lower()
    media_type_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
    mime_type = media_type_map.get(suffix, "image/jpeg")

    # Ask LLM to describe the image for routing purposes
    try:
        response = llm.invoke([
            HumanMessage(content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_data}"
                    }
                },
                {
                    "type": "text",
                    "text": (
                        "Describe this image in 2-3 sentences focusing on: "
                        "what it is (poster, screenshot, photo), what it says or shows, "
                        "and any red flags like fake urgency, OTP requests, suspicious links, "
                        "or promotional content."
                    )
                }
            ])
        ])
        description = response.content
    except Exception as e:
        description = f"Image analysis failed: {str(e)}"

    return {**state, "context": {**context, "media_content": description}}



def voice_node(state: RouterState) -> RouterState:
    """
    Runs only if message has a voice note (checked by router in graph.py).
    Uses Whisper to transcribe the audio file.
    Adds transcript to context as 'media_content'.
    """
    context = state["context"]
    media_path = context.get("media_path")

    if not media_path or not Path(media_path).exists():
        return {**state, "context": {**context, "media_content": "Voice note file not found."}}

    try:
        result = whisper_model.transcribe(media_path)
        transcript = result.get("text", "").strip()
        media_content = f"Voice note transcript: {transcript}"
    except Exception as e:
        media_content = f"Transcription failed: {str(e)}"

    return {**state, "context": {**context, "media_content": media_content}}




def route_node(state: RouterState) -> RouterState:
    """
    The main decision node.
    Takes the message + all context → sends to Groq LLM → gets routing decision.

    LLM is asked to return a JSON with:
    - action: notify / digest / mute
    - message_type: personal / urgent / scam / etc.
    - reason: short explanation
    - confidence: 0 to 1
    - evidence_message_ids: semicolon-separated IDs or 'none'
    """
    message = state["message"]
    context = state["context"]

    # Format context into readable text for the prompt
    context_text = format_context_for_prompt(context)

    # Build the full user prompt
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

    return {**state, "prediction": prediction}




def output_node(state: RouterState) -> RouterState:
    """
    Final node. Takes the prediction dict and formats it
    into the exact output row structure needed for output.csv.
    """
    message = state["message"]
    prediction = state.get("prediction", {})

    result = {
        "message_id":          message.get("message_id"),
        "action":              prediction.get("action", "digest"),
        "message_type":        prediction.get("message_type", "unknown"),
        "reason":              prediction.get("reason", "No reason provided"),
        "confidence":          prediction.get("confidence", 0.3),
        "evidence_message_ids": prediction.get("evidence_message_ids", "none")
    }

    return {**state, "result": result}