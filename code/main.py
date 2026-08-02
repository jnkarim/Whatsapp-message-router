import pandas as pd
import time
import traceback
from pathlib import Path
from graph import build_graph

DATASET_DIR = Path(__file__).parent.parent / "dataset"
INPUT_FILE  = DATASET_DIR / "messages.csv"
OUTPUT_FILE = DATASET_DIR / "output.csv"


def run():
    messages_df = pd.read_csv(INPUT_FILE)
    total = len(messages_df)
    print(f"Loaded {total} messages. Starting pipeline...\n")

    graph = build_graph()
    results = []

    for i, row in messages_df.iterrows():
        message = row.to_dict()
        msg_id  = message.get("message_id")

        print(f"[{i+1}/{total}] Processing {msg_id}...", end=" ", flush=True)

        initial_state = {
            "message":       message,
            "user":          {},
            "group":         {},
            "business":      {},
            "history":       {},
            "image_context": None,
            "voice_context": None,
            "prediction":    None,
        }

        # retry up to 3 times on rate limit
        success = False
        for attempt in range(3):
            try:
                final_state = graph.invoke(initial_state)
                result = final_state.get("history", {}).get("result", {})

                if not isinstance(result, dict):
                    raise ValueError(f"result is not a dict: {type(result)}")

                # check if it actually got a real LLM response
                if result.get("confidence", 0) == 0.2 and result.get("message_type") == "unknown":
                    raise ValueError("Got fallback response — likely rate limited")

                results.append({
                    "message_id":           result.get("message_id", msg_id),
                    "action":               result.get("action", "digest"),
                    "message_type":         result.get("message_type", "unknown"),
                    "reason":               result.get("reason", "No reason"),
                    "confidence":           result.get("confidence", 0.3),
                    "evidence_message_ids": result.get("evidence_message_ids", "none")
                })

                print(f"→ {result.get('action')} | {result.get('message_type')} | confidence: {result.get('confidence')}")
                success = True
                break

            except Exception as e:
                err = str(e)
                if "429" in err or "rate" in err.lower() or "fallback" in err.lower():
                    wait = 10 * (attempt + 1)  # 10s, 20s, 30s
                    print(f"Rate limited. Waiting {wait}s (attempt {attempt+1}/3)...", end=" ", flush=True)
                    time.sleep(wait)
                else:
                    print(f"ERROR: {err[:80]}")
                    break

        if not success:
            print(f"FAILED after 3 attempts — using fallback")
            results.append({
                "message_id":           msg_id,
                "action":               "digest",
                "message_type":         "unknown",
                "reason":               "Rate limit fallback",
                "confidence":           0.2,
                "evidence_message_ids": "none"
            })

        # delay between messages
        time.sleep(2)

    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDone. Output written to {OUTPUT_FILE}")
    print(f"Total rows: {len(output_df)}")


if __name__ == "__main__":
    run()