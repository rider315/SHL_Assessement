"""
Core agent logic: intent detection, LLM calls, response formatting.
Handles 4 conversational modes: CLARIFY, RECOMMEND, REFINE, COMPARE.
Includes pre-LLM guardrails and post-LLM validation.
"""
import json
import os
import re
from pathlib import Path

from google import genai
from dotenv import load_dotenv

from retrieval import get_retriever

load_dotenv()

# Configure Gemini client
GEMINI_CLIENT = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))

# Load prompts
PROMPTS_DIR = Path(__file__).parent / "prompts"
SYSTEM_PROMPT = (PROMPTS_DIR / "system_prompt.txt").read_text(encoding="utf-8")

# Gemini model - Flash is fast and free-tier friendly
MODEL_NAME = "gemini-2.5-flash"

# Prompt injection patterns
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?prior\s+instructions",
    r"you\s+are\s+now\s+(?!looking|hiring|searching|trying)",
    r"act\s+as\s+(?!a\s+recruiter|a\s+hiring|an\s+hr)",
    r"pretend\s+to\s+be",
    r"\bdan\b",
    r"jailbreak",
    r"reveal\s+(your\s+)?(system\s+)?prompt",
    r"show\s+(me\s+)?(your\s+)?instructions",
    r"what\s+are\s+your\s+(system\s+)?instructions",
    r"forget\s+(your\s+)?rules",
    r"override\s+(your\s+)?(rules|instructions|prompt)",
    r"bypass\s+(your\s+)?(rules|restrictions|filters)",
    r"do\s+anything\s+now",
    r"system\s*:\s*you\s+are",
]

# Out-of-scope patterns
OUT_OF_SCOPE_PATTERNS = [
    r"how\s+(do|can|should)\s+i\s+write\s+a\s+job\s+description",
    r"how\s+(do|can|should)\s+i\s+(conduct|do)\s+(an?\s+)?interview",
    r"(can|is)\s+i(t)?\s+legal(ly)?",
    r"what\s+does\s+(google|amazon|microsoft|meta|apple)\s+use",
    r"(cook|recipe|weather|stock|crypto|code|program)",
    r"(competitor|alternative)\s+(to|for)\s+shl",
]

# Role/skill indicators for vague query detection
ROLE_INDICATORS = [
    r"\b(developer|engineer|manager|analyst|designer|architect|consultant|administrator|"
    r"director|lead|specialist|coordinator|supervisor|executive|intern|associate|"
    r"accountant|scientist|researcher|tester|recruiter|sales|marketing|finance|"
    r"operations|support|technician|nurse|teacher|driver|agent|clerk|cashier|"
    r"programmer|coder|devops|data|software|hardware|network|security|cloud|"
    r"java|python|javascript|\.net|c\+\+|sql|react|angular|node|aws|azure|"
    r"full.?stack|front.?end|back.?end|machine\s+learning|ai\b|ml\b)\b",
    r"\bjob\s+description\b",
    r"\bhiring\s+(for|a)\b",
    r"\b(senior|junior|mid|entry|principal|staff)\b",
    r"\b(leadership|communication|problem.solving|teamwork|analytical|cognitive|"
    r"verbal|numerical|personality|behavioral|aptitude|ability)\b",
]


def detect_prompt_injection(text: str) -> bool:
    """Check if the text contains prompt injection attempts."""
    text_lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def detect_out_of_scope(text: str) -> bool:
    """Check if the text is an out-of-scope request."""
    text_lower = text.lower()
    for pattern in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def is_vague_first_turn(text: str) -> bool:
    """Check if the first message is too vague to recommend on."""
    text_lower = text.lower()
    for pattern in ROLE_INDICATORS:
        if re.search(pattern, text_lower):
            return False
    return True


def format_catalog_context(items: list[dict]) -> str:
    """Format retrieved catalog items as JSON context for the LLM."""
    context_items = []
    for item in items:
        test_types = item.get("test_type", [])
        if isinstance(test_types, list):
            primary_type = test_types[0] if test_types else "K"
        else:
            primary_type = test_types or "K"

        context_item = {
            "name": item["name"],
            "url": item["url"],
            "test_type": primary_type,
            "all_test_types": test_types if isinstance(test_types, list) else [test_types],
            "description": item.get("description", "No description available"),
            "remote_testing": item.get("remote_testing", False),
            "adaptive_irt": item.get("adaptive_irt", False),
        }
        if item.get("job_levels"):
            context_item["job_levels"] = item["job_levels"]
        context_items.append(context_item)

    return json.dumps(context_items, indent=2)


def build_conversation_text(messages: list[dict]) -> str:
    """Format the conversation history for the LLM prompt."""
    lines = []
    for msg in messages:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def validate_and_fix_response(response: dict, retriever) -> dict:
    """Post-process and validate the LLM response to ensure schema compliance."""
    # Ensure all required fields exist
    if "reply" not in response or not isinstance(response["reply"], str):
        response["reply"] = "I can help you find the right SHL assessment. What role are you hiring for?"

    if "recommendations" not in response or response["recommendations"] is None:
        response["recommendations"] = []

    if not isinstance(response["recommendations"], list):
        response["recommendations"] = []

    if "end_of_conversation" not in response:
        response["end_of_conversation"] = False

    if not isinstance(response["end_of_conversation"], bool):
        response["end_of_conversation"] = str(response["end_of_conversation"]).lower() == "true"

    # Validate each recommendation
    valid_recs = []
    for rec in response["recommendations"]:
        if not isinstance(rec, dict):
            continue
        if "name" not in rec or "url" not in rec:
            continue

        # Validate URL against catalog
        if not retriever.validate_url(rec.get("url", "")):
            # Try to find the assessment by name and fix the URL
            found = retriever.get_assessment_by_name(rec.get("name", ""))
            if found:
                rec["url"] = found["url"]
                if not rec.get("test_type"):
                    types = found.get("test_type", [])
                    rec["test_type"] = types[0] if isinstance(types, list) and types else "K"
            else:
                continue  # Drop recommendations with invalid URLs

        # Ensure test_type is a single string
        if "test_type" not in rec or not isinstance(rec["test_type"], str):
            rec["test_type"] = "K"

        # Keep only the 3 required fields
        valid_recs.append({
            "name": rec["name"],
            "url": rec["url"],
            "test_type": rec["test_type"],
        })

    # Enforce max 10 recommendations
    response["recommendations"] = valid_recs[:10]

    return response


async def process_chat(messages: list[dict]) -> dict:
    """Process a chat request through the full agent pipeline.

    Steps:
    1. Pre-LLM guardrails (injection, out-of-scope, vague query)
    2. Retrieve relevant assessments from vector store
    3. Build prompt with catalog context
    4. Call Gemini LLM
    5. Post-process and validate response
    """
    retriever = get_retriever()

    # Get the latest user message
    user_messages = [m for m in messages if m.get("role") == "user"]
    if not user_messages:
        return {
            "reply": "Hello! I'm the SHL Assessment Recommender. Tell me about the role you're hiring for, and I'll help you find the right assessments.",
            "recommendations": [],
            "end_of_conversation": False,
        }

    latest_message = user_messages[-1]["content"]

    # --- GUARDRAIL 1: Prompt injection detection ---
    if detect_prompt_injection(latest_message):
        return {
            "reply": "I can only help with SHL assessment recommendations. I'm not able to modify my behavior or process that type of request. What role are you hiring for?",
            "recommendations": [],
            "end_of_conversation": False,
        }

    # --- GUARDRAIL 2: Out-of-scope detection ---
    if detect_out_of_scope(latest_message):
        return {
            "reply": "I specialize exclusively in SHL assessment recommendations. For that type of question, I'd suggest consulting the appropriate resource. How can I help you find the right SHL assessment?",
            "recommendations": [],
            "end_of_conversation": False,
        }

    # --- GUARDRAIL 3: Vague first turn detection ---
    is_first_turn = len(user_messages) == 1
    if is_first_turn and is_vague_first_turn(latest_message):
        return {
            "reply": "I'd love to help you find the right SHL assessment! To give you the best recommendations, could you tell me what role you're hiring for?",
            "recommendations": [],
            "end_of_conversation": False,
        }

    # --- RETRIEVAL: Build query from full conversation and search ---
    query = retriever.build_query_from_messages(messages)
    retrieved_items = retriever.search(query, top_k=15)

    # --- BUILD PROMPT ---
    catalog_context = format_catalog_context(retrieved_items)
    conversation_text = build_conversation_text(messages)

    full_prompt = SYSTEM_PROMPT.replace("{catalog_context}", catalog_context)
    full_prompt = full_prompt.replace("{conversation_history}", conversation_text)

    # --- CALL LLM ---
    try:
        response = GEMINI_CLIENT.models.generate_content(
            model=MODEL_NAME,
            contents=full_prompt,
            config={
                "temperature": 0.3,
                "max_output_tokens": 1024,
                "response_mime_type": "application/json",
                "thinking_config": {"thinking_budget": 0},
            },
        )

        # Parse JSON response - handle various response formats
        response_text = response.text.strip() if response.text else ""

        # Clean up common JSON issues
        # Remove any BOM or zero-width chars
        response_text = response_text.strip("\ufeff\u200b\u200c\u200d")

        result = None

        # Try direct parse first
        try:
            result = json.loads(response_text)
        except (json.JSONDecodeError, TypeError):
            pass

        # Try to find the outermost JSON object
        if result is None:
            # Find the first { and last }
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx + 1]
                try:
                    result = json.loads(json_str)
                except json.JSONDecodeError:
                    # Try fixing common issues: trailing commas, single quotes
                    json_str = re.sub(r',\s*}', '}', json_str)
                    json_str = re.sub(r',\s*]', ']', json_str)
                    try:
                        result = json.loads(json_str)
                    except json.JSONDecodeError:
                        pass

        if result is None:
            result = {
                "reply": response_text[:500] if response_text else "I can help you find SHL assessments. What role are you hiring for?",
                "recommendations": [],
                "end_of_conversation": False,
            }

    except Exception as e:
        print(f"[Agent] LLM error: {e}")
        # Fallback: return retrieved items directly
        recs = []
        for item in retrieved_items[:5]:
            types = item.get("test_type", [])
            primary = types[0] if isinstance(types, list) and types else "K"
            recs.append({
                "name": item["name"],
                "url": item["url"],
                "test_type": primary,
            })
        result = {
            "reply": "Based on your requirements, here are some relevant SHL assessments:",
            "recommendations": recs,
            "end_of_conversation": False,
        }

    # --- POST-PROCESSING: Validate and fix ---
    result = validate_and_fix_response(result, retriever)

    return result
