#!/usr/bin/env python
# coding: utf-8
"""
Chainlit application for Cs_copilot - AI-powered chemical data analysis.
"""

import asyncio
import base64
import hashlib
import json
import mimetypes
import os
import re
import unicodedata
from pathlib import Path

import chainlit as cl
from chainlit.input_widget import Switch
from chainlit.types import ThreadDict
from dotenv import load_dotenv

from cs_copilot.agents.teams import get_cs_copilot_agent_team, get_qsar_agent_team
from cs_copilot.model_config import _is_retriable, arun_with_retry, load_model_from_config
from cs_copilot.storage import S3
from cs_copilot.tools.io.formatting import smiles_to_png_bytes
from cs_copilot.tools.reporting.qsar_latex import escape_latex
from cs_copilot.utils.logging import compact_log_data, get_logger, setup_logging

load_dotenv()
setup_logging()

# Set up logger
logger = get_logger(__name__)
runtime_logger = get_logger("cs_copilot.runtime")

# ---------- User Management System ----------------------------------------- #
# Simple in-memory user storage (in production, use a proper database)
USERS = {
    "admin": {"password_hash": hashlib.sha256("admin123".encode()).hexdigest(), "role": "admin"},
}


def verify_password(username: str, password: str) -> bool:
    """Verify user credentials against stored users."""
    if username not in USERS:
        return False

    password_hash = hashlib.sha256(password.encode()).hexdigest()
    return USERS[username]["password_hash"] == password_hash


def get_user_role(username: str) -> str:
    """Get user role for authorization."""
    return USERS.get(username, {}).get("role", "guest")


# ---------- Authentication Callback ---------------------------------------- #
@cl.password_auth_callback
async def auth_callback(username: str, password: str) -> cl.User | None:
    """Authenticate users based on username and password."""
    if verify_password(username, password):
        return cl.User(
            identifier=username,
            display_name=username.title(),
            metadata={"role": get_user_role(username), "username": username},
        )
    else:
        return None


# ❶ Instantiate the LLM (configured via .modelconf or MODEL_PROVIDER env var)
model = load_model_from_config()

# ❷ Define the agent factory (per chat thread)


def _default_team_mode() -> str:
    team_mode = os.getenv("CS_COPILOT_AGENT_TEAM", "main").strip().lower()
    return "qsar" if team_mode == "qsar" else "main"


def _create_session_agent(team_mode: str | None = None):
    """Create the configured team for the current app deployment."""
    team_mode = (team_mode or _default_team_mode()).strip().lower()
    team_factory = get_qsar_agent_team if team_mode == "qsar" else get_cs_copilot_agent_team
    logger.info("Initializing session agent team: %s", team_mode)
    return team_factory(
        model,
        show_members_responses=False,
    )


def _sync_shared_session_state(session_agent):
    if session_agent is None:
        return
    if session_agent.session_state is None:
        session_agent.session_state = {}

    uploaded_files = cl.user_session.get("uploaded_files_shared") or {}
    if uploaded_files:
        session_agent.session_state.setdefault("uploaded_files", {})
        session_agent.session_state["uploaded_files"].update(uploaded_files)


def _get_or_create_session_agent(team_mode: str):
    agents_by_mode = cl.user_session.get("agents_by_mode") or {}
    session_agent = agents_by_mode.get(team_mode)
    if session_agent is None:
        session_agent = _create_session_agent(team_mode)
        agents_by_mode[team_mode] = session_agent
        cl.user_session.set("agents_by_mode", agents_by_mode)
    _sync_shared_session_state(session_agent)
    cl.user_session.set("agent", session_agent)
    return session_agent


def _set_active_team_mode(team_mode: str):
    normalized = "qsar" if team_mode == "qsar" else "main"
    cl.user_session.set("active_team_mode", normalized)
    cl.user_session.set("last_team_mode", normalized)


def _get_active_team_mode() -> str:
    active_mode = cl.user_session.get("active_team_mode")
    if active_mode in {"main", "qsar"}:
        return active_mode
    last_mode = cl.user_session.get("last_team_mode")
    if last_mode in {"main", "qsar"}:
        return last_mode
    return _default_team_mode()


def _get_runtime_context() -> dict[str, str]:
    thread_id = getattr(cl.context.session, "thread_id", None)
    return {
        "team": _get_active_team_mode(),
        "thread": thread_id or "-",
    }


def _log_tool_event(stage: str, tool_name: str, payload=None):
    context = _get_runtime_context()
    suffix = ""
    compact_payload = compact_log_data(payload)
    if compact_payload != "-":
        suffix = f" | payload={compact_payload}"

    runtime_logger.info(
        "Tool %s: %s | team=%s | thread=%s%s",
        stage,
        tool_name or "tool",
        context["team"],
        context["thread"],
        suffix,
    )


def _normalize_router_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    ascii_like = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return ascii_like.strip().lower()


QSAR_REQUEST_PATTERNS = [
    r"\bqsar\b",
    r"\bchemprop\b",
    r"\bpredict(?:ion|ions|ive|er)?\b",
    r"\bpredire\b",
    r"\bpredit\b",
    r"\bpredictions?\b",
    r"\bapplicability domain\b",
    r"\bdomaine d[' ]applicabilite\b",
    r"\bstandard_qsar\b",
    r"\brobust_qsar\b",
    r"\bchallenging_qsar\b",
    r"\bfast_local\b",
    r"\bquick_train\b",
    r"\blipophilicity\b",
    r"\blipophilicite\b",
    r"\blogp\b",
    r"\blogd\b",
    r"\bcuration\b",
    r"\btrain(?:ing)? model\b",
    r"\bmodele valide\b",
]

NON_QSAR_PATTERNS = [
    r"\bgtm\b",
    r"\bchembl\b",
    r"\bretrosynth",
    r"\bsynplanner\b",
    r"\bautoencoder\b",
    r"\bpeptide\b",
    r"\bwae\b",
    r"\bclustering\b",
    r"\bchemotype\b",
    r"\bsimilarity\b",
    r"\bsar\b",
]

TEAM_FOLLOW_UP_PATTERNS = [
    r"^ok\b",
    r"^okay\b",
    r"^oui\b",
    r"^yes\b",
    r"^vas[ -]?y\b",
    r"^go\b",
    r"^continue\b",
    r"^on y va\b",
    r"^fais[- ]?le\b",
    r"^lance\b",
]


def _select_team_mode_for_message(user_text: str) -> str:
    text = _normalize_router_text(user_text)
    if text == "@latex":
        return "qsar"

    if any(re.search(pattern, text) for pattern in NON_QSAR_PATTERNS):
        return "main"
    if any(re.search(pattern, text) for pattern in QSAR_REQUEST_PATTERNS):
        return "qsar"

    last_mode = cl.user_session.get("last_team_mode")
    if last_mode in {"main", "qsar"} and any(
        re.search(pattern, text) for pattern in TEAM_FOLLOW_UP_PATTERNS
    ):
        return last_mode

    return "main"


# ---------- Chat lifecycle --------------------------------------------------- #
@cl.on_chat_start
async def on_chat_start():
    """Create a fresh agent for this chat thread and stash in session."""
    # Synchronize S3 session with Chainlit thread ID
    thread_id = cl.context.session.thread_id
    if thread_id:
        # Update S3 prefix to match Chainlit session
        S3.prefix = f"sessions/{thread_id}"
        logger.info(f"Set S3 session prefix to: {S3.prefix}")

    # Initialize session routing state for this chat thread.
    cl.user_session.set("agents_by_mode", {})
    cl.user_session.set("agent", None)
    cl.user_session.set("active_team_mode", _default_team_mode())
    cl.user_session.set("last_team_mode", None)
    cl.user_session.set("title_set", False)
    cl.user_session.set("session_initialized", True)

    # Initialize ChatSettings with tool call toggle
    settings = await cl.ChatSettings(
        [
            Switch(
                id="show_tool_calls",
                label="Show Tool Calls",
                initial=True,
            ),
        ]
    ).send()
    cl.user_session.set("show_tool_calls", settings["show_tool_calls"])


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """Resume existing chat session or create new agent if needed."""
    # Synchronize S3 session with Chainlit thread ID
    thread_id = cl.context.session.thread_id
    if thread_id:
        # Update S3 prefix to match Chainlit session
        S3.prefix = f"sessions/{thread_id}"
        logger.info(f"Resumed S3 session prefix: {S3.prefix}")

    # Restore or initialize per-mode routing state without forcing a single team.
    if not cl.user_session.get("session_initialized"):
        cl.user_session.set("agents_by_mode", {})
        cl.user_session.set("agent", None)
        cl.user_session.set("active_team_mode", _default_team_mode())
        cl.user_session.set("last_team_mode", None)
        cl.user_session.set("session_initialized", True)
    else:
        if cl.user_session.get("agents_by_mode") is None:
            cl.user_session.set("agents_by_mode", {})
        if cl.user_session.get("active_team_mode") not in {"main", "qsar"}:
            cl.user_session.set("active_team_mode", _default_team_mode())

    # Restore ChatSettings on resume
    settings = await cl.ChatSettings(
        [
            Switch(
                id="show_tool_calls",
                label="Show Tool Calls",
                initial=True,
            ),
        ]
    ).send()
    cl.user_session.set("show_tool_calls", settings["show_tool_calls"])


@cl.on_settings_update
async def on_settings_update(settings):
    """Handle settings updates from the UI."""
    cl.user_session.set("show_tool_calls", settings["show_tool_calls"])


# Note: on_chat_end can cause issues with some Chainlit versions
# Session cleanup is handled automatically by Chainlit
@cl.on_chat_end
async def on_chat_end():
    """Save the messages history"""
    session_agent = cl.user_session.get("agent")
    if session_agent is not None:
        try:
            # Get thread_id for session identification
            thread_id = cl.context.session.thread_id if hasattr(cl.context, "session") else None

            # Try to get messages from the Team's database if available
            # Note: Team doesn't have get_session_messages(), messages are stored in the database
            if hasattr(session_agent, "db") and session_agent.db is not None and thread_id:
                # Messages are stored in the database, but accessing them requires
                # the Agno database API which may not be available at chat end
                # Chainlit already handles message persistence, so we skip manual saving
                logger.debug(f"Chat ended for thread {thread_id}, messages persisted by Chainlit")
            else:
                logger.debug("No database available or thread_id missing, skipping message save")
        except Exception as e:
            # Gracefully handle any errors since this is optional functionality
            logger.debug(f"Could not save messages on chat end: {e}")


# ---------- regex helpers --------------------------------------------------- #
PATH_RX = re.compile(
    r"^\s*(.*?)\s*[:\-]\s*(/[^ \t]+?\.(?:png|jpe?g|gif|svg))\s*$", re.I
)  # Caption: /path/file.png
SMI_RX = re.compile(r"`?<smiles>([^<]+)</smiles>`?")  # explicit SMILES tags
INLINE_ELEMENT_RX = re.compile(
    r"!\[([^\]]*)\]\(([^)]+)\)|<file>(.*?)</file>",
    re.I,
)

FILE_DOWNLOAD_MODE = os.getenv("CHAINLIT_FILE_DOWNLOAD_MODE", "local").strip().lower()
MAX_INLINE_FILE_BYTES = int(os.getenv("CHAINLIT_FILE_INLINE_MAX_BYTES", str(10 * 1024 * 1024)))


# ---------- utilities ------------------------------------------------------- #
def _pretty(x):
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


def _is_qsar_team_mode() -> bool:
    return _get_active_team_mode() == "qsar"


_QSAR_HANDOFF_LABEL_RE = re.compile(
    r"(?im)^\s*(?:REPORT_LANGUAGE|HANDOFF_STATUS|SELECTED_MODEL|REASONS|"
    r"PREDICTION_TABLE|FILES|CAVEATS|DATASET|COMPUTE|SPLIT|METRICS|"
    r"MODEL_ID|REGISTRY_STATUS|BLOCKERS|NEXT_STEP|SOURCE_DATASET|"
    r"RETAINED_COLUMNS|ROW_COUNTS|CURATION_POLICY|WARNINGS|FINAL_STATUS)\s*[:：]"
)


def _find_polished_qsar_report_start(full_content: str, *, after: int = -1) -> int:
    """Find the start of a human-facing QSAR report inside a team trace."""
    polished_patterns = [
        r"(?im)^\s*Rapport(?:\s|$|[QSARdEéeD'’:\-—])",
        r"(?im)^\s*pEC50 Prediction Report\b",
        r"(?im)^\s*Prediction Report\b",
        r"(?im)^\s*Résumé de l['’]ensemble\b",
        r"(?im)^\s*Resume de l['’]ensemble\b",
        r"(?im)^\s*Ensemble Summary\b",
        r"(?im)^\s*Résumé des prédictions\b",
        r"(?im)^\s*Resume des predictions\b",
        r"(?im)^\s*Résumé des résultats\b",
        r"(?im)^\s*Resume des resultats\b",
        r"(?i)L['’]inférence a été réalisée",
        r"(?i)L['’]inference a ete realisee",
        r"(?i)Voici le résumé",
        r"(?i)Voici le resume",
        r"(?i)Voici les résultats",
        r"(?i)Voici les resultats",
        r"(?i)Here are the results",
    ]
    starts: list[int] = []
    for pattern in polished_patterns:
        for match in re.finditer(pattern, full_content):
            if match.start() > after:
                starts.append(match.start())
    return min(starts) if starts else -1


def _find_latest_backend_inventory_start(full_content: str, *, after: int = -1) -> int:
    """Find the latest backend capability inventory inside a verbose team trace."""
    backend_inventory_patterns = [
        r"(?im)^\s*(?:[#*_>\-\s]*)(?:[^\w\s]\s*)?Backends QSAR disponibles\b",
        r"(?im)^\s*(?:[#*_>\-\s]*)(?:[^\w\s]\s*)?Available QSAR backends\b",
        r"(?im)^\s*(?:[#*_>\-\s]*)(?:[^\w\s]\s*)?QSAR backends available\b",
        r"(?im)^\s*Voici une description complète (?:de tous )?(?:des|de\s+\d+\s+)?backends QSAR disponibles\b",
        r"(?im)^\s*Voici une description complete (?:de tous )?(?:des|de\s+\d+\s+)?backends QSAR disponibles\b",
        r"(?im)^\s*Here (?:is|are) (?:a )?complete description of (?:all )?(?:the )?available QSAR backends\b",
    ]
    starts: list[int] = []
    for pattern in backend_inventory_patterns:
        for match in re.finditer(pattern, full_content):
            if match.start() > after:
                starts.append(match.start())
    return max(starts) if starts else -1


def _extract_qsar_final_report(full_content: str) -> str:
    """Best-effort extraction of the final QSAR report from a verbose team trace."""
    if not full_content:
        return ""

    marked = re.findall(r"<qsar_report>\s*(.*?)\s*</qsar_report>", full_content, flags=re.S | re.I)
    if marked:
        return marked[-1].strip()

    handoff_matches = list(_QSAR_HANDOFF_LABEL_RE.finditer(full_content))
    last_handoff_start = handoff_matches[-1].start() if handoff_matches else -1

    latest_backend_inventory = _find_latest_backend_inventory_start(
        full_content,
        after=last_handoff_start,
    )
    if latest_backend_inventory >= 0:
        return full_content[latest_backend_inventory:].strip()

    polished_after_handoff = _find_polished_qsar_report_start(
        full_content,
        after=last_handoff_start,
    )
    if polished_after_handoff >= 0:
        return full_content[polished_after_handoff:].strip()

    polished_start = _find_polished_qsar_report_start(full_content)
    if polished_start >= 0:
        candidate = full_content[polished_start:].strip()
        handoff_tail = _QSAR_HANDOFF_LABEL_RE.search(candidate)
        if handoff_tail:
            candidate = candidate[:handoff_tail.start()].strip()
        if candidate:
            return candidate

    latest_backend_inventory = _find_latest_backend_inventory_start(full_content)
    if latest_backend_inventory >= 0:
        return full_content[latest_backend_inventory:].strip()

    report_markers = [
        "QSAR Dataset Curation Report",
        "Complete Training Workflow Report",
        "Lipophilicity Predictions for Simple Molecules",
        "Lipophilicity Predictions",
        "Model Selection Analysis",
        "Selected Model:",
        "Prediction Results",
        "Workflow Summary",
        "Final Status",
    ]

    last_idx = -1
    for marker in report_markers:
        idx = full_content.rfind(marker)
        if idx > last_idx:
            last_idx = idx

    if last_idx >= 0:
        return full_content[last_idx:].strip()

    if handoff_matches:
        return full_content[last_handoff_start:].strip()

    return full_content.strip()


def _store_latest_qsar_report(full_content: str) -> str:
    """Persist the latest QSAR report text for later export, with safe fallbacks."""
    final_report = _extract_qsar_final_report(full_content)
    cleaned = final_report.strip() if final_report else ""
    if not cleaned:
        cleaned = (full_content or "").strip()
    if cleaned:
        cl.user_session.set("last_qsar_report_markdown", cleaned)
    return cleaned


def _looks_like_markdown_table_line(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if re.fullmatch(r"\|?[\s:\-]+\|[\s\|:\-]*", stripped):
        return True
    return stripped.count("|") >= 2


def _strip_smiles_tags(text: str) -> str:
    return SMI_RX.sub(lambda m: m.group(1), text)


def _should_suppress_file_path_line(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    attached_refs = cl.user_session.get("attached_file_refs") or set()
    normalized_refs = {ref.strip() for ref in attached_refs}
    normalized_names = {Path(ref).name for ref in normalized_refs if ref.strip()}
    dequoted = stripped.strip("`")

    if stripped in normalized_refs or dequoted in normalized_names:
        return True

    for ref in normalized_refs:
        if ref and ref in stripped:
            return True

    for name in normalized_names:
        if name and name in dequoted and "/" not in dequoted:
            return True

    return False


def _process_smiles_in_text(text: str, callback):
    """
    Process SMILES patterns in text and call callback for each part.

    Args:
        text: Text to process
        callback: Function called with (text_part, is_smiles, smiles_string)
                 where is_smiles is True for SMILES tokens and False for regular text
    """
    pos = 0
    for m in SMI_RX.finditer(text):
        smi = m.group(1)

        # Add text before SMILES
        if m.start() > pos:
            callback(text[pos : m.start()], False, None)

        # Add SMILES token
        callback(f"`{smi}`", True, smi)

        pos = m.end()

    # Add remaining text
    if pos < len(text):
        callback(text[pos:], False, None)


async def _image_bubble(caption: str, src: str) -> cl.Message:
    """Generic local/remote image → cl.Image bubble → new assistant msg."""
    logger.debug(f"_image_bubble called with caption='{caption}', src='{src}'")
    p = Path(src).expanduser()
    name = caption or p.name

    def _to_data_url(data: bytes, filename: str) -> str:
        mt = mimetypes.guess_type(filename)[0] or "image/png"
        b64 = base64.b64encode(data).decode()
        data_size = len(data)
        logger.debug(f"Converted {data_size} bytes to data URL (mime: {mt})")
        return f"data:{mt};base64,{b64}"

    # Prefer local file → data URL; HTTP(S)/data: → pass-through; otherwise try S3 → data URL
    if p.is_file():
        logger.info(f"Loading image from local file: {p}")
        file_data = p.read_bytes()
        logger.debug(f"Read {len(file_data)} bytes from local file")
        data_url = _to_data_url(file_data, p.name)
        img_el = cl.Image(url=data_url, name=name, display="inline")
        logger.debug(f"Created cl.Image element from local file: {name}")
    elif isinstance(src, str) and (
        src.startswith("http://") or src.startswith("https://") or src.startswith("data:")
    ):
        logger.info(f"Using image from URL/data URL: {src[:100]}...")
        img_el = cl.Image(url=src, name=name, display="inline")
        logger.debug(f"Created cl.Image element from URL: {name}")
    else:
        try:
            logger.info(f"Attempting to load image from S3: {src}")
            with S3.open(src, "rb") as fh:
                data = fh.read()
            logger.debug(f"Read {len(data)} bytes from S3")
            data_url = _to_data_url(data, name)
            img_el = cl.Image(url=data_url, name=name, display="inline")
            logger.debug(f"Created cl.Image element from S3: {name}")
        except Exception as e:
            logger.warning(f"Error loading from S3, falling back to URL: {type(e).__name__}: {e}")
            # Fallback: let client try to fetch as URL (e.g. if it's a presigned S3 HTTP URL)
            img_el = cl.Image(url=src, name=name, display="inline")
            logger.debug(f"Created cl.Image element with fallback URL: {name}")

    logger.info(f"Sending image message with caption='{caption or img_el.name}'")
    await cl.Message(content=f"{caption or img_el.name}", elements=[img_el]).send()
    logger.debug("Image message sent successfully")


async def _create_streaming_message() -> cl.Message:
    """Create a message for streaming - only send when we have content"""
    msg = cl.Message(content="", author="assistant")
    return msg


async def _stream_text_to_message(text: str, msg: cl.Message):
    """Stream text to message, handling SMILES with cl.Image when interrupted"""
    if not text:
        return

    # Send message if not sent yet
    if not hasattr(msg, "_sent") or not msg._sent:
        await msg.send()

    # Process text with SMILES
    pos = 0
    for m in SMI_RX.finditer(text):
        smi = m.group(1)
        logger.debug(f"Detected SMILES in stream: '{smi}'")

        # Stream text up to SMILES
        if m.start() > pos:
            await msg.stream_token(text[pos : m.start()])

        # Stream SMILES token
        await msg.stream_token(f"`{smi}`")
        logger.debug(f"Streamed SMILES token: `{smi}`")

        # Try to create molecule image and send as cl.Image
        try:
            logger.info(f"Attempting to convert SMILES to PNG: '{smi}'")
            png = smiles_to_png_bytes(smi)
            if png is not None:
                png_size = len(png)
                logger.info(f"Successfully generated PNG from SMILES '{smi}' ({png_size} bytes)")
                b64 = base64.b64encode(png).decode()
                data_url = f"data:image/png;base64,{b64}"
                logger.debug(f"Created data URL from PNG (size: {len(b64)} chars)")
                img_el = cl.Image(url=data_url, name=smi, display="inline")
                logger.debug(f"Created cl.Image element for SMILES: '{smi}'")
                await cl.Message(content=f"`{smi}`", elements=[img_el]).send()
                logger.info(f"Sent SMILES image message for: '{smi}'")
                # Return new streaming message for continuation
                return await _create_streaming_message()
            else:
                logger.info(f"smiles_to_png_bytes returned None for SMILES: '{smi}'")
        except ValueError as ve:
            logger.info(f"ValueError converting SMILES '{smi}' to image: {ve}")
            # Invalid SMILES, just continue without image
        except Exception as e:
            logger.info(f"Exception converting SMILES '{smi}' to image: {type(e).__name__}: {e}")
            # Invalid SMILES, just continue without image

        pos = m.end()

    # Stream remaining text
    if pos < len(text):
        await msg.stream_token(text[pos:])


async def _stream_plain_text_to_message(text: str, msg: cl.Message):
    """Stream text without SMILES expansion."""
    if not text:
        return

    if not hasattr(msg, "_sent") or not msg._sent:
        await msg.send()

    await msg.stream_token(text)


async def _image_bubble_streaming(caption: str, src: str) -> cl.Message:
    """Send image bubble and return new streaming message"""
    logger.debug(f"_image_bubble_streaming called with caption='{caption}', src='{src}'")
    p = Path(src).expanduser()
    name = caption or p.name

    def _to_data_url(data: bytes, filename: str) -> str:
        mt = mimetypes.guess_type(filename)[0] or "image/png"
        b64 = base64.b64encode(data).decode()
        data_size = len(data)
        logger.debug(f"Converted {data_size} bytes to data URL (mime: {mt})")
        return f"data:{mt};base64,{b64}"

    # Prefer local file → data URL; HTTP(S)/data: → pass-through; otherwise try S3 → data URL
    if p.is_file():
        logger.info(f"Loading image from local file (streaming): {p}")
        file_data = p.read_bytes()
        logger.debug(f"Read {len(file_data)} bytes from local file")
        data_url = _to_data_url(file_data, p.name)
        img_el = cl.Image(url=data_url, name=name, display="inline")
        logger.debug(f"Created cl.Image element from local file (streaming): {name}")
    elif isinstance(src, str) and (
        src.startswith("http://") or src.startswith("https://") or src.startswith("data:")
    ):
        logger.info(f"Using image from URL/data URL (streaming): {src[:100]}...")
        img_el = cl.Image(url=src, name=name, display="inline")
        logger.debug(f"Created cl.Image element from URL (streaming): {name}")
    else:
        try:
            logger.info(f"Attempting to load image from S3 (streaming): {src}")
            with S3.open(src, "rb") as fh:
                data = fh.read()
            logger.debug(f"Read {len(data)} bytes from S3")
            data_url = _to_data_url(data, name)
            img_el = cl.Image(url=data_url, name=name, display="inline")
            logger.debug(f"Created cl.Image element from S3 (streaming): {name}")
        except Exception as e:
            logger.warning(f"Error loading from S3 (streaming), falling back to URL: {type(e).__name__}: {e}")
            # Fallback: let client try to fetch as URL (e.g. if it's a presigned S3 HTTP URL)
            img_el = cl.Image(url=src, name=name, display="inline")
            logger.debug(f"Created cl.Image element with fallback URL (streaming): {name}")

    logger.info(f"Sending image message (streaming) with caption='{caption or img_el.name}'")
    await cl.Message(content=f"{caption or img_el.name}", elements=[img_el]).send()
    logger.debug("Image message sent successfully (streaming)")
    return await _create_streaming_message()


def _is_web_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


def _guess_file_name(path: str) -> str:
    cleaned = path.strip().strip("`").strip("\"").strip("'")
    without_query = cleaned.split("?", 1)[0]
    name = Path(without_query).name
    if not name:
        digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:10]
        return f"download_{digest}"
    return name


def _safe_file_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def _read_file_bytes_from_storage(file_ref: str) -> bytes:
    cleaned = file_ref.strip().strip("`").strip("\"").strip("'")
    local_candidate = Path(cleaned).expanduser()
    if local_candidate.exists():
        return local_candidate.read_bytes()
    with S3.open(file_ref, "rb") as fh:
        return fh.read()


async def _materialize_file_to_local(file_ref: str, file_name: str) -> Path:
    cache = cl.user_session.get("downloadable_files_cache") or {}
    cached = cache.get(file_ref)
    if cached:
        cached_path = Path(cached)
        if cached_path.exists():
            return cached_path

    downloads_dir = Path(".files") / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256(file_ref.encode("utf-8")).hexdigest()[:12]
    local_path = downloads_dir / f"{digest}_{_safe_file_name(file_name)}"

    if not local_path.exists():
        data = await asyncio.to_thread(_read_file_bytes_from_storage, file_ref)
        await asyncio.to_thread(local_path.write_bytes, data)
        logger.info("Downloaded file from storage to local cache: %s", local_path)

    cache[file_ref] = str(local_path)
    cl.user_session.set("downloadable_files_cache", cache)
    return local_path


async def _build_download_file_element(file_ref: str) -> cl.File:
    file_name = _guess_file_name(file_ref)

    if _is_web_url(file_ref):
        return cl.File(name=file_name, url=file_ref, display="inline")

    if FILE_DOWNLOAD_MODE == "content":
        try:
            data = await asyncio.to_thread(_read_file_bytes_from_storage, file_ref)
            if len(data) <= MAX_INLINE_FILE_BYTES:
                return cl.File(name=file_name, content=data, display="inline")
            logger.info(
                "File %s is too large for inline bytes (%d > %d), falling back to local path mode",
                file_ref,
                len(data),
                MAX_INLINE_FILE_BYTES,
            )
        except Exception as e:
            logger.warning(
                "Could not stream file bytes directly for %s (%s: %s). Falling back to local path mode.",
                file_ref,
                type(e).__name__,
                e,
            )

    local_path = await _materialize_file_to_local(file_ref, file_name)
    return cl.File(name=file_name, path=str(local_path), display="inline")


async def _file_bubble_streaming(file_ref: str) -> cl.Message:
    normalized_ref = file_ref.strip()
    if not normalized_ref:
        return await _create_streaming_message()

    try:
        file_el = await _build_download_file_element(normalized_ref)
        await cl.Message(content=f"`{file_el.name}`", elements=[file_el]).send()
        attached_refs = cl.user_session.get("attached_file_refs") or set()
        attached_refs.add(normalized_ref)
        cl.user_session.set("attached_file_refs", attached_refs)
    except FileNotFoundError as e:
        logger.warning(
            "Downloadable file tag points to a missing file: %s (%s)",
            normalized_ref,
            e,
        )
        await cl.Message(
            content=f"Could not prepare downloadable file: `{normalized_ref}`",
            author="assistant",
        ).send()
    except Exception as e:
        logger.error(
            "Failed to create downloadable file for %s: %s: %s",
            normalized_ref,
            type(e).__name__,
            e,
            exc_info=True,
        )
        await cl.Message(
            content=f"Could not prepare downloadable file: `{normalized_ref}`",
            author="assistant",
        ).send()

    return await _create_streaming_message()


def _find_prediction_export_agent(agent_like):
    """Find an agent-like object carrying prediction history in session state."""
    if agent_like is None:
        return None

    state = getattr(agent_like, "session_state", None) or {}
    prediction_state = state.get("prediction_models", {}) if isinstance(state, dict) else {}
    history = prediction_state.get("prediction_history") or []
    if history:
        return agent_like

    for member in getattr(agent_like, "members", []) or []:
        resolved = _find_prediction_export_agent(member)
        if resolved is not None:
            return resolved
    return None


def _markdown_report_to_latex(report_text: str, title: str = "Rapport QSAR") -> str:
    """Convert a markdown-like QSAR report into a styled LaTeX document."""
    lines = report_text.splitlines()
    body: list[str] = []
    in_list = False
    in_enum = False
    table_buffer: list[str] = []
    extracted_title = title
    intro_paragraph = ""

    def close_list():
        nonlocal in_list, in_enum
        if in_list:
            body.append(r"\end{itemize}")
            in_list = False
        if in_enum:
            body.append(r"\end{enumerate}")
            in_enum = False

    def _convert_inline(text: str) -> str:
        placeholders: list[str] = []

        def _store(value: str) -> str:
            placeholders.append(value)
            return f"@@PLACEHOLDER_{len(placeholders) - 1}@@"

        converted = text
        converted = re.sub(
            r"<file>(.*?)</file>",
            lambda m: _store(r"\texttt{" + escape_latex(m.group(1).strip()) + "}"),
            converted,
        )
        converted = re.sub(
            r"`([^`]+)`",
            lambda m: _store(r"\texttt{" + escape_latex(m.group(1)) + "}"),
            converted,
        )
        converted = re.sub(
            r"\*\*([^*]+)\*\*",
            lambda m: _store(r"\textbf{" + escape_latex(m.group(1)) + "}"),
            converted,
        )
        converted = re.sub(
            r"\*([^*]+)\*",
            lambda m: _store(r"\emph{" + escape_latex(m.group(1)) + "}"),
            converted,
        )
        converted = escape_latex(converted)
        for index, value in enumerate(placeholders):
            converted = converted.replace(escape_latex(f"@@PLACEHOLDER_{index}@@"), value)
        return converted

    def flush_table():
        nonlocal table_buffer
        if not table_buffer:
            return
        parsed_rows = []
        for raw in table_buffer:
            stripped = raw.strip().strip("|")
            cells = [cell.strip() for cell in stripped.split("|")]
            parsed_rows.append(cells)
        table_buffer = []
        if len(parsed_rows) < 2:
            for row in parsed_rows:
                body.append(" ".join(_convert_inline(cell) for cell in row))
            return
        header = parsed_rows[0]
        data_rows = [row for row in parsed_rows[2:] if any(cell for cell in row)]
        col_count = max(len(header), max((len(r) for r in data_rows), default=0), 1)
        widths = {
            1: "X",
            2: ">{\\raggedright\\arraybackslash}p{0.30\\textwidth}X",
            3: ">{\\raggedright\\arraybackslash}p{0.22\\textwidth}>{\\raggedright\\arraybackslash}p{0.22\\textwidth}X",
            4: ">{\\raggedright\\arraybackslash}p{0.22\\textwidth}>{\\raggedright\\arraybackslash}p{0.22\\textwidth}>{\\raggedright\\arraybackslash}p{0.22\\textwidth}X",
        }
        colspec = widths.get(
            col_count,
            " ".join([">{\\raggedright\\arraybackslash}p{0.15\\textwidth}"] * (col_count - 1))
            + " X",
        )
        body.append(r"\renewcommand{\arraystretch}{1.2}")
        body.append(r"\rowcolors{2}{prismgray}{white}")
        body.append(r"\begin{tabularx}{\textwidth}{" + colspec + "}")
        body.append(r"\toprule")
        body.append(r"\rowcolor{prismblue!15}")
        padded_header = header + [""] * (col_count - len(header))
        body.append(" & ".join(_convert_inline(cell) for cell in padded_header[:col_count]) + r" \\")
        body.append(r"\midrule")
        for row in data_rows:
            padded = row + [""] * (col_count - len(row))
            body.append(" & ".join(_convert_inline(cell) for cell in padded[:col_count]) + r" \\")
        body.append(r"\bottomrule")
        body.append(r"\end{tabularx}")

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            flush_table()
            close_list()
            body.append("")
            continue

        if _looks_like_markdown_table_line(line):
            close_list()
            table_buffer.append(line)
            continue
        flush_table()

        if line.startswith("# "):
            close_list()
            heading = line[2:].strip()
            if extracted_title == title:
                extracted_title = heading
            else:
                body.append(r"\section*{" + _convert_inline(heading) + "}")
            continue
        if line.startswith("## "):
            close_list()
            heading = line[3:].strip()
            if not body and not intro_paragraph:
                extracted_title = heading
            else:
                body.append(r"\section*{" + _convert_inline(heading) + "}")
            continue
        normalized_heading = line
        if normalized_heading.startswith(r"\#"):
            normalized_heading = normalized_heading.replace(r"\#", "#")

        if normalized_heading.startswith("### "):
            close_list()
            body.append(r"\subsection*{" + _convert_inline(normalized_heading[4:].strip()) + "}")
            continue
        if normalized_heading.startswith("#### "):
            close_list()
            body.append(r"\subsubsection*{" + _convert_inline(normalized_heading[5:].strip()) + "}")
            continue
        if line.startswith("- ") or line.startswith("* "):
            if in_enum:
                body.append(r"\end{enumerate}")
                in_enum = False
            if not in_list:
                body.append(r"\begin{itemize}")
                in_list = True
            body.append(r"\item " + _convert_inline(line[2:].strip()))
            continue
        if re.match(r"^\d+\.\s+", line):
            if in_list:
                body.append(r"\end{itemize}")
                in_list = False
            if not in_enum:
                body.append(r"\begin{enumerate}")
                in_enum = True
            item_text = re.sub(r"^\d+\.\s+", "", line)
            body.append(r"\item " + _convert_inline(item_text.strip()))
            continue

        close_list()
        converted_line = _convert_inline(line)
        if not intro_paragraph and not body:
            intro_paragraph = converted_line
        else:
            body.append(converted_line)

    flush_table()
    close_list()

    document_title = extracted_title or title

    return "\n".join(
        [
            r"\documentclass[11pt,a4paper]{article}",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage[french]{babel}",
            r"\usepackage{geometry}",
            r"\usepackage{booktabs}",
            r"\usepackage[table]{xcolor}",
            r"\usepackage{array}",
            r"\usepackage{tabularx}",
            r"\usepackage{enumitem}",
            r"\usepackage{hyperref}",
            r"\usepackage{titlesec}",
            r"\usepackage[most]{tcolorbox}",
            r"\geometry{margin=2.5cm}",
            r"\setlength{\parindent}{0pt}",
            r"\setlength{\parskip}{0.6em}",
            r"\definecolor{prismblue}{RGB}{34,79,117}",
            r"\definecolor{prismgray}{RGB}{245,247,250}",
            r"\definecolor{prismline}{RGB}{210,216,224}",
            r"\definecolor{prismlightblue}{RGB}{234,242,249}",
            r"\hypersetup{hidelinks}",
            r"\titleformat{name=\section,numberless}{\Large\bfseries\color{prismblue}}{}{0pt}{}",
            r"\tcbset{enhanced,boxrule=0.8pt,arc=2mm,left=4mm,right=4mm,top=2.5mm,bottom=2.5mm}",
            r"\begin{document}",
            r"\begin{tcolorbox}[colback=prismblue,colframe=prismblue,boxrule=0pt,arc=0mm]",
            r"{\color{white}\bfseries\LARGE Rapport QSAR}\hfill {\color{white}\large ChemSpaceCopilot}",
            "",
            r"{\color{white!90}\large " + _convert_inline(extracted_title) + "}",
            r"\end{tcolorbox}",
            "",
            (
                r"\begin{tcolorbox}[colback=prismlightblue,colframe=prismblue,title=Synthese rapide,fonttitle=\bfseries]"
                + "\n"
                + intro_paragraph
                + "\n"
                + r"\end{tcolorbox}"
                if intro_paragraph
                else ""
            ),
            r"\title{" + _convert_inline(document_title) + "}",
            r"\author{ChemSpaceCopilot}",
            r"\date{}",
            *body,
            r"\end{document}",
            "",
        ]
    )


async def _handle_latex_shortcut(session_agent) -> bool:
    """Handle `@Latex` as a direct export command for the latest prediction."""
    last_report = cl.user_session.get("last_qsar_report_markdown")
    if not last_report:
        await cl.Message(
            content="Aucun rapport QSAR final n'est disponible pour generer un fichier LaTeX.",
            author="assistant",
        ).send()
        return True

    try:
        reports_dir = (Path(".files") / "qsar_reports").resolve()
        reports_dir.mkdir(parents=True, exist_ok=True)
        basename = "latest_qsar_report"
        tex_path = reports_dir / f"{basename}.tex"
        source_path = reports_dir / f"{basename}.md"

        source_path.write_text(last_report, encoding="utf-8")
        tex_path.write_text(
            _markdown_report_to_latex(last_report, title="Rapport QSAR"),
            encoding="utf-8",
        )

        await cl.Message(content="Export LaTeX genere.", author="assistant").send()
        await _file_bubble_streaming(str(tex_path))
        await _file_bubble_streaming(str(source_path))
    except Exception as e:
        logger.error("LaTeX shortcut export failed: %s: %s", type(e).__name__, e, exc_info=True)
        await cl.Message(
            content=f"Echec de l'export LaTeX : {type(e).__name__}: {e}",
            author="assistant",
        ).send()
    return True


async def _stream_line_with_elements(
    line: str,
    assistant: cl.Message | None,
    append_newline: bool = True,
) -> cl.Message:
    if "<file>" in line:
        line = re.sub(r"^\s*[-*•]\s+", "", line)
        line = re.sub(r"\s+[—-]\s*<file>", " <file>", line)

    if _should_suppress_file_path_line(line):
        return assistant or await _create_streaming_message()

    if _looks_like_markdown_table_line(line):
        if assistant is None:
            assistant = await _create_streaming_message()
        table_line = _strip_smiles_tags(line)
        await _stream_plain_text_to_message(
            table_line + ("\n" if append_newline else ""),
            assistant,
        )
        return assistant

    # 1) stand-alone Caption: /path/img.png
    if m := PATH_RX.fullmatch(line.strip()):
        caption, src = m.groups()
        logger.info(f"Relay detected image path pattern: caption='{caption}', src='{src}'")
        if assistant is None:
            assistant = await _create_streaming_message()
        await _stream_text_to_message(line, assistant)
        return await _image_bubble_streaming(caption.strip(), src)

    # 2) inline markdown images and <file> tags, preserving order
    pos = 0
    for m in INLINE_ELEMENT_RX.finditer(line):
        if m.start() > pos:
            leading_text = line[pos : m.start()]
            if file_src := m.group(3):
                leading_text = leading_text.rstrip(" -—:\t")
            if assistant is None:
                assistant = await _create_streaming_message()
            await _stream_text_to_message(leading_text, assistant)

        image_alt, image_src, file_src = m.groups()
        if image_src is not None:
            logger.info(f"Relay detected markdown image: alt='{image_alt}', src='{image_src}'")
            assistant = await _image_bubble_streaming(image_alt, image_src)
        elif file_src is not None:
            normalized_file_src = file_src.strip()
            logger.info("Relay detected file tag: '%s'", normalized_file_src)
            assistant = await _file_bubble_streaming(normalized_file_src)

        pos = m.end()

    # 3) tail text
    if assistant is None:
        assistant = await _create_streaming_message()
    tail = line[pos:] + ("\n" if append_newline else "")
    new_assistant = await _stream_text_to_message(tail, assistant)
    if new_assistant is not None:
        assistant = new_assistant

    return assistant


# async def _update_message_for_persistence(msg: cl.Message, full_content: str):
#     """Update message with complete content for proper persistence"""
#     # Process the full content to ensure proper persistence with SMILES images
#     processed_content = _process_content_for_persistence(full_content)
#     if processed_content != msg.content:
#         msg.content = processed_content
#         await msg.update()

# def _process_content_for_persistence(content: str) -> str:
#     """Process content to ensure proper persistence with SMILES tokens only"""
#     content_parts = []

#     def persistence_callback(text_part, is_smiles, smiles_string):
#         if is_smiles:
#             # Add SMILES token only (images are handled separately with cl.Image)
#             content_parts.append(text_part)
#         else:
#             # Add regular text
#             content_parts.append(text_part)

#     _process_smiles_in_text(content, persistence_callback)
#     return "".join(content_parts)


async def _send_text_with_smiles(text: str):
    """Send text message with SMILES processing using cl.Image"""
    logger.debug(f"_send_text_with_smiles called with text length: {len(text)}")
    # Check if there are any SMILES patterns
    if not SMI_RX.search(text):
        # No SMILES, send as regular message
        logger.debug("No SMILES patterns found in text, sending as regular message")
        await cl.Message(content=text, author="assistant").send()
        return

    logger.info("SMILES patterns detected, processing...")
    # Process text with SMILES
    pos = 0
    for m in SMI_RX.finditer(text):
        smi = m.group(1)
        logger.debug(f"Detected SMILES: '{smi}'")

        # Send text up to SMILES
        if m.start() > pos:
            await cl.Message(content=text[pos : m.start()], author="assistant").send()

        # Send SMILES token
        await cl.Message(content=f"`{smi}`", author="assistant").send()
        logger.debug(f"Sent SMILES token message: `{smi}`")

        # Try to create molecule image and send as cl.Image
        try:
            logger.info(f"Attempting to convert SMILES to PNG: '{smi}'")
            png = smiles_to_png_bytes(smi)
            if png is not None:
                png_size = len(png)
                logger.info(f"Successfully generated PNG from SMILES '{smi}' ({png_size} bytes)")
                b64 = base64.b64encode(png).decode()
                data_url = f"data:image/png;base64,{b64}"
                logger.debug(f"Created data URL from PNG (size: {len(b64)} chars)")
                img_el = cl.Image(url=data_url, name=smi, display="inline")
                logger.debug(f"Created cl.Image element for SMILES: '{smi}'")
                await cl.Message(content=f"`{smi}`", elements=[img_el], author="assistant").send()
                logger.info(f"Sent SMILES image message for: '{smi}'")
            else:
                logger.warning(f"smiles_to_png_bytes returned None for SMILES: '{smi}'")
        except ValueError as ve:
            logger.warning(f"ValueError converting SMILES '{smi}' to image: {ve}")
            # Invalid SMILES, just continue without image
        except Exception as e:
            logger.error(f"Exception converting SMILES '{smi}' to image: {type(e).__name__}: {e}")
            # Invalid SMILES, just continue without image

        pos = m.end()

    # Send remaining text
    if pos < len(text):
        await cl.Message(content=text[pos:], author="assistant").send()


async def _handle_file_uploads(files: list, session_id: str) -> list[str]:
    """
    Upload files to session-scoped storage in the current backend.

    Args:
        files: List of cl.File objects from the message
        session_id: Current Chainlit thread/session ID

    Returns:
        List of storage paths where files were uploaded
    """
    uploaded_paths = []
    logger.debug(f"_handle_file_uploads called with {len(files)} files")

    for file in files:
        try:
            logger.debug(f"Processing file: {file.name if hasattr(file, 'name') else 'unknown'}")
            logger.debug(f"File attributes: {dir(file)}")

            # Read file content
            file_content = None

            if hasattr(file, 'content') and file.content:
                file_content = file.content
                logger.debug(f"Got content from file.content ({len(file_content)} bytes)")
            elif hasattr(file, 'path') and file.path:
                # Read from file path
                logger.debug(f"Reading from file.path: {file.path}")
                with open(file.path, 'rb') as f:
                    file_content = f.read()
                logger.debug(f"Read {len(file_content)} bytes from file")

            if file_content is None:
                logger.warning(f"Could not read content from file {file.name}")
                continue

            # Upload to session storage using a stable uploads/ prefix.
            # S3.prefix is already set to sessions/{session_id} by on_chat_start/resume.
            relative_path = f"uploads/{file.name}"
            logger.debug(f"Relative storage path: {relative_path}")
            logger.debug(f"Current S3.prefix: {S3.prefix}")

            # Write file using the unified storage abstraction.
            logger.debug("Opening session storage file for writing...")
            with S3.open(relative_path, 'wb') as s3_file:
                s3_file.write(file_content)

            # Get the full storage path for display and later reuse by agents.
            full_storage_path = S3.path(relative_path)
            uploaded_paths.append(full_storage_path)
            logger.info(f"Uploaded file {file.name} to {full_storage_path}")

        except Exception as e:
            logger.error(f"Error uploading file {getattr(file, 'name', 'unknown')}: {type(e).__name__}: {e}", exc_info=True)
            # Continue with other files even if one fails
            continue

    logger.debug(f"Upload complete. {len(uploaded_paths)} files uploaded successfully")
    return uploaded_paths


# ---------- main relay ------------------------------------------------------ #
async def relay(stream):
    assistant = None  # Will be created when we have content
    buf = ""  # accumulate until newline
    full_content = ""  # collect all content for final message
    qsar_mode = _is_qsar_team_mode()
    qsar_progress_msg = None
    active_tool_calls: dict[str, dict] = {}
    active_tool_name_queues: dict[str, list[str]] = {}
    tool_event_sequence = 0

    # Check if tool calls should be displayed
    show_tool_calls = cl.user_session.get("show_tool_calls", True)

    if qsar_mode:
        runtime_logger.info(
            "QSAR workflow started | team=%s | thread=%s",
            _get_runtime_context()["team"],
            _get_runtime_context()["thread"],
        )
        qsar_progress_msg = cl.Message(content="Workflow QSAR en cours...", author="assistant")
        await qsar_progress_msg.send()

    def _extract_tool_name(tool) -> str:
        if tool is None:
            return "tool"
        return getattr(tool, "tool_name", None) or getattr(tool, "name", None) or "tool"

    def _extract_tool_call_id(chunk, tool) -> str | None:
        candidates = [
            getattr(chunk, "tool_call_id", None),
            getattr(chunk, "call_id", None),
            getattr(chunk, "id", None),
            getattr(tool, "tool_call_id", None),
            getattr(tool, "call_id", None),
            getattr(tool, "id", None),
            getattr(tool, "tool_use_id", None),
        ]
        for candidate in candidates:
            if candidate:
                return str(candidate)
        return None

    def _remember_active_tool(call_id: str, tool_name: str, step=None) -> None:
        active_tool_calls[call_id] = {"tool_name": tool_name, "step": step}
        active_tool_name_queues.setdefault(tool_name, []).append(call_id)

    def _pop_active_tool(call_id: str | None, tool_name: str | None) -> dict | None:
        resolved_id = None
        if call_id and call_id in active_tool_calls:
            resolved_id = call_id
        elif tool_name:
            queue = active_tool_name_queues.get(tool_name) or []
            while queue:
                candidate = queue.pop(0)
                if candidate in active_tool_calls:
                    resolved_id = candidate
                    break
        elif len(active_tool_calls) == 1:
            resolved_id = next(iter(active_tool_calls))

        if not resolved_id:
            return None

        entry = active_tool_calls.pop(resolved_id, None)
        if entry is None:
            return None

        queue = active_tool_name_queues.get(entry["tool_name"]) or []
        active_tool_name_queues[entry["tool_name"]] = [
            candidate for candidate in queue if candidate != resolved_id
        ]
        return entry

    async for chunk in stream:
        # ── tool events → COT sidebar as Steps ───────────────────────────────
        ev = getattr(chunk, "event", None)
        if ev == "ToolCallStarted":
            t = chunk.tool
            tool_name = _extract_tool_name(t)
            tool_args = t.tool_args or getattr(t, "arguments", {})
            call_id = _extract_tool_call_id(chunk, t)
            if call_id is None:
                tool_event_sequence += 1
                call_id = f"{tool_name}#{tool_event_sequence}"
            _log_tool_event("started", tool_name, tool_args)
            step = None
            if show_tool_calls:
                step = cl.Step(name=tool_name, type="tool")
                step.input = tool_args
                await step.send()
            _remember_active_tool(call_id, tool_name, step)
            continue

        if ev and ev.endswith("Completed"):
            tool = getattr(chunk, "tool", None)
            call_id = _extract_tool_call_id(chunk, tool)
            hinted_name = _extract_tool_name(tool) if tool is not None else None
            active_entry = _pop_active_tool(call_id, hinted_name)
            tool_name = active_entry["tool_name"] if active_entry else (hinted_name or "tool")
            payload = chunk.content or getattr(chunk, "text", "") or "done"
            _log_tool_event("completed", tool_name, payload)
            if show_tool_calls and active_entry and active_entry.get("step") is not None:
                active_entry["step"].output = chunk.content or "✅ done"
                await active_entry["step"].update()
            continue

        # ── plain text from the LLM / agent ─────────────────────────────────
        text = (
            chunk
            if isinstance(chunk, str)
            else getattr(chunk, "content", "") or getattr(chunk, "text", "")
        )
        if not text:
            continue

        # Collect for persistence
        full_content += text

        if qsar_mode:
            continue

        buf += text
        while "\n" in buf:  # process complete lines
            line, buf = buf.split("\n", 1)
            assistant = await _stream_line_with_elements(line, assistant, append_newline=True)

    # ── flush tail (no final newline) ────────────────────────────────────────
    if buf and not qsar_mode:
        assistant = await _stream_line_with_elements(buf, assistant, append_newline=False)

    if qsar_mode:
        final_report = _store_latest_qsar_report(full_content)
        if qsar_progress_msg is not None:
            qsar_progress_msg.content = "Workflow QSAR terminé."
            await qsar_progress_msg.update()
        runtime_logger.info(
            "QSAR workflow completed | team=%s | thread=%s",
            _get_runtime_context()["team"],
            _get_runtime_context()["thread"],
        )
        if final_report:
            final_assistant = None
            final_buf = final_report
            while "\n" in final_buf:
                line, final_buf = final_buf.split("\n", 1)
                final_assistant = await _stream_line_with_elements(
                    line,
                    final_assistant,
                    append_newline=True,
                )
            if final_buf:
                await _stream_line_with_elements(
                    final_buf,
                    final_assistant,
                    append_newline=False,
                )

    # # Update the final message with complete content for persistence
    # if assistant and full_content.strip():
    #     await _update_message_for_persistence(assistant, full_content)


# ---------- Chainlit entry-point ------------------------------------------- #
@cl.on_message
async def main(user_msg: cl.Message):
    try:
        # Ensure session is properly initialized
        if not cl.user_session.get("session_initialized"):
            await on_chat_start()

        # Lazily set chat title from the first user message
        if not cl.user_session.get("title_set"):
            try:
                await cl.set_chat_title(user_msg.content[:60] or "New chat")
            except Exception:
                pass
            cl.user_session.set("title_set", True)

        requested_team_mode = _select_team_mode_for_message(user_msg.content)
        _set_active_team_mode(requested_team_mode)

        # Ensure S3 session is synchronized
        thread_id = cl.context.session.thread_id
        if thread_id:
            S3.prefix = f"sessions/{thread_id}"
            logger.info(f"Set S3 session prefix in main(): {S3.prefix}")

        # Get or create the agent matching this request's routing mode.
        session_agent = _get_or_create_session_agent(requested_team_mode)

        if user_msg.content.strip() == "@Latex":
            await _handle_latex_shortcut(session_agent)
            return

        # Handle file uploads if present
        # Debug: Check multiple possible locations for files
        files = None

        # Try different ways files might be attached
        if hasattr(user_msg, 'files') and user_msg.files:
            files = user_msg.files
            logger.debug(f"Found files in user_msg.files: {[f.name for f in files]}")
        elif hasattr(user_msg, 'elements') and user_msg.elements:
            # Filter for File elements
            files = [el for el in user_msg.elements if isinstance(el, cl.File)]
            if files:
                logger.debug(f"Found files in user_msg.elements: {[f.name for f in files]}")

        if files:
            logger.debug(f"Processing {len(files)} file(s)")
            # Get thread ID for session-specific folder
            thread_id = cl.context.session.thread_id
            logger.debug(f"Thread ID: {thread_id}")

            if thread_id:
                uploaded_paths = await _handle_file_uploads(files, thread_id)
                logger.debug(f"Uploaded paths: {uploaded_paths}")

                if uploaded_paths:
                    # Store uploaded files in shared session state so both teams
                    # can access the same user uploads without cross-agent calls.
                    shared_uploaded_files = cl.user_session.get("uploaded_files_shared") or {}
                    for s3_path in uploaded_paths:
                        filename = s3_path.split('/')[-1]
                        shared_uploaded_files[filename] = s3_path
                        logger.info(f"Added to shared upload state: {filename} → {s3_path}")

                    cl.user_session.set("uploaded_files_shared", shared_uploaded_files)
                    _sync_shared_session_state(session_agent)
                    logger.info(f"Total files in shared upload state: {len(shared_uploaded_files)}")

                    # Display confirmation message
                    file_list = "\n".join(
                        [f"- `{path.split('/')[-1]}` → {path}" for path in uploaded_paths]
                    )
                    await cl.Message(
                        content=f"📁 **Files uploaded to session storage:**\n{file_list}",
                        author="assistant",
                    ).send()
                    logger.debug("Confirmation message sent to UI")
                else:
                    logger.debug("No files were successfully uploaded")
            else:
                logger.debug("No thread_id available, skipping upload")
        else:
            logger.debug("No files found in message")

        # Process the message with session-scoped memory.
        # Two layers of retry protect against transient Ollama errors
        # (e.g. malformed tool-call JSON):
        #  - Inner: arun_with_retry wraps the async stream with retry
        #    logic that is transparent to relay().
        #  - Outer: this loop catches errors that surface after relay()
        #    has already emitted partial UI content.  On retry a fresh
        #    stream + relay is started and the user is notified.
        max_retries = 3
        base_delay = 2.0
        for attempt in range(max_retries + 1):
            try:
                stream = await arun_with_retry(
                    session_agent,
                    user_msg.content,
                    stream=True,
                    session_id=thread_id,  # Isolate memory per chat thread
                    max_retries=1,  # Light inner retry; outer loop is primary
                )
                await relay(stream)
                break  # Success – exit retry loop
            except Exception as e:
                if _is_retriable(e) and attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "Retriable error in main() on attempt %d/%d: %s "
                        "– retrying in %.1fs …",
                        attempt + 1,
                        max_retries + 1,
                        e,
                        delay,
                    )
                    await cl.Message(
                        content=(
                            f"The model encountered a transient error. "
                            f"Retrying (attempt {attempt + 2}/{max_retries + 1})..."
                        ),
                        author="assistant",
                    ).send()
                    await asyncio.sleep(delay)
                    continue
                # Non-retriable or final attempt – fall through to error handler
                raise

    except Exception as e:
        # Log error and send user-friendly message
        logger.error(f"Error processing message: {e}", exc_info=True)
        await cl.Message(
            content="Sorry, I encountered an error processing your message. Please try again.",
            author="assistant",
        ).send()
