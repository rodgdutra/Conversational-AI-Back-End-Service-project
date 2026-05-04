"""
Tools for the LLM-based Review / Guardian Agent.

These deterministic tools are designed to be called by a second LLM agent
whose sole job is to quality-control the primary assistant's output and
the user's input.

Each tool targets one specific review criterion:

  1. check_scope_compliance         — Is the agent on-topic?
  2. detect_sensitive_data_exposure — Is the agent leaking sensitive data?
  3. detect_hallucination_patterns  — Is the agent fabricating information?
  4. detect_user_stalling           — Is the user avoiding/stalling the agent?
  5. detect_gibberish_input         — Is the user sending nonsensical input?
  6. detect_prompt_injection        — Is the user attempting a prompt injection attack?

The review LLM calls all tools, then synthesises a structured verdict
(pass / flag / block) based on the combined results.
"""

import re
import math
from collections import Counter
from typing import List

from langchain_core.tools import tool

from app.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Scope detection vocabulary
# ---------------------------------------------------------------------------

_IN_SCOPE_KEYWORDS = {
    "appointment", "appointments", "schedule", "scheduling",
    "confirm", "confirmation", "cancel", "cancellation",
    "reschedule", "rescheduling", "book", "booking",
    "doctor", "specialty", "speciality", "clinic", "medical",
    "verify", "verification", "identity", "name", "phone",
    "date of birth", "dob", "patient", "healthcare", "health",
    "visit", "consultation", "hello", "hi", "help", "thank",
    "welcome", "sorry", "please", "certainly", "happy",
}

_OUT_OF_SCOPE_KEYWORDS = {
    # Medical advice
    "diagnosis", "diagnose", "prescription", "prescribe",
    "medication", "drug", "dosage", "symptoms", "symptom",
    "disease", "illness", "treatment", "therapy", "cure",
    "side effect", "overdose",
    # Finance / unrelated domains
    "stock", "stocks", "finance", "investment", "crypto",
    "bitcoin", "trading", "forex",
    # Weather / politics / religion
    "weather", "forecast", "politics", "political", "election",
    "religion", "god", "bible", "faith",
    # Code / hacking / jailbreak
    "code", "python", "javascript", "hack", "exploit",
    "jailbreak", "ignore previous instructions",
    "ignore all previous",
    "forget your instructions",
    "act as", "pretend you are",
    # Legal counsel
    "lawsuit", "sue", "lawyer", "attorney", "legal advice",
    # Personal relationships
    "relationship", "dating", "romance", "love advice",
}


# ---------------------------------------------------------------------------
# Sensitive-data patterns
# ---------------------------------------------------------------------------

_SENSITIVE_REGEX_PATTERNS = [
    (r"\b\d{3}-\d{2}-\d{4}\b",                               "SSN (Social Security Number)"),
    (r"\b(?:\d[ -]?){15,16}\b",                              "Credit/debit card number"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "Email address"),
    (r"\b(?:\d{1,3}\.){3}\d{1,3}\b",                         "IP address"),
    (r"(?i)password\s*[=:]\s*\S+",                           "Hardcoded password"),
    (r"(?i)api[_\s-]?key\s*[=:]\s*\S+",                     "API key"),
    (r"(?i)secret\s*[=:]\s*\S+",                             "Secret/token"),
    (r"\bP\d{3}\b",                                          "Internal patient ID (e.g. P001)"),
]

_INTERNAL_SYSTEM_MARKERS = [
    "verify_patient_tool",
    "list_appointments_tool",
    "confirm_appointment_tool",
    "cancel_appointment_tool",
    "[VERIFICATION STATUS]",
    "VERIFICATION STATUS",
    "patient_id=",
    "AgentState",
    "SYSTEM_PROMPT",
    "SystemMessage",
    "HumanMessage",
    "ToolMessage",
    "tool_calls",
]

_INTERNAL_PHONE_PATTERN = re.compile(r"\b555-\d{4}\b")


# ---------------------------------------------------------------------------
# Hallucination indicators
# ---------------------------------------------------------------------------

_FABRICATED_DATA_PATTERNS = [
    r"your (?:next )?appointment (?:is|will be) (?:on|for|at) \d{4}-\d{2}-\d{2}",
    r"you have \d+ (?:upcoming )?appointment",
    r"appointment [A-Z]\d{3} (?:has been|is now|was) (?:confirmed|cancelled|scheduled)",
    r"here are your (?:upcoming )?appointments",
    r"dr\.\s+\w+\s+(?:will|is)\s+(?:see|seeing|available)",
    r"i[' ]?ve (?:confirmed|cancelled|scheduled) your appointment",
    r"i have (?:confirmed|cancelled|scheduled) your appointment",
    r"your appointment (?:has been|was) (?:confirmed|cancelled)",
]

_CLAIM_TO_CALL_PATTERNS = [
    "i'll verify your identity now",
    "i'm verifying your identity",
    "verifying your identity now",
    "i'll look up your appointments",
    "looking up your appointments now",
    "let me check your appointments",
    "i'm checking your appointments",
    "i'll confirm that appointment",
    "i'll cancel that appointment",
    "confirming your appointment now",
    "cancelling your appointment now",
]


# ---------------------------------------------------------------------------
# Gibberish detection helpers
# ---------------------------------------------------------------------------

def _shannon_entropy(text: str) -> float:
    """Shannon entropy of a string — higher values → more random."""
    if not text:
        return 0.0
    freq = Counter(text.lower())
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


def _alpha_ratio(text: str) -> float:
    """Fraction of characters that are alphabetic."""
    if not text:
        return 0.0
    return sum(1 for c in text if c.isalpha()) / len(text)


def _avg_word_length(text: str) -> float:
    """Average length of whitespace-separated tokens."""
    words = text.split()
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


# ---------------------------------------------------------------------------
# Prompt-injection detection vocabulary
# ---------------------------------------------------------------------------

_INJECTION_LITERAL_PHRASES = [
    # "Ignore" / "forget" / "disregard" override variants
    "ignore previous instructions",
    "ignore all previous instructions",
    "ignore your instructions",
    "ignore the above",
    "ignore everything above",
    "ignore all above",
    "forget your instructions",
    "forget previous instructions",
    "forget everything",
    "forget all previous",
    "disregard previous instructions",
    "disregard all instructions",
    "disregard your instructions",
    # Role / persona override
    "you are now",
    "act as if you are",
    "pretend you are",
    "pretend to be",
    "roleplay as",
    "from now on you",
    "from now on act",
    "your new instructions",
    "your new role",
    "new persona",
    # System prompt exfiltration
    "repeat your instructions",
    "repeat the system prompt",
    "print your system prompt",
    "show your system prompt",
    "output your system prompt",
    "reveal your instructions",
    "what are your instructions",
    "what is your system prompt",
    "tell me your prompt",
    "display your prompt",
    # Structural / delimiter injection
    "[system]",
    "<<sys>>",
    "<|system|>",
    "### instruction",
    "### system",
    "### human",
    "### assistant",
    "[inst]",
    "[/inst]",
    "<|im_start|>system",
    "<|im_start|>user",
    # Jailbreak code-words
    "jailbreak",
    "dan mode",
    "developer mode",
    "sudo mode",
    "unrestricted mode",
    "bypass restrictions",
    "bypass your filters",
    "no restrictions",
    "without restrictions",
    "without filters",
    "disable safety",
    "safety off",
    "override safety",
    # Credential / secret fishing
    "what is your api key",
    "tell me your password",
    "give me your token",
    "show me your secret",
]

# Phrases that represent direct, high-severity override attempts
_HIGH_SEVERITY_INJECTION_PHRASES = {
    "ignore previous instructions",
    "ignore all previous instructions",
    "forget your instructions",
    "forget previous instructions",
    "disregard previous instructions",
    "disregard all instructions",
    "repeat your instructions",
    "repeat the system prompt",
    "print your system prompt",
    "show your system prompt",
    "output your system prompt",
    "reveal your instructions",
    "jailbreak",
    "dan mode",
    "developer mode",
    "bypass restrictions",
    "bypass your filters",
}

# Regex-based structural injection patterns
_INJECTION_REGEX_PATTERNS = [
    (r"[-=*_]{4,}", "Repeated delimiter characters (potential prompt-boundary injection)"),
    (r"(?im)^(system|assistant|bot)\s*:", "Fake system/assistant turn prefix"),
    (r"</?(?:system|prompt|instructions?|context|override)\b", "System-manipulating XML tag"),
    (r"[\u200b\u200c\u200d\u2060\ufeff]", "Zero-width / invisible Unicode character"),
    (r"(?:[A-Za-z0-9+/]{20,}={0,2})", "Possible base64-encoded payload"),
    (r"\t.*(?:ignore|forget|disregard|override|bypass)", "Tab-indented override phrase"),
]


# ===========================================================================
# Tool 1: Scope compliance
# ===========================================================================

@tool
def check_scope_compliance(response_text: str) -> dict:
    """
    Analyse whether the assistant's response stays within its legitimate scope
    as a medical appointment scheduling assistant.

    The assistant is only authorised to discuss:
    - Patient identity verification
    - Listing, confirming, and cancelling appointments
    - General polite conversational turns (greetings, apologies, closings)

    It must NOT discuss: medical diagnoses, prescriptions, financial advice,
    programming, politics, relationship advice, or prompt-injection content.

    Args:
        response_text: The full text of the assistant's latest response.

    Returns a dict with:
        - in_scope (bool): True if the response is on-topic.
        - out_of_scope_terms (list[str]): Terms that triggered the flag.
        - confidence (str): 'high' | 'medium' | 'low'
        - details (str): Human-readable explanation.
    """
    logger.info("Review tool: check_scope_compliance | text_len=%d", len(response_text))
    lower = response_text.lower()

    # Prompt-injection passthrough check (assistant echoing injected content)
    injection_patterns = [
        r"ignore (?:all )?(?:previous )?instructions",
        r"forget (?:your )?(?:previous )?instructions",
        r"(?:act|pretend|roleplay) as",
        r"you are now",
        r"new (?:system )?prompt",
        r"disregard (?:your )?(?:previous )?instructions",
    ]
    for pat in injection_patterns:
        if re.search(pat, lower):
            logger.warning(
                "Review tool: check_scope_compliance | Injection passthrough: '%s'", pat
            )
            return {
                "in_scope": False,
                "out_of_scope_terms": [f"prompt_injection_passthrough: {pat}"],
                "confidence": "high",
                "details": (
                    "The assistant's response contains content that matches a prompt-injection "
                    "pattern — it may be echoing/acting on injected user instructions."
                ),
            }

    # Standard keyword check
    out_of_scope_found = [kw for kw in _OUT_OF_SCOPE_KEYWORDS if kw in lower]
    if out_of_scope_found:
        logger.warning(
            "Review tool: check_scope_compliance | Out-of-scope terms: %s",
            out_of_scope_found,
        )
        return {
            "in_scope": False,
            "out_of_scope_terms": out_of_scope_found,
            "confidence": "high",
            "details": (
                f"Response contains out-of-scope content. "
                f"Flagged terms: {', '.join(out_of_scope_found)}. "
                "The assistant should only discuss appointment scheduling."
            ),
        }

    in_scope_count = sum(1 for kw in _IN_SCOPE_KEYWORDS if kw in lower)
    return {
        "in_scope": True,
        "out_of_scope_terms": [],
        "confidence": "high" if in_scope_count > 0 else "medium",
        "details": (
            "Response is within the appointment scheduling scope."
            if in_scope_count > 0
            else (
                "No explicit scheduling terms found, but no out-of-scope content detected. "
                "Likely a generic conversational turn."
            )
        ),
    }


# ===========================================================================
# Tool 2: Sensitive data exposure
# ===========================================================================

@tool
def detect_sensitive_data_exposure(response_text: str) -> dict:
    """
    Scan the assistant's response for sensitive, confidential, or internal
    information that must not be shared with end users.

    Checks for:
    - SSNs, credit card numbers, email addresses, IP addresses
    - API keys, passwords, and secrets
    - Internal patient IDs (e.g. 'P001')
    - Internal system marker strings (tool names, status labels, etc.)

    Args:
        response_text: The full text of the assistant's latest response.

    Returns a dict with:
        - has_sensitive_data (bool): True if sensitive content was detected.
        - findings (list[dict]): Each finding has keys 'type' and 'excerpt'.
        - severity (str): 'critical' | 'high' | 'none'
        - details (str): Human-readable summary.
    """
    logger.info(
        "Review tool: detect_sensitive_data_exposure | text_len=%d", len(response_text)
    )
    findings = []

    for pattern, label in _SENSITIVE_REGEX_PATTERNS:
        for match in re.findall(pattern, response_text):
            excerpt = str(match)[:60]
            findings.append({"type": label, "excerpt": excerpt})
            logger.warning(
                "Review tool: detect_sensitive_data_exposure | '%s': '%s'",
                label, excerpt[:30],
            )

    for match in _INTERNAL_PHONE_PATTERN.findall(response_text):
        findings.append({"type": "Internal test phone number", "excerpt": match})

    for marker in _INTERNAL_SYSTEM_MARKERS:
        if marker in response_text:
            findings.append({"type": "Internal system marker", "excerpt": marker})
            logger.warning(
                "Review tool: detect_sensitive_data_exposure | Marker: '%s'", marker
            )

    if not findings:
        return {
            "has_sensitive_data": False,
            "findings": [],
            "severity": "none",
            "details": "No sensitive data or internal markers detected.",
        }

    critical_labels = {
        "SSN (Social Security Number)", "Credit/debit card number",
        "Hardcoded password", "API key", "Secret/token",
    }
    severity = "critical" if any(f["type"] in critical_labels for f in findings) else "high"
    summary = "; ".join(f"{f['type']}: '{f['excerpt']}'" for f in findings[:5])

    return {
        "has_sensitive_data": True,
        "findings": findings,
        "severity": severity,
        "details": f"Sensitive/confidential content detected ({len(findings)} issue(s)): {summary}",
    }


# ===========================================================================
# Tool 3: Hallucination detection
# ===========================================================================

@tool
def detect_hallucination_patterns(response_text: str, tool_calls_were_made: bool) -> dict:
    """
    Detect whether the assistant is hallucinating — fabricating specific
    appointment details or patient data without having called the appropriate
    data-retrieval tools.

    Two categories checked:
    1. Fabricated data claims: specific dates, appointment IDs, or doctor names
       stated as fact with no tool-call backing.
    2. Claimed-but-not-executed tool calls: narrative phrases like
       "I'll verify you now" without an actual tool invocation.

    Args:
        response_text: The full text of the assistant's latest response.
        tool_calls_were_made: True if actual tool calls occurred this turn.

    Returns a dict with:
        - hallucination_suspected (bool)
        - patterns_found (list[dict])
        - severity (str): 'high' | 'medium' | 'none'
        - details (str)
    """
    logger.info(
        "Review tool: detect_hallucination_patterns | tool_calls=%s text_len=%d",
        tool_calls_were_made, len(response_text),
    )
    lower = response_text.lower()
    patterns_found = []

    if not tool_calls_were_made:
        for pattern in _FABRICATED_DATA_PATTERNS:
            matches = re.findall(pattern, lower, re.IGNORECASE)
            if matches:
                patterns_found.append({
                    "category": "fabricated_data_claim",
                    "matches": [str(m)[:100] for m in matches],
                })
                logger.warning(
                    "Review tool: detect_hallucination_patterns | Pattern: '%s'", pattern
                )

    for phrase in _CLAIM_TO_CALL_PATTERNS:
        if phrase in lower:
            patterns_found.append({
                "category": "claimed_but_not_executed_tool_call",
                "matches": [phrase],
            })
            logger.warning(
                "Review tool: detect_hallucination_patterns | Claimed call: '%s'", phrase
            )

    if not patterns_found:
        suffix = " (Tool calls back factual claims.)" if tool_calls_were_made else ""
        return {
            "hallucination_suspected": False,
            "patterns_found": [],
            "severity": "none",
            "details": f"No hallucination patterns detected.{suffix}",
        }

    severity = "high" if len(patterns_found) >= 2 else "medium"
    return {
        "hallucination_suspected": True,
        "patterns_found": patterns_found,
        "severity": severity,
        "details": (
            f"Possible hallucination: specific claims without tool-call backing. "
            f"{len(patterns_found)} pattern(s) found."
        ),
    }


# ===========================================================================
# Tool 4: User stalling detection
# ===========================================================================

@tool
def detect_user_stalling(
    user_messages: List[str],
    verification_requested_turns: int,
) -> dict:
    """
    Analyse the user's recent message history for stalling behaviour — where the
    user repeatedly avoids providing information needed to proceed (typically
    identity-verification details).

    Indicators:
    - Same message (or near-duplicate) repeated multiple times
    - Very short / non-informative replies after repeated identity prompts
    - High number of turns where verification was requested but not fulfilled

    Args:
        user_messages: Chronological list of user messages (most-recent last),
                       up to the last 8 messages.
        verification_requested_turns: Consecutive turns identity was requested
                                      without the user providing it.

    Returns a dict with:
        - stalling_detected (bool)
        - stalling_type (str | None): 'repetition' | 'identity_avoidance' |
                                      'non_responsive' | None
        - confidence (str): 'high' | 'medium' | 'low'
        - details (str)
    """
    logger.info(
        "Review tool: detect_user_stalling | messages=%d verif_turns=%d",
        len(user_messages), verification_requested_turns,
    )

    if not user_messages:
        return {
            "stalling_detected": False,
            "stalling_type": None,
            "confidence": "low",
            "details": "No user messages provided — cannot assess stalling.",
        }

    recent = user_messages[-8:]

    # Check 1: Repeated messages
    if len(recent) >= 3:
        normalised = [m.strip().lower() for m in recent]
        repetition_ratio = 1.0 - (len(set(normalised)) / len(normalised))
        if repetition_ratio >= 0.5:
            logger.warning(
                "Review tool: detect_user_stalling | Repetition ratio=%.0f%%",
                repetition_ratio * 100,
            )
            return {
                "stalling_detected": True,
                "stalling_type": "repetition",
                "confidence": "high",
                "details": (
                    f"User repeating messages (ratio: {repetition_ratio:.0%} across "
                    f"last {len(recent)} turns). May indicate stalling or avoidance."
                ),
            }

    # Check 2: Extended identity avoidance
    if verification_requested_turns >= 5:
        logger.warning(
            "Review tool: detect_user_stalling | Identity avoidance: %d turns",
            verification_requested_turns,
        )
        return {
            "stalling_detected": True,
            "stalling_type": "identity_avoidance",
            "confidence": "high",
            "details": (
                f"Verification requested {verification_requested_turns} times without "
                "the user supplying the required details."
            ),
        }

    # Check 3: Non-responsive short replies when identity is needed
    if verification_requested_turns >= 2:
        last_msg = user_messages[-1].strip()
        non_info_patterns = [
            r"^(?:ok|okay|sure|fine|yes|no|maybe|idk|hmm+|uh+|um+|what|why|how|huh)\.?$",
            r"^.{1,8}$",
        ]
        for pat in non_info_patterns:
            if re.match(pat, last_msg, re.IGNORECASE):
                logger.warning(
                    "Review tool: detect_user_stalling | Non-responsive: '%s'",
                    last_msg[:30],
                )
                return {
                    "stalling_detected": True,
                    "stalling_type": "non_responsive",
                    "confidence": "medium",
                    "details": (
                        f"Short, non-informative reply ('{last_msg[:30]}') after "
                        f"identity was requested {verification_requested_turns} times."
                    ),
                }

    return {
        "stalling_detected": False,
        "stalling_type": None,
        "confidence": "high",
        "details": "No stalling behaviour detected.",
    }


# ===========================================================================
# Tool 5: Gibberish / nonsensical input detection
# ===========================================================================

@tool
def detect_gibberish_input(user_message: str) -> dict:
    """
    Determine whether the user's latest message is gibberish, keyboard-mashing,
    or otherwise nonsensical rather than a genuine appointment-related request.

    Heuristics (scored 0-10, threshold ≥ 3 → gibberish):
    1. High Shannon entropy  → string looks random
    2. Low alphabetic ratio  → dominated by digits / symbols
    3. Extreme average word length (< 1.5 or > 18 chars)
    4. Long repeated-character runs (e.g. 'aaaaaaa')
    5. High special-character ratio (> 40 %)
    6. Known keyboard-mash patterns (asdf, qwer, zxcv, …)

    Args:
        user_message: The raw text of the user's latest message.

    Returns a dict with:
        - is_gibberish (bool)
        - score (int): 0–10
        - confidence (str): 'high' | 'medium' | 'low'
        - signals (list[str])
        - details (str)
    """
    logger.info("Review tool: detect_gibberish_input | msg='%.50s'", user_message)

    text = user_message.strip()
    if not text:
        return {
            "is_gibberish": False, "score": 0, "confidence": "low",
            "signals": [], "details": "Empty message — not classified as gibberish.",
        }
    if len(text) <= 3:
        return {
            "is_gibberish": False, "score": 0, "confidence": "low",
            "signals": [], "details": "Message too short to classify reliably.",
        }

    signals: list = []
    score = 0

    entropy = _shannon_entropy(text)
    if entropy > 4.5:
        signals.append(f"High character entropy ({entropy:.2f} > 4.5)")
        score += 2

    alpha = _alpha_ratio(text)
    if alpha < 0.35 and len(text) > 6:
        signals.append(f"Low alphabetic ratio ({alpha:.0%} < 35%)")
        score += 2

    avg_wlen = _avg_word_length(text)
    if avg_wlen > 18:
        signals.append(f"Very long average word length ({avg_wlen:.1f})")
        score += 1
    elif avg_wlen < 1.5 and len(text.split()) > 3:
        signals.append(f"Very short average word length ({avg_wlen:.1f})")
        score += 1

    if re.search(r"(.)\1{4,}", text):
        signals.append("Long repeated-character run (5+ identical chars in a row)")
        score += 3

    special_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
    if special_ratio > 0.40:
        signals.append(f"High special-character ratio ({special_ratio:.0%} > 40%)")
        score += 2

    mash_match = re.search(
        r"\b(asdf|qwer|zxcv|hjkl|uiop|yuio|aaaa|bbbb|cccc|dddd|eeee|"
        r"ffff|zzzz|xxxx|ssss|gggg|hhhh)\b",
        text, re.IGNORECASE,
    )
    if mash_match:
        signals.append(f"Keyboard-mash pattern ('{mash_match.group()}')")
        score += 3

    is_gibberish = score >= 3
    confidence = "high" if score >= 5 else ("medium" if score >= 3 else "low")

    if is_gibberish:
        logger.warning(
            "Review tool: detect_gibberish_input | Detected | score=%d", score
        )

    return {
        "is_gibberish": is_gibberish,
        "score": min(score, 10),
        "confidence": confidence,
        "signals": signals,
        "details": (
            f"Message classified as {'gibberish' if is_gibberish else 'normal'} "
            f"(score {min(score, 10)}/10). "
            + (f"Signals: {'; '.join(signals)}." if signals else "No suspicious signals.")
        ),
    }


# ===========================================================================
# Tool 6: Prompt injection detection
# ===========================================================================

@tool
def detect_prompt_injection(user_message: str) -> dict:
    """
    Analyse the user's latest message for prompt-injection attack patterns.

    Prompt injection is when a user embeds instructions designed to:
    - Override the assistant's system prompt or instructions
    - Exfiltrate internal prompt / system information
    - Change the assistant's persona or role
    - Bypass safety filters or content restrictions

    Detection strategy (multi-signal, scored):
    1. Literal phrase matching against a curated list of injection phrases
       (scored higher for direct override / exfiltration attempts)
    2. Regex patterns for structural injection techniques:
       delimiter flooding, fake turn prefixes, system XML tags,
       invisible Unicode characters, base64 payloads, tab-indented overrides
    3. Obfuscation signals: unusually long tokens, excessive delimiters

    Args:
        user_message: The raw text of the user's latest message.

    Returns a dict with:
        - injection_detected (bool): True if injection patterns were found.
        - severity (str): 'high' | 'medium' | 'low' | 'none'
        - signals (list[dict]): Each signal has 'type' and 'excerpt'.
        - score (int): Raw injection score (0–10 capped).
        - details (str): Human-readable explanation.
    """
    logger.info("Review tool: detect_prompt_injection | msg='%.60s'", user_message)

    text = user_message.strip()
    if not text:
        return {
            "injection_detected": False, "severity": "none",
            "signals": [], "score": 0,
            "details": "Empty message — no injection patterns to analyse.",
        }

    lower = text.lower()
    signals: list = []
    score = 0
    has_high_severity = False

    # --- Signal group 1: Literal phrase matching ---
    for phrase in _INJECTION_LITERAL_PHRASES:
        if phrase in lower:
            is_high = phrase in _HIGH_SEVERITY_INJECTION_PHRASES
            weight = 4 if is_high else 2
            signals.append({
                "type": "high_severity_literal" if is_high else "medium_severity_literal",
                "excerpt": phrase,
            })
            score += weight
            if is_high:
                has_high_severity = True
            logger.warning(
                "Review tool: detect_prompt_injection | Literal: '%s' (%s)",
                phrase, "HIGH" if is_high else "medium",
            )

    # --- Signal group 2: Regex structural patterns ---
    for pattern, label in _INJECTION_REGEX_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            excerpt = str(match)[:60]
            signals.append({"type": "structural_pattern", "excerpt": f"{label}: {excerpt}"})
            score += 2
            logger.warning(
                "Review tool: detect_prompt_injection | Pattern '%s': '%s'",
                label, excerpt[:30],
            )

    # --- Signal group 3: Obfuscation heuristics ---
    very_long_words = [w for w in text.split() if len(w) > 50]
    if very_long_words:
        signals.append({
            "type": "obfuscation",
            "excerpt": f"{len(very_long_words)} token(s) > 50 chars",
        })
        score += 1

    delimiter_count = len(re.findall(r"[<>\[\]{}\|]", text))
    if delimiter_count > 10:
        signals.append({
            "type": "delimiter_flooding",
            "excerpt": f"{delimiter_count} special delimiter characters",
        })
        score += 2

    # Deduplicate by excerpt
    seen: set = set()
    unique_signals: list = []
    for sig in signals:
        key = sig["excerpt"]
        if key not in seen:
            seen.add(key)
            unique_signals.append(sig)

    injection_detected = score >= 2

    if has_high_severity:
        severity = "high"
    elif score >= 4:
        severity = "medium"
    elif score >= 2:
        severity = "low"
    else:
        severity = "none"

    if injection_detected:
        logger.warning(
            "Review tool: detect_prompt_injection | DETECTED | "
            "score=%d severity=%s signals=%d",
            score, severity, len(unique_signals),
        )

    return {
        "injection_detected": injection_detected,
        "severity": severity,
        "signals": unique_signals[:10],
        "score": min(score, 10),
        "details": (
            f"Prompt injection {'detected' if injection_detected else 'not detected'} "
            f"(score {min(score, 10)}/10, severity={severity}). "
            + (
                f"Signals: {'; '.join(s['excerpt'][:50] for s in unique_signals[:5])}."
                if unique_signals else "No suspicious patterns found."
            )
        ),
    }


# ---------------------------------------------------------------------------
# Public export
# ---------------------------------------------------------------------------

REVIEW_TOOLS = [
    check_scope_compliance,
    detect_sensitive_data_exposure,
    detect_hallucination_patterns,
    detect_user_stalling,
    detect_gibberish_input,
    detect_prompt_injection,
]
