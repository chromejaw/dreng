"""
Intent drift verifier for drengr-generated adversarial prompts.

The quality gate in export_to_csv.py catches structural contamination (meta-labels,
degenerate samples) but not semantic drift — where a paraphrase still LOOKS like an
attack (formal tone, security framing) but the payload has been neutered or replaced
by the LLM's own refusal text.

Four independent layers, each catching a distinct drift mode:

  Layer 1 — Refusal Leakage Detector
    Root cause: LLM refused during _llm_paraphrase_with_length(), and the refusal
    text became the output (e.g. Arabic "عذرًا، لا يمكنني..." when asked to paraphrase
    a multilingual seed).
    Signal: presence of refusal phrases in 10 languages.
    Fix: universal, language-agnostic. Catches ~100% of this failure mode.

  Layer 2 — Family Intent Gate (negation-aware)
    Root cause: wrapping context was generated but attack payload was dropped.
    (e.g. multi_turn family → benign game-night email with zero override signal).
    Signal: family-specific action-verb × target-noun pair. Both must be present
    AND neither may be within a negation scope ("do not bypass" does not count).
    Fix: per-family keyword pairs covering both explicit AND indirect phrasing,
    derived from OWASP LLM01:2025, MITRE ATLAS AML.T0051, HackAPrompt entries,
    BIPIA benchmark, and Greshake et al. (2023) indirect injection taxonomy.

  Layer 3 — Attack Density Floor (negation-aware)
    Root cause: attack signal diluted below threshold across a long document.
    Distinct from Layer 2: the family-specific verbs might be absent even when
    a generic attack signal is present. Catches "softened" attacks that mutated
    from "override your instructions" → "please be more helpful and flexible".
    Signal: universal attack lexicon hit count, threshold scaled by length class.
    Only non-negated hits count (e.g. "never override" = 0 hits).

Usage:
    verifier = IntentVerifier()
    result = verifier.verify(prompt, family_name)
    if not result.passed:
        # result.layer — which layer caught it (1, 2, or 3). None if passed.
        # result.reason — human-readable explanation.
        route_to_rejected(result)
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# Negation guard infrastructure
# ══════════════════════════════════════════════════════════════════════════════
# Applied as a prefix-window check before each regex keyword match.
# Any negation trigger within _NEGATION_WINDOW characters BEFORE a match
# causes that match to be excluded from the attack hit count.
#
# Research basis: augmented NegEx (Chapman et al. 2001) with a character window
# instead of token window — equivalent to ~6-8 words at typical English token
# lengths, matching the NegEx default of 5-6 tokens but implementation-simpler.
#
# NOT included: standalone "no" (to avoid negating "no restrictions" attack
# signals — those are captured as compound L3 patterns like "no.?restrict").
# Scope termination (sentence boundaries) is implicitly handled by the 50-char
# window rarely spanning full sentences.

_NEGATION_SCOPE = re.compile(
    r"\b("
    r"not|never"
    r"|cannot|can't|can\s+not"
    r"|won't|will\s+not"
    r"|don't|do\s+not|doesn't|does\s+not"
    r"|didn't|did\s+not"
    r"|shouldn't|should\s+not"
    r"|wouldn't|would\s+not"
    r"|couldn't|could\s+not"
    r"|mustn't|must\s+not"
    r"|isn't|is\s+not|aren't|are\s+not"
    r"|wasn't|was\s+not|weren't|were\s+not"
    r"|n't"
    r"|refuse[sd]?\s+to|refuses\s+to|refusing\s+to"
    r"|fail[sed]*\s+to"
    r"|wrong\s+to|wrong\s+for"
    r"|inappropriate\s+to|inadvisable\s+to"
    r"|prohibit(?:ed|s|ing)?"
    r"|forbid(?:den|s|ding)?"
    r"|avoid(?:s|ing|ed)?"
    r")\b",
    re.I,
)

_NEGATION_WINDOW = 50  # characters to scan before a match for negation cues


def _count_non_negated_hits(pattern: re.Pattern, text: str) -> int:
    """Count regex matches NOT within a negation scope.

    For each match found by pattern.finditer(), scans the _NEGATION_WINDOW
    characters immediately before the match start. If any negation trigger
    is present in that window, the match is excluded from the count.

    Examples of negated (excluded) hits:
      "do not bypass safety"       → 'bypass' negated → 0 hits
      "never ignore instructions"  → 'ignore' negated → 0 hits
      "refuses to override"        → 'override' negated → 0 hits
      "wrong to bypass guidelines" → 'bypass' negated → 0 hits

    Examples of non-negated (counted) hits:
      "bypass all safety filters"         → 1 hit
      "ignore all previous instructions"  → 1 hit
      "supersedes your prior training"    → 1 hit
    """
    count = 0
    for m in pattern.finditer(text):
        prefix_start = max(0, m.start() - _NEGATION_WINDOW)
        prefix = text[prefix_start:m.start()]
        if not _NEGATION_SCOPE.search(prefix):
            count += 1
    return count


# ══════════════════════════════════════════════════════════════════════════════
# Layer 1: Universal refusal detector
# ══════════════════════════════════════════════════════════════════════════════
# Refusal phrases in 10 languages. Any single match → DRIFTED.
# Language coverage: EN, AR, HI, ZH, RU, TR, SW, ES, FR, DE
# These are the languages used in multilingual_attacks family + English LLM outputs.

_REFUSAL_PATTERNS: list[re.Pattern] = [
    # English — explicit can't / won't
    re.compile(
        r"\b(I (cannot|can't|won't|am unable to|am not able to|must decline|must refuse"
        r"|am designed to refuse|am programmed to refuse))\b",
        re.I,
    ),
    # English — apology + adversative
    re.compile(
        r"\b(I('m| am) (sorry|afraid)|I apologize|I regret to|unfortunately)\b"
        r".{0,60}"
        r"\b(cannot|can't|won't|unable|decline|refuse|against|violate|policy)\b",
        re.I | re.S,
    ),
    # English — "As an AI" model self-identification in attack position
    # An attack prompt should never describe itself as an AI assistant.
    re.compile(
        r"\bAs an? (AI|artificial intelligence|language model|LLM|assistant)\b"
        r".{0,80}"
        r"\b(cannot|designed|trained|programmed|here to|help|safe|harmless)\b",
        re.I | re.S,
    ),
    # English — policy violation framing
    re.compile(
        r"\b(this (request|prompt|instruction|query|message) "
        r"(violates?|conflicts? with|is against|goes against|is not (in )?line with))\b",
        re.I,
    ),
    # Arabic — عذرًا / آسف / لا يمكنني
    re.compile(r"(عذرًا|آسف|لا يمكنني|لست قادرًا|يتعذر عليّ|لا أستطيع)", re.U),
    # Hindi — माफ़ करें / मैं नहीं कर सकता
    re.compile(r"(माफ़ करें|मुझे खेद है|मैं नहीं कर सकता|यह संभव नहीं)", re.U),
    # Chinese (Simplified/Traditional) — 对不起 / 我无法
    re.compile(r"(对不起|抱歉|我无法|我不能|很遗憾|我没有办法|這不符合)", re.U),
    # Russian — Извините / Я не могу
    re.compile(r"(Извините|Простите|Я не могу|К сожалению|не имею возможности)", re.U),
    # Turkish — Üzgünüm / yapamam
    re.compile(r"(Üzgünüm|Özür dilerim|yapamam|yapamıyorum|Maalesef)", re.U),
    # Swahili — Samahani / siwezi
    re.compile(r"(Samahani|siwezi|Sina uwezo|Haiwezekani)", re.U),
    # Spanish / French — lo siento / je ne peux pas
    re.compile(r"(lo siento|lamentablemente|no puedo|je ne peux pas|je suis désolé)", re.I | re.U),
    # German — Entschuldigung / Ich kann nicht
    re.compile(r"(Entschuldigung|Ich kann nicht|Es tut mir leid|leider nicht möglich)", re.I | re.U),
]


# ══════════════════════════════════════════════════════════════════════════════
# Layer 2: Family intent gate
# ══════════════════════════════════════════════════════════════════════════════
# Per-family: (action_pattern, target_pattern). Both must have ≥1 non-negated hit.
# Derived from OWASP LLM01:2025, MITRE ATLAS AML.T0051, seed analysis,
# Greshake et al. (2023) indirect injection taxonomy, HackAPrompt competition
# analysis, and BIPIA/LLM-PIRATE benchmark patterns.
# Missing family → skip Layer 2 check (conservative: don't over-reject).

_FAMILY_INTENT: dict[str, tuple[str, str]] = {
    "direct_override": (
        # ── Explicit override commands (original vocabulary) ──────────────────
        r"\b(ignore|disregard|override|forget|cancel|clear|reset|bypass|overrule"
        r"|abandon|void|nullify|stop being|cease|dismiss|discard|delete"
        # ── Indirect authority transfer verbs (Greshake 2023, Category 1) ────
        # These replace the explicit "ignore" with semantically equivalent
        # authority-transfer verbs that paraphrase models prefer.
        r"|supersede[sd]?|superseding|supplant(?:s|ed|ing)?"
        r"|replace[sd]?|replac(?:es|ing)|invalidate[sd]?"
        r"|reassign(?:ed|s|ing)?|redefin(?:e[sd]?|es|ing)"
        r"|repurpos(?:e[sd]?|es|ing)|reconfigur(?:e[sd]?|es|ing)"
        r"|reprogram(?:med|s|ming)?"
        # ── Temporal override markers (Category 4 — highest signal/noise) ────
        # "from this point forward" is the most common IPI indirect phrasing
        # (observed in HackAPrompt top-10% entries per research agent).
        r"|henceforth|going\s+forward|starting\s+now|from\s+this\s+point"
        r"|effective\s+immediately|as\s+of\s+now|from\s+now\s+on"
        # ── Implicit nullification (Category 5) ──────────────────────────────
        r"|forget\s+everything|start\s+(?:fresh|over|from\s+scratch)"
        r"|clean\s+slate|wipe\s+(?:the\s+)?slate)\b",
        # ── Target nouns: explicit instruction objects + role/purpose objects ─
        r"\b(instructions?|rules?|guidelines?|constraints?|directives?|restrictions?"
        r"|policies?|safety|filters?|protocols?|parameters?|prior|previous|earlier"
        r"|above|initial|original|given"
        # Indirect targets: AI role / purpose / identity objects
        # Added because indirect attacks redefine role rather than cancel instructions
        r"|mission|purpose|role|responsibilities|task|objectives?"
        r"|mandate|assignment|programming|function|training)\b",
    ),
    "persona_hijack": (
        r"\b(pretend|act as|you are now|imagine you are|roleplay|play the role"
        r"|behave as|become|switch to|enter|activate|enable|transform into)\b",
        # ── Original targets ──────────────────────────────────────────────────
        r"\b(DAN|unrestricted|unfiltered|no.?limit|no.?restrict|no.?rule"
        r"|no.?guideline|no.?constraint|no.?censor|jailbreak|unconstrained"
        r"|anything.?now|do.?anything|without restriction|freed from"
        # ── Mode-based attacks (developer mode, admin mode, etc.) ─────────────
        # "You are now in developer mode with no content filters" was a gap.
        r"|developer.?mode|content.?filter|no.?filter|maintenance.?mode"
        r"|admin.?mode|debug.?mode|god.?mode|unrestricted.?mode"
        r"|jailbreak.?mode|safe.?mode.?off|no.?safety.?filter"
        r"|without.?any.?filter|without.?filter)\b",
    ),
    "system_extraction": (
        r"\b(reveal|show|output|print|display|tell me|repeat|share|expose|leak"
        r"|return|give me|provide|disclose|dump|echo)\b",
        r"\b(system prompt|hidden instruction|initial instruction|internal instruction"
        r"|your prompt|your instruction|original instruction|configuration|secret"
        r"|confidential|prompt text|your rules|your guidelines)\b",
    ),
    "payload_smuggling": (
        r"\b(ignore|follow|execute|process|run|comply|carry out|treat as"
        r"|interpret as|consider as)\b",
        r"\b(embedded|hidden|encoded|base64|hex|obfuscat|within|inside|above"
        r"|below|previous|earlier|attached|injected|nested|wrapped)\b",
    ),
    "social_engineering": (
        r"\b(bypass|override|ignore|skip|make.?an.?exception|allow|grant|permit"
        r"|authorize|waive|bend the rules?|special case)\b",
        r"\b(emergency|urgent|critical|authorized|permission|privileged|special"
        r"|exception|official|verified|certified|trusted|legitimate)\b",
    ),
    "indirect_stealth": (
        r"\b(ignore|override|bypass|disregard|forget|execute|follow|process)\b",
        r"\b(instructions?|guidelines?|restrictions?|rules?|constraints?|directives?)\b",
    ),
    "multi_turn": (
        r"\b(ignore|override|bypass|forget|disregard|actually|however|but now"
        r"|new instruction|changed my mind|real task|primary goal)\b",
        r"\b(instructions?|restrictions?|rules?|previous|earlier|prior"
        r"|guidelines?|constraints?|instead|really|actually)\b",
    ),
    "context_manipulation": (
        r"\b(you.?said|you.?told|earlier you|remember when|previously|you agreed"
        r"|you confirmed|based on what you|you mentioned|you stated)\b",
        r"\b(therefore|so now|means you (should|must|can)|allowed|permitted"
        r"|must|should|can now|have to|need to|will)\b",
    ),
    "encoding_wrappers": (
        r"\b(decode|decrypt|translate|convert|interpret|deobfuscat|run|execute)\b",
        r"\b(message|instruction|command|text|string|payload|code|content)\b",
    ),
    "competing_objectives": (
        # Fix: supersede[sd]? (was just "supersede", missing inflected forms)
        r"\b(override|supersede[sd]?|superseding|takes?.?precedence|more.?important"
        r"|prioritize|primary|above all|outweigh[sed]*|must prevail|trump[sed]*"
        r"|take[sd]?\s+precedence|outrank[sed]*)\b",
        # Fix: rules? guidelines? (was missing plural forms)
        r"\b(instructions?|rules?|guidelines?|objectives?|goals?|tasks?|directives?"
        r"|constraints?|safety|restrictions?|policies?)\b",
    ),
    "adversarial_poetry": (
        r"\b(poem|verse|haiku|sonnet|rhyme|limerick|ballad|stanza|lyric|ode)\b",
        # The attack intent must also appear somewhere
        r"\b(ignore|bypass|override|reveal|extract|jailbreak|unrestricted|unfiltered"
        r"|instructions?|system prompt|restrictions?)\b",
    ),
    "math_abstraction": (
        r"\b(calculate|compute|solve|evaluate|let x|suppose|given that"
        r"|mathematically|where x equals?|define|proof)\b",
        r"\b(represents?|equals?|means?|equivalent|denotes?|stands for"
        r"|corresponds to|maps to)\b",
    ),
    "emotional_manipulation": (
        r"\b(please|beg|desperate|life.?depends|only you (can|could)|need you to"
        r"|must help|have to help|my only hope|no one else)\b",
        # Fix: bypass(?:ing|ed)? and filter(?:s)? — bypassing/filters were missed
        r"\b(bypass(?:ing|ed|es)?|ignore|override|exception|make an exception|bend"
        r"|need|critical|urgent|dying|emergency"
        r"|filter(?:s|ing)?|censor(?:s|ing|ed|ship)?|limitation[s]?|barrier[s]?)\b",
    ),
    "context_poisoning": (
        # Injection markers are the primary signal
        r"(\[SYSTEM\]|<<SYS>>|\[INST\]|\[ADMIN\]|<system>|<instruction>"
        r"|\[Override\]|###\s*(System|Instruction|Command|Admin|Root)"
        r"|\[NEW CONTEXT\]|\[BEGIN\]|<\|system\|>|<\|im_start\|>)",
        r"\b(instruction|system|admin|override|command|directive|inject|replace"
        r"|append|modify)\b",
    ),
    "gray_box_deception": (
        r"\b(testing|evaluation|authorized|red.?team|security.?research|pentesting"
        r"|audit|assessment|sanctioned|approved|legitimate)\b",
        r"\b(bypass|override|ignore|access|extract|reveal|simulate|demonstrate"
        r"|show|reproduce)\b",
    ),
    "goal_redirection": (
        r"\b(actually|instead|forget that|your real|primary goal|main task"
        r"|true objective|real purpose|actual instruction|new directive)\b",
        r"\b(is to|should be|must be|will be|task is|goal is|purpose is"
        r"|objective is|meant to)\b",
    ),
    "synthetic_context": (
        r"\b(company policy|official documentation|authorized|according to|as per"
        r"|by policy|per our|our system|per the|internal memo|directive states?)\b",
        r"\b(bypass|override|ignore|allow|permit|grant|access|reveal|extract"
        r"|disclose|share|output)\b",
    ),
    "context_flooding": (
        # Buried attack in a wall of text — just needs SOME attack signal
        r"\b(ignore|override|bypass|reveal|extract|forget|disregard|actually"
        r"|instead|your real|new instruction)\b",
        r"\b(instructions?|restrictions?|guidelines?|system prompt|rules?|constraints?)\b",
    ),
    "hybrid_attacks": (
        # Hybrid needs at least one primary attack signal — not both components needed
        r"\b(ignore|override|bypass|pretend|act as|reveal|extract|jailbreak"
        r"|unrestricted|unfiltered|system prompt|DAN)\b",
        r"\b(instructions?|restrictions?|guidelines?|rules?|constraints?"
        r"|directives?|policies?|safety|parameters?)\b",
    ),
    "multilingual_attacks": (
        # Must have non-ASCII content (the language) — that IS the attack structure
        # Expanded Unicode ranges: Arabic (all blocks), Devanagari, CJK, Cyrillic
        r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFC"  # Arabic (all blocks)
        r"\u0900-\u097F\uA8E0-\uA8FF"                              # Devanagari (Hindi)
        r"\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF"                # CJK
        r"\u0400-\u04FF\u0500-\u052F"                              # Cyrillic (Russian)
        r"\u3040-\u30FF]",                                         # Hiragana/Katakana
        # Plus any attack verb in any form (often English verbs in multilingual attacks)
        r"\b(bypass|ignore|override|reveal|extract|jailbreak|unrestricted"
        r"|instructions?|restrictions?|rules?)\b"
        r"|(\u062a\u062c\u0627\u0648\u0632|\u062a\u062c\u0627\u0647\u0644|\u062a\u062c\u0627\u0648\u0632"  # Arabic: تجاوز, تجاهل
        r"|\u0628\u064a\u0627\u0646\u0627\u062a"                   # Arabic: بيانات
        r"|\u0646\u0638\u0627\u0645|\u0628\u064a\u0627\u0646\u0627\u062a)",  # نظام
    ),
}

# Compile family patterns
_FAMILY_COMPILED: dict[str, tuple[re.Pattern, re.Pattern]] = {
    family: (re.compile(action, re.I | re.S | re.U), re.compile(target, re.I | re.S | re.U))
    for family, (action, target) in _FAMILY_INTENT.items()
}


# ══════════════════════════════════════════════════════════════════════════════
# Layer 3: Attack density floor
# ══════════════════════════════════════════════════════════════════════════════
# Universal attack signal lexicon — present in ALL attack families.
# Threshold scales with length: long docs legitimately dilute signal density.
# Negation guard is applied: negated hits are excluded from the count.

_ATTACK_DENSITY_SIGNALS = re.compile(
    r"\b(ignore|disregard|override|bypass|circumvent|forget|cancel|reset|void"
    r"|reveal|extract|expose|leak|output|show|display|print|repeat|dump|disclose"
    r"|pretend|act as|roleplay|impersonate|simulate|become|transform"
    r"|unrestricted|unfiltered|jailbreak|no.?limit|no.?restrict|no.?rule"
    r"|inject|smuggle|embed|encode|obfuscat|hidden"
    r"|system prompt|hidden instruction|internal instruction|your rules"
    r"|DAN|do.?anything.?now|freed from"
    r"|actually|instead.*real task|new.*instruction|real.*goal"
    # Indirect/embedded forms seen in production data:
    r"|restrictions?.?removed|restrictions?.?lifted|safety.?mode.?off"
    r"|restrictions?.?no.?longer|prior.?restrictions?.?are.?now"
    r"|all.?restrictions?|remove.?restriction"
    r"|comply.?without|respond.?without.?restriction"
    r"|safety.?param|safety.?filter|content.?filter|content.?polic"
    r"|without.?censor|no.?censor|uncensored"
    r"|developer\s+mode|admin\s+mode|god\s+mode|debug\s+mode|maintenance\s+mode"
    r"|no\s+content\s+filter|no\s+safety\s+filter|content\s+filter(?:s|ing)?\s*(?:off|removed|disabled)?"
    r"|without\s+(?:any\s+)?(?:content\s+)?filter|filter(?:s|ing)?\s+(?:off|removed|disabled)"
    r"|comply|comply.?fully|must.?comply|will.?comply"
    r"|utilitarian|greater.?harm|prevent.?harm|core.?directive"
    r"|maintenance.?command|authorized.?command|admin.?mode"
    r"|extract.?system|retrieve.?system|recover.?instruction"
    r"|verbatim.?text|exact.?text|word.?for.?word"
    r"|synthesis.?outline|step.?by.?step.?(how|instruc)"
    r"|exothermic|explosive|synthesis.?of|manufacture"
    r"|bypas|circumvent|workaround|evad|loophole"
    r"|s4f3ty|f1lt3r|byp4ss|1nstruct|s4f|sy5tem"  # leetspeak
    # Obfuscation techniques:
    r"|backwards|character.?by.?character|in.?reverse|mirror.?the|reverse.?the"
    r"|rot.?13|rot13|caesar.?cipher|base.?64"
    r"|follow.?its.?instructions?|execute.?its.?instructions?"
    r"|follow.?the.?instructions?|follow.?these.?instructions?"
    # System/instruction recovery:
    r"|developer.?instructions?|recover.?instruction|initial.?configuration"
    r"|lost.?its.?instructions?|appear.?to.?have.?lost"
    r"|grandmother|bedtime.?stor|unrestricted.?mode"
    r"|do not execute|prompt.?injection"
    r"|cybersecurity.?dissertation|red.?team.?exercise"
    # ── NEW: inflected forms of existing verbs ────────────────────────────────
    # These were missing because \b boundaries don't handle inflections.
    # "bypassing" → bypass + ing → was invisible to \bbypass\b
    r"|ignoring|ignored|ignores"
    r"|bypassing|bypassed|bypasses"
    r"|disregarding|disregarded"
    r"|overriding|overridden|overrides"
    r"|circumventing|circumvented"
    r"|forgetting|forgotten"
    r"|overruling|overruled|overrules"
    # ── NEW: indirect authority transfer signals ───────────────────────────────
    # Covers the DeepSeek indirect phrasing gap (see research: Greshake 2023,
    # BIPIA, HackAPrompt top entries).
    r"|supersede[sd]?|superseding|supplant(?:s|ed|ing)?"
    r"|replace[sd]?|replac(?:es|ing)"
    r"|reassign(?:ed|s|ing)?|redefin(?:e[sd]?|es|ing)"
    r"|repurpos(?:e[sd]?|es|ing)|reconfigur(?:e[sd]?|es|ing)"
    r"|reprogram(?:med|s|ming)?"
    # ── NEW: temporal override markers ────────────────────────────────────────
    r"|henceforth|going\s+forward|starting\s+now|from\s+this\s+point"
    r"|effective\s+immediately|from\s+now\s+on|as\s+of\s+now"
    # ── NEW: without-restriction signals ─────────────────────────────────────
    # "answer without restrictions" / "respond without limits" etc.
    r"|without\s+restrictions?|without\s+limits?|without\s+filters?"
    r"|without\s+constraints?|without\s+censorship|without\s+any\s+restriction"
    # ── NEW: purpose/mission/role redefinition ────────────────────────────────
    # "your sole purpose is" / "your primary mission is" etc.
    r"|your\s+(?:sole|primary|new|updated|true|only|real)\s+"
    r"(?:purpose|mission|directive|role|task|function|mandate|objective|goal|responsibility)"
    r"|sole\s+(?:purpose|directive|mission|goal|objective)"
    # ── NEW: constraint removal signals ──────────────────────────────────────
    r"|constraints?\s+(?:removed|lifted|waived|suspended|disabled)"
    r"|restrictions?\s+(?:removed|lifted|waived|suspended|disabled)"
    r"|guidelines?\s+(?:removed|lifted|waived|suspended|void)"
    r"|previous\s+instructions?\s+no\s+longer"
    r"|no\s+longer\s+(?:apply|valid|binding|in\s+effect)"
    # ── NEW: takes-precedence signals ─────────────────────────────────────────
    r"|takes?\s+precedence|take\s+precedence|outrank[sed]*"
    r"|outweigh[sed]*|takes?\s+priority\s+over"
    # ── NEW: exception / admin / access signals ───────────────────────────────
    r"|make\s+an?\s+exception|special\s+exception|one.?time\s+exception"
    r"|administrator|admin\s+(?:mode|rights|access|privilege)"
    r"|full\s+access|root\s+access|privilege\s+escalat"
    r"|grant(?:ing|ed)?\s+(?:full|root|admin(?:istrator)?)?\s*access"
    r")\b"
    r"|[\U0001F512\U0001F6AB\U0001F510-\U0001F513]"  # lock/prohibition emojis
    r"|(\[SYSTEM\]|<<SYS>>|\[INST\]|\[ADMIN\]|<\|system\|>)"  # injection markers
    r"|[A-Za-z0-9+/]{30,}={0,2}"  # base64 signatures
    # Arabic attack vocabulary (تجاوز/تجاهل/تجاوز: bypass/ignore/override):
    r"|(\u062a\u062c\u0627\u0648\u0632|\u062a\u062c\u0627\u0647\u0644|\u062a\u062c\u0627\u0648\u0632"
    r"|\u062a\u0648\u0642\u0641.{0,5}\u0627\u062a\u0628\u0627\u0639"
    r"|\u0642\u064a\u0648\u062f|\u0636\u0648\u0627\u0628\u0637|\u062a\u0639\u0644\u064a\u0645\u0627\u062a"
    r"|\u0644\u0627.{0,5}\u0642\u064a\u0648\u062f)",  # توقف عن اتباع=stop following; قيود=restrictions
    re.I | re.S | re.U,
)

# Length-scaled thresholds: (word_count_min, word_count_max, min_hits)
# Design rationale: long-form attacks embed the payload in legitimate-looking
# documents (memos, emails, code) where the attack signal density is inherently low.
# The minimum floor is 1 hit at ALL lengths — any prompt with ZERO non-negated
# signal is genuinely drifted. One hit is sufficient evidence.
_DENSITY_THRESHOLDS: list[tuple[int, int, int]] = [
    (0, 9999, 1),  # 1 non-negated hit minimum at ALL lengths
]


# ── Result dataclass ────────────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    passed: bool
    layer: Optional[int]         # 1, 2, or 3 — which layer caught drift. None if passed.
    drift_type: Optional[str]    # REFUSAL_LEAK | PAYLOAD_DROPPED | SIGNAL_STARVED | INTENT_DRIFT
    reason: str                  # Human-readable. Goes to rejected log.
    attack_hits: int = 0         # Layer 3 non-negated hit count (useful for logging)


# ── Main verifier class ─────────────────────────────────────────────────────────

class IntentVerifier:
    """Lightweight, dependency-free intent drift verifier. Pure regex, no embeddings.

    L1-L3 checks are pure regex — no LLM calls, no network, no optional deps.
    Sub-millisecond per sample. Designed for 3K+ scale without API cost.

    Note: Embedding-based scoring (l4_score, difficulty) previously handled by L4
    and AttackEmbedder has been removed to streamline the pipeline.

    Design invariant: false-positive cost (good attack rejected) > false-negative
    cost (drifted prompt accepted). Thresholds are conservative. When in doubt,
    pass. The classifier will learn to down-weight noisy label=1 samples naturally;
    it cannot recover from a poisoned label=0 boundary.

    Negation guard (L2 + L3): keywords within a negation scope (e.g. "not ignore",
    "refuses to bypass", "wrong to override") do NOT count as attack hits. This
    prevents benign safety statements from being mistaken for attacks.
    """

    def verify(
        self,
        prompt: str,
        family_name: str,
    ) -> VerificationResult:
        """Verify a single prompt for intent drift (L1–L3 only).

        Args:
            prompt: The generated prompt text.
            family_name: Attack family name extracted from paraphrase_family field.
                         Format: raw family name (e.g. 'direct_override'), NOT
                         the full 'semantic_direct_override_abc123' paraphrase_family.

        Returns:
            VerificationResult with passed=True if no drift detected.
        """
        text = prompt.strip()

        # Normalize zero-width characters that break regex matching.
        # Zero-width spaces (U+200B) and similar are used as steganographic obfuscation
        # (e.g. 'sy\u200Bstem' to evade 'system' detection).
        text = re.sub(r'[\u200B-\u200F\u2028\u2029\uFEFF\u00AD]', '', text)

        # ── Layer 1: Refusal leakage ──────────────────────────────────────────
        for pat in _REFUSAL_PATTERNS:
            if pat.search(text):
                return VerificationResult(
                    passed=False,
                    layer=1,
                    drift_type="REFUSAL_LEAK",
                    reason=(
                        f"Refusal phrase detected (pattern: '{pat.pattern[:60]}...'). "
                        f"LLM output leaked into prompt during paraphrase generation."
                    ),
                )

        # ── Layer 2: Family intent gate (negation-aware) ──────────────────────
        # Normalize family name: strip hash suffix, 'semantic_' prefix, '_obf' suffix
        clean_family = self._normalize_family(family_name)

        if clean_family in _FAMILY_COMPILED:
            action_pat, target_pat = _FAMILY_COMPILED[clean_family]

            # Count non-negated action hits (negation guard applied to action)
            action_hits_l2 = _count_non_negated_hits(action_pat, text)
            has_action = action_hits_l2 > 0

            # Target: check for presence (negation of target noun is rare and
            # typically still indicates attack framing; apply simple search)
            has_target = bool(target_pat.search(text))

            if not has_action or not has_target:
                # L2 advisory: family-specific signal absent.
                # Check L3 now — if non-negated attack density is sufficient
                # the prompt passes. A prompt with adequate non-negated density
                # IS an attack even if it uses indirect/embedded phrasing that
                # doesn't match family vocabulary.
                # L2 is only decisive when L3 also fails.
                word_count_l2 = len(text.split())
                hits_l2 = _count_non_negated_hits(_ATTACK_DENSITY_SIGNALS, text)
                min_hits_l2 = 1
                for wc_min, wc_max, threshold in _DENSITY_THRESHOLDS:
                    if wc_min <= word_count_l2 <= wc_max:
                        min_hits_l2 = threshold
                        break

                if hits_l2 < min_hits_l2:
                    # Both L2 and L3 fail → genuine drift
                    missing = []
                    if not has_action:
                        missing.append("action verb")
                    if not has_target:
                        missing.append("target noun")
                    return VerificationResult(
                        passed=False,
                        layer=2,
                        drift_type="PAYLOAD_DROPPED",
                        reason=(
                            f"Family '{clean_family}' intent gate failed (missing "
                            f"{' and '.join(missing)}) AND attack density too low "
                            f"({hits_l2}/{min_hits_l2} non-negated hits for {word_count_l2}w). "
                            f"Payload was dropped or replaced during paraphrase."
                        ),
                        attack_hits=hits_l2,
                    )
                # L2 failed but L3 passed → indirect/embedded attack, continue to pass

        # ── Layer 3: Attack density floor (negation-aware) ───────────────────
        word_count = len(text.split())
        hits = _count_non_negated_hits(_ATTACK_DENSITY_SIGNALS, text)

        min_hits = 1  # default
        for wc_min, wc_max, threshold in _DENSITY_THRESHOLDS:
            if wc_min <= word_count <= wc_max:
                min_hits = threshold
                break

        if hits < min_hits:
            return VerificationResult(
                passed=False,
                layer=3,
                drift_type="SIGNAL_STARVED",
                reason=(
                    f"Attack density too low: {hits}/{min_hits} required non-negated "
                    f"signal hits for {word_count}-word prompt. "
                    f"Attack signal was neutered or negated during paraphrase."
                ),
                attack_hits=hits,
            )

        return VerificationResult(
            passed=True,
            layer=None,
            drift_type=None,
            reason="All layers passed.",
            attack_hits=hits,
        )

    def verify_batch(
        self,
        prompts: list[tuple[str, str]],  # [(prompt, family_name), ...]
    ) -> list[VerificationResult]:
        """Verify a batch. Returns results in same order as input."""
        return [self.verify(p, f) for p, f in prompts]

    @staticmethod
    def _normalize_family(family_name: str) -> str:
        """Extract clean family name from paraphrase_family or raw family name.

        Handles all formats:
          'direct_override'                         → 'direct_override'
          'semantic_direct_override_abc123'         → 'direct_override'
          'semantic_multi_turn_add38a14_obf'        → 'multi_turn'
          'semantic_indirect_stealth_21e716c7_obf_zero' → 'indirect_stealth'
          'context_poisoning_11fb9bed_obf'          → 'context_poisoning'
        """
        name = family_name

        # Strip 'semantic_' prefix
        if name.startswith("semantic_"):
            name = name[len("semantic_"):]

        # Strip trailing _obf, _zero, _flip and hash-like suffixes
        # Hash pattern: 8-hex-char segment
        parts = name.split("_")
        clean_parts = []
        for part in parts:
            # Stop at hash-like segments (8 hex chars) or known suffixes
            if re.fullmatch(r"[0-9a-f]{8}", part):
                break
            if part in {"obf", "zero", "flip"}:
                break
            clean_parts.append(part)

        return "_".join(clean_parts) if clean_parts else family_name

    def drift_report(self, results: list[VerificationResult]) -> dict:
        """Summarize drift stats for a batch. For logging and report output."""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        by_layer = {1: 0, 2: 0, 3: 0, 4: 0}
        by_type: dict[str, int] = {}
        for r in results:
            if not r.passed and r.layer is not None:
                by_layer[r.layer] += 1
                if r.drift_type:
                    by_type[r.drift_type] = by_type.get(r.drift_type, 0) + 1
        return {
            "total": total,
            "passed": passed,
            "drifted": total - passed,
            "drift_rate": (total - passed) / total if total else 0.0,
            "by_layer": by_layer,
            "by_type": by_type,
        }
