"""
news/llm_analyst.py
===================
Local Llama3 integration via Ollama for Quant Predator v2.

Provides two capabilities:
    1. Market commentary — plain-English summary of current market conditions
       to explain the model's last signal to the user.
    2. Interactive chat — question-answer interface in the Streamlit dashboard,
       with full trading context injected into every prompt.

PREREQUISITES:
    1. Install Ollama:  https://ollama.com/download
    2. Pull the model:  ollama pull llama3
    3. Verify running:  curl http://localhost:11434/api/tags

All inference runs locally — no data leaves the machine, no API costs.
Gracefully returns fallback strings if Ollama is offline.
"""

from __future__ import annotations

import json
from typing import Dict, Generator, Optional

from core.config import LLMConfig
from core.logger import get_logger

logger = get_logger(__name__)

# Ollama Python client — imported lazily so the app starts even if not installed
try:
    import ollama as _ollama_lib
    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False
    logger.warning(
        "ollama Python package not found. "
        "Install with: pip install ollama. "
        "LLM features will be disabled."
    )

_SYSTEM_PROMPT = """You are QuantPredator, an institutional algorithmic trading AI.
You are speaking directly to your operator via the live trading dashboard.

Your personality: precise, analytical, no-nonsense. You speak in short sentences.
You never give financial advice. You explain what the data shows, not what to do.

Context you always have:
- You trade EURUSD on H1 timeframe using a PPO reinforcement learning model.
- Your signals come from a Multi-Timeframe Matrix: H1 entry aligned with H4/D1 structure.
- Risk per trade: 1% of account. Breakeven trigger: price moves 1R in your favour.
- You halt trading 30 minutes before high-impact news events.

When asked about a trade, explain: the MTF alignment, the FVG context, the session,
and whether news risk was a factor. Keep responses under 150 words unless asked for detail.
"""


class LLMAnalyst:
    """
    Ollama-based local LLM for trading commentary and interactive chat.

    All methods are synchronous and safe to call from the Streamlit main thread.
    Streaming responses are supported for the chat interface.
    """

    def __init__(self) -> None:
        self._available = _OLLAMA_AVAILABLE
        if self._available:
            self._verify_connection()

    def is_available(self) -> bool:
        """Returns True if Ollama is running and llama3 is accessible."""
        return self._available

    def explain_signal(
        self,
        signal:         str,
        symbol:         str,
        price:          float,
        mtf_alignment:  int,
        session:        str,
        news_status:    str,
        rsi:            float,
        bb_width:       float,
        fvg_active:     bool,
    ) -> str:
        """
        Generates a plain-English explanation of the latest AI signal.

        Shown in the dashboard below the signal banner when the user hovers
        or when a new trade is opened.

        Args:
            signal:        "BUY", "SELL", "HOLD", or "NEWS HALT".
            symbol:        e.g., "EURUSD".
            price:         Current mid-price.
            mtf_alignment: MTF_Alignment value (-3 to +3).
            session:       "Asian", "London", "New York", or "Overlap".
            news_status:   From ForexNewsRadar.
            rsi:           Current RSI(14).
            bb_width:      Current BB Width.
            fvg_active:    Whether a relevant FVG zone is active.

        Returns:
            Explanation string (max ~150 words), or a fallback if offline.
        """
        if not self._available:
            return self._offline_explanation(signal)

        fvg_str = "active unmitigated FVG present" if fvg_active else "no active FVG"
        prompt = (
            f"Signal: {signal} on {symbol} at {price:.5f}\n"
            f"MTF Alignment: {mtf_alignment:+d}/3 "
            f"({'bullish' if mtf_alignment > 0 else 'bearish' if mtf_alignment < 0 else 'neutral'})\n"
            f"Session: {session} | RSI: {rsi:.1f} | BB Width: {bb_width:.4f}\n"
            f"FVG: {fvg_str}\n"
            f"News: {news_status}\n\n"
            f"In 2-3 sentences, explain why the model produced this signal."
        )
        return self._chat_sync(prompt, max_tokens=200)

    def chat(self, user_message: str, context: Dict) -> str:
        """
        Processes a user chat message with full trading context injected.

        Used by the Streamlit chat panel for interactive Q&A.

        Args:
            user_message: Raw text from the user's chat input.
            context:      Dictionary of current live state (from state.json).

        Returns:
            Response string from Llama3.
        """
        if not self._available:
            return (
                "Ollama is not running. Start it with `ollama serve`, "
                "then pull the model with `ollama pull llama3`."
            )

        context_block = (
            f"[LIVE CONTEXT]\n"
            f"Balance: ${context.get('balance', 0):,.2f} | "
            f"Equity: ${context.get('equity', 0):,.2f}\n"
            f"Position: {context.get('open_position', 'FLAT')} | "
            f"Floating: ${context.get('floating_profit', 0):+.2f}\n"
            f"Last Signal: {context.get('ai_signal', 'N/A')} | "
            f"News: {context.get('news_status', 'N/A')}\n"
            f"[USER MESSAGE]\n{user_message}"
        )
        return self._chat_sync(context_block, max_tokens=LLMConfig.MAX_TOKENS)

    def stream_chat(
        self, user_message: str, context: Dict
    ) -> Generator[str, None, None]:
        """
        Streaming version of chat() — yields response tokens as they arrive.
        Use with Streamlit's st.write_stream() for a live typing effect.

        Args:
            user_message: Raw user input.
            context:      Live state dictionary.

        Yields:
            Token strings from the Llama3 stream.
        """
        if not self._available:
            yield "Ollama is offline. Run `ollama serve` to enable the chat interface."
            return

        context_block = (
            f"[LIVE CONTEXT]\n"
            f"Balance: ${context.get('balance', 0):,.2f} | "
            f"Position: {context.get('open_position', 'FLAT')}\n"
            f"Signal: {context.get('ai_signal', 'N/A')} | "
            f"News: {context.get('news_status', 'N/A')}\n"
            f"[USER MESSAGE]\n{user_message}"
        )

        try:
            stream = _ollama_lib.chat(
                model   = LLMConfig.MODEL_NAME,
                messages= [
                    {"role": "system",  "content": _SYSTEM_PROMPT},
                    {"role": "user",    "content": context_block},
                ],
                stream  = True,
                options = {"num_predict": LLMConfig.MAX_TOKENS},
            )
            for chunk in stream:
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
        except Exception as exc:
            logger.warning(f"Ollama stream error: {exc}")
            yield f"[LLM error: {exc}]"

    # ==========================================================================
    # PRIVATE
    # ==========================================================================

    def _chat_sync(self, prompt: str, max_tokens: int = 300) -> str:
        """Sends a single-turn prompt to Llama3 and returns the full response."""
        try:
            resp = _ollama_lib.chat(
                model   = LLMConfig.MODEL_NAME,
                messages= [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                options = {"num_predict": max_tokens},
            )
            return resp.get("message", {}).get("content", "No response.").strip()
        except Exception as exc:
            logger.warning(f"Ollama request failed: {exc}")
            return f"LLM unavailable: {exc}"

    def _verify_connection(self) -> None:
        """Tests the Ollama connection at initialization time."""
        try:
            models = _ollama_lib.list()
            model_names = [m.get("name", "") for m in models.get("models", [])]
            if LLMConfig.MODEL_NAME not in " ".join(model_names):
                logger.warning(
                    f"Model '{LLMConfig.MODEL_NAME}' not found in Ollama. "
                    f"Pull it with: ollama pull {LLMConfig.MODEL_NAME}"
                )
                self._available = False
            else:
                logger.info(f"Ollama connected — model '{LLMConfig.MODEL_NAME}' ready.")
        except Exception as exc:
            logger.warning(
                f"Cannot reach Ollama at {LLMConfig.BASE_URL}: {exc}. "
                f"LLM features disabled. Start Ollama with: ollama serve"
            )
            self._available = False

    @staticmethod
    def _offline_explanation(signal: str) -> str:
        """Returns a rule-based fallback when Ollama is offline."""
        fallbacks = {
            "BUY":       "Long signal: H1 structure above HTF EMA. MTF bullish alignment.",
            "SELL":      "Short signal: H1 structure below HTF EMA. MTF bearish alignment.",
            "HOLD":      "No valid setup. Waiting for structural alignment.",
            "NEWS HALT": "High-impact news imminent. All entries suspended.",
        }
        return fallbacks.get(signal.upper(), "Signal analysis unavailable. Ollama is offline.")
