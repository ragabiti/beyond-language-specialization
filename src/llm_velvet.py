"""
llm_velvet.py — Unified Velvet API wrapper.

Drop-in replacement aligned with llm.py / llm_vllm_chat.py, without vLLM.

Use for:
  - UniQA IT/EN
  - PharmaQA
  - English QA datasets (TriviaQA, NQ, BioASQ)
  - Velvet models served through the API endpoint in LLM_API_URL

Behavior:
  - generate(prompt, max_new_tokens) -> List[str]
  - returns only generated answer, never prompt + answer
  - no dataset-specific system prompt: task instructions must stay inside the user prompt
  - deterministic generation: temperature=0.0
  - repetition_penalty=1.1 if accepted by the endpoint
  - timeout=300
  - BPE/mojibake cleanup aligned with llm_vllm_chat.py
"""

import os
import re
import requests
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)

API_URL = os.getenv("LLM_API_URL")
API_TOKEN = os.getenv("LLM_API_TOKEN")
TIMEOUT = 300


def _clean_bpe_artifacts(text: str) -> str:
    """
    Same cleanup logic used in llm_vllm_chat.py:
      Ġ (U+0120) -> space
      Ċ (U+010A) -> newline
      tries to fix common Latin-1/UTF-8 mojibake
    """
    if text is None:
        return ""

    text = text.replace("\u0120", " ")
    text = text.replace("\u010a", "\n")

    try:
        text = text.encode("raw_unicode_escape").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass

    text = re.sub(r" +", " ", text)
    return text.strip()


class LLM:
    """
    Unified Velvet API wrapper.

    Interface:
        llm = LLM(model_id)
        outputs = llm.generate(prompt_or_prompts, max_new_tokens=N)

    Output:
        List[str], containing only generated text.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        quantization_bits: Optional[int] = None,
        stop_list: Optional[List[str]] = None,
        model_max_length: int = 8192,
        timeout: int = TIMEOUT,
    ):
        self.model_id = model_id
        self.stop_list = stop_list or []
        self.model_max_length = model_max_length
        self.timeout = timeout

        if not API_URL:
            raise RuntimeError("LLM_API_URL non impostata nel file .env o nell'ambiente.")
        if not API_TOKEN:
            raise RuntimeError("LLM_API_TOKEN non impostata nel file .env o nell'ambiente.")

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_TOKEN}",
        }

        print(f"Modello Velvet API: {model_id}")

    def generate(self, prompts, max_new_tokens: int = 300) -> List[str]:
        """
        Generate for one prompt or a list of prompts.

        Default max_new_tokens=300, aligned with llm_vllm_chat.py.
        Override per dataset:
          - PharmaQA / extractive QA: 50
          - TriviaQA / NQ / BioASQ short-answer: 50-100
          - UniQA: 300
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        results = []
        for prompt in prompts:
            text = self._call_api(prompt, max_new_tokens)
            text = _clean_bpe_artifacts(text)
            text = self._apply_stop_list(text)
            results.append(text)

        return results

    def _call_api(self, prompt: str, max_new_tokens: int) -> str:
        payload = {
            "model": self.model_id,
            "stream": False,
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": 1.1,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }

        try:
            response = requests.post(
                API_URL,
                headers=self.headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]

        except requests.exceptions.Timeout:
            print("[Warning] Timeout sulla richiesta Velvet API.")
            return ""

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "unknown"
            body = e.response.text if e.response is not None else ""
            print(f"[Warning] Errore HTTP {status}: {body}")

            # Fallback for endpoints that reject repetition_penalty/top_p.
            if status in (400, 422):
                return self._call_api_minimal(prompt, max_new_tokens)

            return ""

        except (KeyError, IndexError, TypeError) as e:
            print(f"[Warning] Formato risposta inatteso: {e}")
            return ""

        except requests.exceptions.RequestException as e:
            print(f"[Warning] Errore di rete: {e}")
            return ""

    def _call_api_minimal(self, prompt: str, max_new_tokens: int) -> str:
        """
        Minimal fallback for API endpoints that do not accept OpenAI-compatible
        sampling parameters such as repetition_penalty.
        """
        payload = {
            "model": self.model_id,
            "stream": False,
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }

        try:
            response = requests.post(
                API_URL,
                headers=self.headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]

        except requests.exceptions.Timeout:
            print("[Warning] Timeout sulla richiesta Velvet API minimal.")
            return ""
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "unknown"
            body = e.response.text if e.response is not None else ""
            print(f"[Warning] Errore HTTP minimal {status}: {body}")
            return ""
        except (KeyError, IndexError, TypeError) as e:
            print(f"[Warning] Formato risposta inatteso minimal: {e}")
            return ""
        except requests.exceptions.RequestException as e:
            print(f"[Warning] Errore di rete minimal: {e}")
            return ""

    def _apply_stop_list(self, text: str) -> str:
        if not self.stop_list:
            return text

        cut_positions = [text.find(stop) for stop in self.stop_list if stop and stop in text]
        if not cut_positions:
            return text

        return text[:min(cut_positions)].strip()
