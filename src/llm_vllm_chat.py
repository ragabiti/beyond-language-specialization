"""
llm_vllm_chat.py — Drop-in replacement di llm.py via vLLM.

Gestisce automaticamente due modalità in base al modello:
  - Chat template  → /v1/chat/completions  (gemma, granite, ministral, fastweb, villanova)
  - Raw completion → /v1/completions        (llama, qwen, altri)

Equivalente a llm.py per tutti i modelli UniQA. Nessun crop — il contesto
viene passato intero a vLLM (context window gestita da --max-model-len).

Prerequisito per UniQA (contesto max ~21000 token):
    vllm serve <model_id> \
        --port 8000 --dtype bfloat16 \
        --gpu-memory-utilization 0.92 \
        --max-model-len 32768 \
        --max-num-seqs 8 \
        --disable-log-requests

Comparabilità con llm.py:
  - temperature=0.0       → equivalente a do_sample=False
  - repetition_penalty=1.1
  - chat template applicato agli stessi modelli di llm.py
  - Qwen thinking mode soppresso via /no_think (come nei run RAGAS)
"""

import re
import requests
from typing import List

VLLM_CHAT_URL       = "http://localhost:8000/v1/chat/completions"
VLLM_COMPLETION_URL = "http://localhost:8000/v1/completions"
TIMEOUT             = 300  # secondi — UniQA ha risposte lunghe (max_new_tokens=300)

# Stessa lista di llm.py — questi modelli usano chat template
CHAT_TEMPLATE_MODELS = ["gemma-4", "granite-4.1", "ministral", "fastweb", "villanova"]

# Modelli con thinking mode attivo di default — soppresso via /no_think
THINKING_MODE_MODELS = ["qwen3.5", "qwen3-14"]


def _clean_bpe_artifacts(text: str) -> str:
    """
    Fix artefatti byte-level BPE e mojibake per modelli come FastwebMIIA.
      Ġ (\u0120) → spazio
      Ċ (\u010a) → newline
      Ã¨ → è  (mojibake Latin-1/UTF-8)
    """
    text = text.replace('\u0120', ' ')
    text = text.replace('\u010a', '\n')
    try:
        text = text.encode('raw_unicode_escape').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    text = re.sub(r' +', ' ', text)
    return text.strip()


class LLM:
    """
    Drop-in replacement di llm.py via vLLM.
    Interfaccia identica: generate(prompt, max_new_tokens) → List[str]
    """

    def __init__(self, model_id: str, **kwargs):
        self.model_id    = model_id
        self._lower      = model_id.lower()
        self._use_chat   = any(m in self._lower for m in CHAT_TEMPLATE_MODELS)
        self._thinking   = any(m in self._lower for m in THINKING_MODE_MODELS)

        mode = "chat/completions" if self._use_chat else "completions (raw)"
        think = " [thinking suppressed]" if self._thinking else ""

        try:
            r = requests.get("http://localhost:8000/v1/models", timeout=5)
            r.raise_for_status()
            print(f"✓ vLLM raggiungibile — modello: {model_id}  modo: {mode}{think}")
        except Exception as e:
            raise RuntimeError(
                f"vLLM non raggiungibile su localhost:8000.\n"
                f"Lancia: vllm serve {model_id} --port 8000 --dtype bfloat16 "
                f"--gpu-memory-utilization 0.92 --max-model-len 32768 --max-num-seqs 8\n"
                f"Errore: {e}"
            )

    def _generate_one(self, prompt: str, max_new_tokens: int) -> str:
        if self._use_chat:
            return self._chat(prompt, max_new_tokens)
        else:
            return self._completion(prompt, max_new_tokens)

    def _chat(self, prompt: str, max_new_tokens: int) -> str:
        """
        /v1/chat/completions — per modelli con chat template.
        Equivalente a llm.py apply_chat_template([{"role": "user", "content": prompt}]).
        """
        messages = []
        # Sopprimi thinking mode per Qwen via system prompt
        if self._thinking:
            messages.append({"role": "system", "content": "/no_think"})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model":              self.model_id,
            "messages":           messages,
            "max_tokens":         max_new_tokens,
            "temperature":        0.0,
            "repetition_penalty": 1.1,
        }
        resp = requests.post(VLLM_CHAT_URL, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _completion(self, prompt: str, max_new_tokens: int) -> str:
        """
        /v1/completions — per modelli raw (Llama, Qwen in raw mode, ecc.).
        Equivalente a llm.py senza chat template.
        """
        # Per Qwen thinking mode: prepend /no_think al prompt
        actual_prompt = f"/no_think\n{prompt}" if self._thinking else prompt

        payload = {
            "model":              self.model_id,
            "prompt":             actual_prompt,
            "max_tokens":         max_new_tokens,
            "temperature":        0.0,
            "repetition_penalty": 1.1,
        }
        resp = requests.post(VLLM_COMPLETION_URL, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()["choices"][0]["text"]

    def generate(self, prompts, max_new_tokens: int = 300) -> List[str]:
        """
        Genera risposte per uno o più prompt.
        Interfaccia identica a llm.py — ritorna lista di stringhe.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        results = []
        for prompt in prompts:
            text = self._generate_one(prompt, max_new_tokens)
            text = _clean_bpe_artifacts(text)
            results.append(text)

        return results
