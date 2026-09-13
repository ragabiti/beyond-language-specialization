"""
llm_velvet_uniqa.py — Wrapper API Velvet per il benchmark UniQA IT.

Versione adattata da llm_velvet.py per UniQA:
  - System prompt in italiano per risposte complete (non estrattive max 5 token)
  - Output: solo la risposta generata, senza il prompt (compatibile con generate_rag_uniqa.py)
  - max_tokens default: 300 (era 15)

NON usare per il run EN (TriviaQA/NQ/BioASQ) — quello usa llm_velvet.py originale.
"""

import os
import requests
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

API_URL   = os.getenv("LLM_API_URL")
API_TOKEN = os.getenv("LLM_API_TOKEN")

# System prompt per UniQA IT — risposta completa in italiano
# Diverso dal run EN che usava "extractive QA, max 5 words"
UNIQA_SYSTEM_PROMPT = (
    "Sei un assistente universitario esperto. "
    "Ti vengono forniti dei documenti e una domanda. "
    "Rispondi IN ITALIANO in modo completo e preciso, "
    "basandoti ESCLUSIVAMENTE sulle informazioni presenti nei documenti forniti. "
    "Se i documenti non contengono informazioni sufficienti, rispondi con NO-RES."
)


class LLM:
    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        quantization_bits: Optional[int] = None,
        stop_list: Optional[List[str]] = None,
        model_max_length: int = 8192,
    ):
        self.model_id = model_id
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_TOKEN}",
        }
        print(f"Modello Velvet (UniQA): {model_id}")

    def generate(self, prompts, max_new_tokens: int = 300) -> List[str]:
        if isinstance(prompts, str):
            prompts = [prompts]

        results = []
        for prompt in prompts:
            response_text = self._call_api(prompt, max_new_tokens)
            results.append(response_text)  # solo risposta, senza prompt

        return results

    def _call_api(self, prompt: str, max_new_tokens: int) -> str:
        payload = {
            "model": self.model_id,
            "stream": False,
            "max_tokens": max_new_tokens,
            "messages": [
                {"role": "system", "content": UNIQA_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        }

        try:
            response = requests.post(
                API_URL,
                headers=self.headers,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]

        except requests.exceptions.Timeout:
            print("[Warning] Timeout sulla richiesta.")
            return ""
        except requests.exceptions.HTTPError as e:
            print(f"[Warning] Errore HTTP {e.response.status_code}: {e.response.text}")
            return ""
        except (KeyError, IndexError) as e:
            print(f"[Warning] Formato risposta inatteso: {e}")
            return ""
        except requests.exceptions.RequestException as e:
            print(f"[Warning] Errore di rete: {e}")
            return ""
