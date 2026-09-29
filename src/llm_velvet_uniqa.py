"""
llm_velvet_uniqa.py — Wrapper API Velvet per il benchmark UniQA IT.

Versione specializzata di llm_velvet.py per il dataset UniQA italiano:
  - System prompt in italiano che istruisce il modello a fornire risposte
    complete e precise basate sui documenti forniti
  - Output: solo la risposta generata, senza il prompt (compatibile con
    generate_rag_uniqa.py)
  - max_tokens default: 300 (adatto a risposte discorsive, non estrattive)
  - Pulizia artefatti BPE/mojibake allineata con llm_velvet.py
  - Fallback automatico su payload minimale se l'endpoint rifiuta
    repetition_penalty/top_p (errore 400/422)

NON usare per il run EN (TriviaQA/NQ/BioASQ) — quello usa llm_velvet.py originale.

Dipendenze principali:
  - requests (chiamate HTTP all'API)
  - python-dotenv (caricamento variabili d'ambiente da .env)

Esempio d'uso:
    from src.llm_velvet_uniqa import LLM
    llm = LLM("nome-modello-velvet")
    risposte = llm.generate("Contesto: ... Domanda: ...", max_new_tokens=300)
    print(risposte)  # ["La risposta basata sui documenti..."]
"""

import os
import re
import time
import requests
from typing import List, Optional
from dotenv import load_dotenv

# Caricamento delle variabili d'ambiente dal file .env
load_dotenv(dotenv_path=".env", override=True)

# URL e token dell'API Velvet, letti dalle variabili d'ambiente
API_URL   = os.getenv("LLM_API_URL")
API_TOKEN = os.getenv("LLM_API_TOKEN")

# Timeout in secondi per le richieste HTTP all'API
TIMEOUT = 300

# Rate limiting: 1000 richieste/ora -> minimo 3.6s tra le chiamate
REQUEST_INTERVAL = 3.6
# Retry su 429: numero massimo di tentativi e backoff iniziale
MAX_RETRIES_429 = 5
RETRY_BASE_DELAY = 5.0

# System prompt per UniQA IT: istruisce il modello a rispondere in italiano
# in modo completo e preciso, basandosi solo sui documenti forniti.
# Diverso dal run EN che usava "extractive QA, max 5 words"
UNIQA_SYSTEM_PROMPT = (
    "Sei un assistente universitario esperto. "
    "Ti vengono forniti dei documenti e una domanda. "
    "Rispondi IN ITALIANO in modo completo e preciso, "
    "basandoti ESCLUSIVAMENTE sulle informazioni presenti nei documenti forniti. "
    "Se i documenti non contengono informazioni sufficienti, rispondi con NO-RES."
)


def _clean_bpe_artifacts(text: str) -> str:
    """Pulisce artefatti byte-level BPE e mojibake dal testo generato."""
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
    Wrapper API Velvet specializzato per il benchmark UniQA IT.

    Rispetto a llm_velvet.py, aggiunge un system prompt in italiano che
    guida il modello a produrre risposte complete (non estrattive) basate
    esclusivamente sui documenti forniti nel contesto.

    Interfaccia identica a llm.py e llm_velvet.py per garantire
    compatibilita' drop-in con generate_rag_uniqa.py.
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
        """
        Inizializza il wrapper per UniQA IT configurando le credenziali API.

        Args:
            model_id: nome del modello Velvet da invocare.
            device: dispositivo (ignorato, l'inferenza avviene lato server).
            quantization_bits: bit di quantizzazione (ignorato, gestito dal server).
            stop_list: lista di sequenze di stop. Se presenti, il testo generato
                viene troncato alla prima occorrenza di una di queste stringhe.
            model_max_length: lunghezza massima del contesto in token (informativo).
            timeout: timeout in secondi per le richieste HTTP.

        Raises:
            RuntimeError: se LLM_API_URL o LLM_API_TOKEN non sono configurate.
        """
        self.model_id = model_id
        self.stop_list = stop_list or []
        self.model_max_length = model_max_length
        self.timeout = timeout
        self._last_request_time = 0.0

        # Verifica che le credenziali API siano configurate
        if not API_URL:
            raise RuntimeError("LLM_API_URL non impostata nel file .env o nell'ambiente.")
        if not API_TOKEN:
            raise RuntimeError("LLM_API_TOKEN non impostata nel file .env o nell'ambiente.")

        # Preparazione degli header HTTP con token di autenticazione
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_TOKEN}",
        }
        print(f"Modello Velvet (UniQA): {model_id}")

    def generate(self, prompts, max_new_tokens: int = 300) -> List[str]:
        """
        Genera risposte per uno o piu' prompt tramite l'API Velvet con system prompt UniQA.

        Args:
            prompts: stringa singola o lista di stringhe contenenti i prompt.
            max_new_tokens: numero massimo di token da generare (default 300).

        Returns:
            Lista di stringhe contenenti le risposte generate, ripulite
            dagli artefatti BPE e troncate secondo la stop_list.
        """
        # Normalizza l'input: stringa singola -> lista
        if isinstance(prompts, str):
            prompts = [prompts]

        results = []
        for prompt in prompts:
            # Chiamata all'API per ciascun prompt (include system prompt UniQA)
            text = self._call_api(prompt, max_new_tokens)
            # Pulizia artefatti BPE e mojibake
            text = _clean_bpe_artifacts(text)
            # Troncamento alle sequenze di stop, se configurate
            text = self._apply_stop_list(text)
            results.append(text)

        return results

    def _wait_rate_limit(self):
        """Attende il tempo necessario per rispettare il rate limit (1000 req/ora)."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_INTERVAL:
            time.sleep(REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _call_api(self, prompt: str, max_new_tokens: int) -> str:
        """
        Effettua una singola chiamata all'API Velvet con system prompt UniQA
        e parametri di sampling completi.
        Su errore 429 (rate limit), riprova con backoff esponenziale.

        Args:
            prompt: testo del prompt utente (contesto + domanda).
            max_new_tokens: numero massimo di token da generare.

        Returns:
            Testo generato dal modello, oppure stringa vuota in caso di errore.
        """
        # Payload completo: system prompt UniQA + parametri di sampling
        payload = {
            "model": self.model_id,
            "stream": False,
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": 1.1,
            "messages": [
                {"role": "system", "content": UNIQA_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        }

        # Rispetta il rate limit prima di ogni chiamata
        self._wait_rate_limit()

        for attempt in range(MAX_RETRIES_429 + 1):
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

                # Retry con backoff esponenziale su 429 (rate limit)
                if status == 429 and attempt < MAX_RETRIES_429:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"[Rate limit] 429 — retry {attempt + 1}/{MAX_RETRIES_429} tra {delay:.0f}s")
                    time.sleep(delay)
                    self._last_request_time = time.time()
                    continue

                print(f"[Warning] Errore HTTP {status}: {body}")

                # Fallback: riprova senza repetition_penalty/top_p
                if status in (400, 422):
                    return self._call_api_minimal(prompt, max_new_tokens)
                return ""

            except (KeyError, IndexError, TypeError) as e:
                print(f"[Warning] Formato risposta inatteso: {e}")
                return ""

            except requests.exceptions.RequestException as e:
                print(f"[Warning] Errore di rete: {e}")
                return ""

        print("[Warning] Rate limit: tentativi esauriti.")
        return ""

    def _call_api_minimal(self, prompt: str, max_new_tokens: int) -> str:
        """Fallback minimale senza repetition_penalty e top_p."""
        payload = {
            "model": self.model_id,
            "stream": False,
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
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
        """Tronca il testo alla prima sequenza di stop trovata."""
        if not self.stop_list:
            return text
        cut_positions = [text.find(stop) for stop in self.stop_list if stop and stop in text]
        if not cut_positions:
            return text
        return text[:min(cut_positions)].strip()
