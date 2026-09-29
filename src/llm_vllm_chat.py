"""
llm_vllm_chat.py — Drop-in replacement di llm.py via vLLM.

Gestisce automaticamente due modalita' in base al modello:
  - Chat template  -> /v1/chat/completions  (gemma, granite, ministral, fastweb, villanova)
  - Raw completion -> /v1/completions        (llama, qwen, altri)

Equivalente a llm.py per tutti i modelli UniQA. Nessun crop — il contesto
viene passato intero a vLLM (context window gestita da --max-model-len).

Prerequisito per UniQA (contesto max ~21000 token):
    vllm serve <model_id> \\
        --port 8000 --dtype bfloat16 \\
        --gpu-memory-utilization 0.92 \\
        --max-model-len 32768 \\
        --max-num-seqs 8 \\
        --disable-log-requests

Comparabilita' con llm.py:
  - temperature=0.0       -> equivalente a do_sample=False
  - repetition_penalty=1.1
  - chat template applicato agli stessi modelli di llm.py
  - Qwen thinking mode soppresso via /no_think (come nei run RAGAS)

Dipendenze principali:
  - requests (chiamate HTTP al server vLLM locale)
  - re (pulizia artefatti BPE)
  - Server vLLM in esecuzione su localhost:8000

Esempio d'uso:
    from src.llm_vllm_chat import LLM
    llm = LLM("meta-llama/Llama-3-8B")
    risposte = llm.generate("Qual e' la capitale d'Italia?", max_new_tokens=50)
    print(risposte)  # ["Roma"]
"""

import re
import requests
from typing import List

# URL degli endpoint vLLM locali per le due modalita' di generazione
VLLM_CHAT_URL       = "http://localhost:8000/v1/chat/completions"
VLLM_COMPLETION_URL = "http://localhost:8000/v1/completions"

# Timeout in secondi per le richieste HTTP — UniQA ha risposte lunghe (max_new_tokens=300)
TIMEOUT             = 300

# Stessa lista di llm.py — questi modelli richiedono chat template obbligatorio
CHAT_TEMPLATE_MODELS = ["gemma-4", "granite-4.1", "ministral", "fastweb", "villanova"]

# Modelli con thinking mode attivo di default — soppresso via /no_think
# nel system prompt (chat) o come prefisso del prompt (completion)
THINKING_MODE_MODELS = ["qwen3.5", "qwen3-14"]


def _clean_bpe_artifacts(text: str) -> str:
    """
    Pulisce artefatti byte-level BPE e mojibake dal testo generato.

    Correzioni applicate:
      - Ġ (U+0120) -> spazio
      - Ċ (U+010A) -> newline
      - Mojibake Latin-1/UTF-8 (es. Ã¨ -> è) per modelli come FastwebMIIA
      - Spazi multipli consecutivi -> singolo spazio

    Args:
        text: testo grezzo restituito dal server vLLM, potenzialmente contenente
            artefatti di tokenizzazione BPE.

    Returns:
        Testo pulito e normalizzato.
    """
    # Sostituzione dei caratteri BPE byte-level
    text = text.replace('\u0120', ' ')
    text = text.replace('\u010a', '\n')
    # Tentativo di correzione mojibake Latin-1/UTF-8
    try:
        text = text.encode('raw_unicode_escape').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    # Normalizzazione degli spazi multipli
    text = re.sub(r' +', ' ', text)
    return text.strip()


class LLM:
    """
    Drop-in replacement di llm.py via vLLM.

    Seleziona automaticamente la modalita' di generazione in base al modello:
      - Chat template (/v1/chat/completions) per modelli chat-only
      - Raw completion (/v1/completions) per tutti gli altri modelli
    Sopprime automaticamente il thinking mode di Qwen3 via /no_think.

    Interfaccia identica a llm.py: generate(prompt, max_new_tokens) -> List[str]
    """

    def __init__(self, model_id: str, **kwargs):
        """
        Inizializza il wrapper verificando la raggiungibilita' del server vLLM.

        Determina automaticamente la modalita' di generazione (chat vs. raw)
        e se il thinking mode deve essere soppresso, in base al model_id.

        Args:
            model_id: identificativo del modello servito da vLLM
                (es. "meta-llama/Llama-3-8B").
            **kwargs: argomenti aggiuntivi (ignorati, per compatibilita'
                con l'interfaccia di llm.py).

        Raises:
            RuntimeError: se il server vLLM non e' raggiungibile su localhost:8000.
        """
        self.model_id    = model_id
        self._lower      = model_id.lower()
        # Determina se il modello richiede chat template o raw completion
        self._use_chat   = any(m in self._lower for m in CHAT_TEMPLATE_MODELS)
        # Determina se il modello ha thinking mode da sopprimere
        self._thinking   = any(m in self._lower for m in THINKING_MODE_MODELS)

        mode = "chat/completions" if self._use_chat else "completions (raw)"
        think = " [thinking suppressed]" if self._thinking else ""

        # Verifica che il server vLLM sia raggiungibile prima di procedere
        try:
            r = requests.get("http://localhost:8000/v1/models", timeout=5)
            r.raise_for_status()
            print(f"vLLM raggiungibile — modello: {model_id}  modo: {mode}{think}")
        except Exception as e:
            raise RuntimeError(
                f"vLLM non raggiungibile su localhost:8000.\n"
                f"Lancia: vllm serve {model_id} --port 8000 --dtype bfloat16 "
                f"--gpu-memory-utilization 0.92 --max-model-len 32768 --max-num-seqs 8\n"
                f"Errore: {e}"
            )

    def _generate_one(self, prompt: str, max_new_tokens: int) -> str:
        """
        Genera la risposta per un singolo prompt, scegliendo automaticamente
        la modalita' appropriata (chat template o raw completion).

        Args:
            prompt: testo del prompt utente.
            max_new_tokens: numero massimo di token da generare.

        Returns:
            Testo generato dal modello.
        """
        # Smistamento alla modalita' corretta in base al tipo di modello
        if self._use_chat:
            return self._chat(prompt, max_new_tokens)
        else:
            return self._completion(prompt, max_new_tokens)

    def _chat(self, prompt: str, max_new_tokens: int) -> str:
        """
        Generazione via /v1/chat/completions per modelli con chat template.

        Equivalente a llm.py apply_chat_template([{{"role": "user", "content": prompt}}]).
        Per i modelli Qwen con thinking mode, aggiunge un system prompt
        /no_think per sopprimere la catena di pensiero.

        Args:
            prompt: testo del prompt utente.
            max_new_tokens: numero massimo di token da generare.

        Returns:
            Testo generato dal modello (solo il contenuto della risposta).
        """
        messages = []
        # Sopprimi thinking mode per Qwen via system prompt /no_think
        if self._thinking:
            messages.append({"role": "system", "content": "/no_think"})
        messages.append({"role": "user", "content": prompt})

        # Costruzione del payload con parametri di generazione deterministici
        payload = {
            "model":              self.model_id,
            "messages":           messages,
            "max_tokens":         max_new_tokens,
            "temperature":        0.0,
            "repetition_penalty": 1.1,
        }
        # Invio della richiesta al server vLLM locale (endpoint chat)
        resp = requests.post(VLLM_CHAT_URL, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _completion(self, prompt: str, max_new_tokens: int) -> str:
        """
        Generazione via /v1/completions per modelli raw (LLaMA, Qwen in modalita' raw, ecc.).

        Equivalente a llm.py senza chat template. Per i modelli Qwen con thinking
        mode, antepone /no_think al prompt per sopprimere la catena di pensiero.

        Args:
            prompt: testo del prompt utente.
            max_new_tokens: numero massimo di token da generare.

        Returns:
            Testo generato dal modello (solo la continuazione, senza il prompt).
        """
        # Per Qwen thinking mode: antepone /no_think al prompt come prefisso
        actual_prompt = f"/no_think\n{prompt}" if self._thinking else prompt

        # Costruzione del payload con parametri di generazione deterministici
        payload = {
            "model":              self.model_id,
            "prompt":             actual_prompt,
            "max_tokens":         max_new_tokens,
            "temperature":        0.0,
            "repetition_penalty": 1.1,
        }
        # Invio della richiesta al server vLLM locale (endpoint completion)
        resp = requests.post(VLLM_COMPLETION_URL, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()["choices"][0]["text"]

    def generate(self, prompts, max_new_tokens: int = 300) -> List[str]:
        """
        Genera risposte per uno o piu' prompt via server vLLM.

        Interfaccia identica a llm.py — ritorna lista di stringhe contenenti
        solo il testo generato. Applica automaticamente la pulizia degli
        artefatti BPE su ciascuna risposta.

        Args:
            prompts: stringa singola o lista di stringhe contenenti i prompt.
            max_new_tokens: numero massimo di token da generare per ciascun
                prompt (default 300, adatto a UniQA).

        Returns:
            Lista di stringhe contenenti solo le risposte generate, una per
            ciascun prompt fornito, con artefatti BPE rimossi.
        """
        # Normalizza l'input: stringa singola -> lista
        if isinstance(prompts, str):
            prompts = [prompts]

        results = []
        for prompt in prompts:
            # Generazione tramite la modalita' appropriata (chat o completion)
            text = self._generate_one(prompt, max_new_tokens)
            # Pulizia artefatti BPE e mojibake dal testo generato
            text = _clean_bpe_artifacts(text)
            results.append(text)

        return results
