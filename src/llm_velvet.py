"""
llm_velvet.py — Wrapper unificato per le API Velvet.

Sostituzione drop-in compatibile con llm.py e llm_vllm_chat.py, senza
necessita' di un server vLLM locale. Comunica con l'endpoint API Velvet
configurato tramite variabili d'ambiente.

Dataset supportati:
  - UniQA IT/EN
  - PharmaQA
  - Dataset inglesi (TriviaQA, NQ, BioASQ)
  - Qualsiasi modello servito dall'endpoint Velvet API (LLM_API_URL)

Comportamento:
  - generate(prompt, max_new_tokens) -> List[str]
  - Restituisce solo la risposta generata, mai prompt + risposta
  - Nessun system prompt specifico per dataset: le istruzioni del task
    devono essere incluse nel prompt utente
  - Generazione deterministica: temperature=0.0
  - repetition_penalty=1.1 (se accettata dall'endpoint)
  - timeout=300 secondi
  - Pulizia artefatti BPE/mojibake allineata con llm_vllm_chat.py

Dipendenze principali:
  - requests (chiamate HTTP all'API)
  - python-dotenv (caricamento variabili d'ambiente da .env)

Esempio d'uso:
    from src.llm_velvet import LLM
    llm = LLM("nome-modello-velvet")
    risposte = llm.generate("Qual e' la capitale d'Italia?", max_new_tokens=50)
    print(risposte)  # ["Roma"]
"""

import os
import re
import requests
from typing import List, Optional
from dotenv import load_dotenv

# Caricamento delle variabili d'ambiente dal file .env
load_dotenv(dotenv_path=".env", override=True)

# URL e token dell'API Velvet, letti dalle variabili d'ambiente
API_URL = os.getenv("LLM_API_URL")
API_TOKEN = os.getenv("LLM_API_TOKEN")

# Timeout in secondi per le richieste HTTP all'API
TIMEOUT = 300


def _clean_bpe_artifacts(text: str) -> str:
    """
    Pulisce artefatti byte-level BPE e mojibake dal testo generato.

    Stessa logica di pulizia usata in llm_vllm_chat.py:
      - Ġ (U+0120) viene convertito in spazio
      - Ċ (U+010A) viene convertito in newline
      - Tenta di correggere mojibake comuni Latin-1/UTF-8 (es. Ã¨ -> è)
      - Collassa spazi multipli consecutivi in un singolo spazio

    Args:
        text: testo grezzo restituito dall'API, potenzialmente contenente
            artefatti di tokenizzazione BPE.

    Returns:
        Testo pulito e normalizzato, oppure stringa vuota se l'input e' None.
    """
    # Gestione del caso None (risposta vuota dall'API)
    if text is None:
        return ""

    # Sostituzione dei caratteri BPE byte-level
    text = text.replace("\u0120", " ")
    text = text.replace("\u010a", "\n")

    # Tentativo di correzione mojibake Latin-1/UTF-8
    try:
        text = text.encode("raw_unicode_escape").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass

    # Normalizzazione degli spazi multipli
    text = re.sub(r" +", " ", text)
    return text.strip()


class LLM:
    """
    Wrapper unificato per le API Velvet.

    Interfaccia identica a llm.py e llm_vllm_chat.py per garantire
    compatibilita' drop-in in tutti gli script di valutazione RAG.

    Interfaccia:
        llm = LLM(model_id)
        risposte = llm.generate(prompt_o_lista_prompt, max_new_tokens=N)

    Output:
        List[str], contenente solo il testo generato (senza il prompt).
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
        Inizializza il wrapper verificando la presenza delle credenziali API.

        Args:
            model_id: nome del modello Velvet da invocare (es. "velvet-minerva-7b").
            device: dispositivo (ignorato, l'inferenza avviene lato server).
            quantization_bits: bit di quantizzazione (ignorato, gestito dal server).
            stop_list: lista di sequenze di stop. Se presenti, il testo generato
                viene troncato alla prima occorrenza di una di queste stringhe.
            model_max_length: lunghezza massima del contesto in token (informativo).
            timeout: timeout in secondi per le richieste HTTP.

        Raises:
            RuntimeError: se LLM_API_URL o LLM_API_TOKEN non sono configurate
                nelle variabili d'ambiente o nel file .env.
        """
        self.model_id = model_id
        self.stop_list = stop_list or []
        self.model_max_length = model_max_length
        self.timeout = timeout

        # Verifica che le credenziali API siano configurate
        if not API_URL:
            raise RuntimeError("LLM_API_URL non impostata nel file .env o nell'ambiente.")
        if not API_TOKEN:
            raise RuntimeError("LLM_API_TOKEN non impostata nel file .env o nell'ambiente.")

        # Preparazione degli header HTTP per le richieste all'API
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_TOKEN}",
        }

        print(f"Modello Velvet API: {model_id}")

    def generate(self, prompts, max_new_tokens: int = 300) -> List[str]:
        """
        Genera risposte per uno o piu' prompt tramite l'API Velvet.

        Il valore di default max_new_tokens=300 e' allineato con llm_vllm_chat.py.
        Valori consigliati per dataset specifici:
          - PharmaQA / QA estrattiva: 50
          - TriviaQA / NQ / BioASQ (risposta breve): 50-100
          - UniQA: 300

        Args:
            prompts: stringa singola o lista di stringhe contenenti i prompt.
            max_new_tokens: numero massimo di token da generare per ciascun prompt.

        Returns:
            Lista di stringhe contenenti solo le risposte generate, ripulite
            dagli artefatti BPE e troncate secondo la stop_list.
        """
        # Normalizza l'input: stringa singola -> lista
        if isinstance(prompts, str):
            prompts = [prompts]

        results = []
        for prompt in prompts:
            # Chiamata all'API per ciascun prompt
            text = self._call_api(prompt, max_new_tokens)
            # Pulizia artefatti BPE e mojibake
            text = _clean_bpe_artifacts(text)
            # Troncamento alle sequenze di stop, se configurate
            text = self._apply_stop_list(text)
            results.append(text)

        return results

    def _call_api(self, prompt: str, max_new_tokens: int) -> str:
        """
        Effettua una singola chiamata all'API Velvet con tutti i parametri di sampling.

        Costruisce un payload compatibile con l'interfaccia OpenAI chat/completions,
        includendo temperature, top_p e repetition_penalty. Se l'endpoint rifiuta
        i parametri estesi (errore 400/422), esegue un fallback alla versione minimale.

        Args:
            prompt: testo del prompt utente.
            max_new_tokens: numero massimo di token da generare.

        Returns:
            Testo generato dal modello, oppure stringa vuota in caso di errore.
        """
        # Costruzione del payload con parametri di sampling completi
        # Formato compatibile OpenAI: modello, messaggi, parametri di generazione
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
            # Invio della richiesta POST all'endpoint API
            response = requests.post(
                API_URL,
                headers=self.headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            # Estrazione del contenuto della risposta dal formato OpenAI
            result = response.json()
            return result["choices"][0]["message"]["content"]

        except requests.exceptions.Timeout:
            # Gestione timeout: la richiesta ha superato il tempo limite
            print("[Warning] Timeout sulla richiesta Velvet API.")
            return ""

        except requests.exceptions.HTTPError as e:
            # Gestione errori HTTP: log dello status code e del corpo della risposta
            status = e.response.status_code if e.response is not None else "unknown"
            body = e.response.text if e.response is not None else ""
            print(f"[Warning] Errore HTTP {status}: {body}")

            # Fallback per endpoint che non accettano repetition_penalty/top_p:
            # riprova con payload minimale (solo temperature e max_tokens)
            if status in (400, 422):
                return self._call_api_minimal(prompt, max_new_tokens)

            return ""

        except (KeyError, IndexError, TypeError) as e:
            # Gestione formato risposta inatteso (JSON malformato o struttura diversa)
            print(f"[Warning] Formato risposta inatteso: {e}")
            return ""

        except requests.exceptions.RequestException as e:
            # Gestione errori di rete generici (connessione rifiutata, DNS, ecc.)
            print(f"[Warning] Errore di rete: {e}")
            return ""

    def _call_api_minimal(self, prompt: str, max_new_tokens: int) -> str:
        """
        Fallback minimale per endpoint che non accettano parametri di sampling
        estesi come repetition_penalty e top_p.

        Viene invocato automaticamente da _call_api quando l'endpoint restituisce
        un errore 400 o 422, indicando che i parametri non sono supportati.

        Args:
            prompt: testo del prompt utente.
            max_new_tokens: numero massimo di token da generare.

        Returns:
            Testo generato dal modello, oppure stringa vuota in caso di errore.
        """
        # Payload ridotto: solo i parametri essenziali (senza repetition_penalty e top_p)
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
            # Richiesta HTTP con payload minimale
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
        """
        Tronca il testo generato alla prima occorrenza di una sequenza di stop.

        Scorre tutte le sequenze nella stop_list e, se ne trova almeno una
        presente nel testo, taglia alla posizione piu' vicina all'inizio.

        Args:
            text: testo generato dal modello, gia' ripulito dagli artefatti BPE.

        Returns:
            Testo troncato alla prima sequenza di stop trovata, oppure il testo
            originale se nessuna sequenza di stop e' presente.
        """
        # Se la stop_list e' vuota, restituisce il testo invariato
        if not self.stop_list:
            return text

        # Trova la posizione di ciascuna sequenza di stop presente nel testo
        cut_positions = [text.find(stop) for stop in self.stop_list if stop and stop in text]
        if not cut_positions:
            return text

        # Tronca alla posizione piu' vicina all'inizio del testo
        return text[:min(cut_positions)].strip()
