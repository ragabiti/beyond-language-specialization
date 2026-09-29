"""
llm.py — Wrapper locale HuggingFace per generazione testo.

Fornisce una classe LLM che carica modelli causali (CausalLM) da HuggingFace
e genera risposte in modalita' greedy (do_sample=False) con repetition_penalty.

Modelli supportati:
  - Qwen3 (con soppressione automatica del thinking mode)
  - Gemma-4, Granite-4.1, Ministral, Fastweb, Minerva, Villanova (chat-only)
  - LLaMA, Qwen e altri modelli raw completion

Dipendenze principali:
  - torch
  - transformers (AutoTokenizer, AutoModelForCausalLM)

Esempio d'uso:
    from src.llm import LLM
    llm = LLM("meta-llama/Llama-3-8B", model_max_length=8192)
    risposte = llm.generate(["Qual e' la capitale d'Italia?"], max_new_tokens=50)
    print(risposte)  # ["Roma"]
"""

import re
import torch
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM


# Modelli con thinking mode attivo di default (Qwen3): il thinking viene
# soppresso a livello di prompt nei wrapper vLLM, qui non serve
THINKING_MODE_MODELS = ["qwen3.5", "qwen3-14"]

# Modelli che richiedono chat template obbligatorio (chat-only):
# il prompt viene avvolto in apply_chat_template prima della tokenizzazione
CHAT_TEMPLATE_MODELS = ["gemma-4", "granite-4.1", "ministral", "fastweb", "minerva","villanova"]

# Modelli che richiedono una classe specifica (Mistral3ForConditionalGeneration)
# invece della generica AutoModelForCausalLM
MISTRAL3_MODELS = ["ministral-3"]


class LLM:
    """
    Wrapper locale per modelli HuggingFace CausalLM.

    Gestisce automaticamente:
      - Raw completion per la maggior parte dei modelli
      - Chat template obbligatorio per modelli chat-only (Gemma, Granite, ecc.)
      - Caricamento con classe specifica per Mistral3
      - repetition_penalty differenziata (1.3 per Minerva, 1.1 per gli altri)
      - Decodifica robusta che restituisce solo il testo generato, senza il prompt

    Args:
        model_id: identificativo HuggingFace del modello (es. "meta-llama/Llama-3-8B").
        device: dispositivo di destinazione (ignorato, si usa device_map="auto").
        quantization_bits: bit di quantizzazione (riservato per usi futuri).
        stop_list: lista di stringhe di stop (riservato per usi futuri).
        model_max_length: lunghezza massima della finestra di contesto in token.
    """

    def __init__(
        self,
        model_id: str,
        device=None,
        quantization_bits: Optional[int] = None,
        stop_list: Optional[List[str]] = None,
        model_max_length: int = 8192,
    ):
        """
        Inizializza il wrapper caricando tokenizer e modello da HuggingFace.

        Args:
            model_id: identificativo HuggingFace del modello.
            device: dispositivo target (ignorato, si usa device_map="auto").
            quantization_bits: bit di quantizzazione (riservato per usi futuri).
            stop_list: sequenze di stop personalizzate (riservato per usi futuri).
            model_max_length: lunghezza massima in token per il contesto.
        """
        self.model_id = model_id
        self.model_max_length = model_max_length
        self._model_lower = model_id.lower()

        # --- Caricamento tokenizer ---
        print(f"Caricamento tokenizer: {model_id}")
        tokenizer_kwargs = dict(
            padding_side="left",
            truncation_side="left",
            model_max_length=model_max_length,
            trust_remote_code=True,
        )
        # Mistral3 necessita di un fix specifico per le regex del tokenizer
        if self._is_mistral3():
            tokenizer_kwargs["fix_mistral_regex"] = True

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        # Imposta il pad token uguale all'eos token per evitare errori di padding
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Tokenizer caricato.")

        # --- Caricamento modello ---
        print(f"Caricamento modello: {model_id}")
        model_kwargs = dict(
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Mistral3 richiede una classe dedicata al posto di AutoModelForCausalLM
        if self._is_mistral3():
            from transformers.models.mistral3 import Mistral3ForConditionalGeneration
            self.model = Mistral3ForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

        # Imposta il modello in modalita' valutazione (disabilita dropout, ecc.)
        self.model.eval()
        print(f"Modello caricato su: {next(self.model.parameters()).device}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_mistral3(self) -> bool:
        """Verifica se il modello appartiene alla famiglia Mistral3.

        Returns:
            True se il model_id contiene uno degli identificativi Mistral3.
        """
        return any(m in self._model_lower for m in MISTRAL3_MODELS)

    def _has_thinking_mode(self) -> bool:
        """Verifica se il modello ha il thinking mode attivo di default (Qwen3).

        Returns:
            True se il model_id contiene uno degli identificativi dei modelli
            con thinking mode.
        """
        return any(m in self._model_lower for m in THINKING_MODE_MODELS)

    def _needs_chat_template(self) -> bool:
        """Verifica se il modello richiede l'applicazione obbligatoria del chat template.

        I modelli chat-only (Gemma, Granite, Ministral, Fastweb, Minerva, Villanova)
        necessitano che il prompt venga avvolto in un template conversazionale
        prima della tokenizzazione.

        Returns:
            True se il model_id corrisponde a un modello chat-only.
        """
        return any(m in self._model_lower for m in CHAT_TEMPLATE_MODELS)

    def _decode_output(self, output_ids, input_ids) -> str:
        """
        Decodifica solo i token generati in modo robusto per tutti i tokenizer.

        Strategia: decodifica la sequenza completa e sottrai il prompt decodificato.
        Evita il problema di LlamaTokenizer che perde spazi su token isolati.
        Post-processing: rimuove artefatti byte-level BPE (Ġ→spazio, Ċ→newline).
        """
        full_decoded   = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        prompt_decoded = self.tokenizer.decode(input_ids,  skip_special_tokens=True)

        if full_decoded.startswith(prompt_decoded):
            decoded = full_decoded[len(prompt_decoded):]
        else:
            # Fallback: decodifica diretta dei new tokens
            input_len  = len(input_ids)
            new_tokens = output_ids[input_len:]
            decoded    = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Fix artefatti byte-level BPE (LlamaTokenizer / Mistral / Minerva)
        decoded = decoded.replace('\u0120', ' ')   # Ġ → spazio
        decoded = decoded.replace('\u010a', '\n')  # Ċ → newline
        decoded = re.sub(r' +', ' ', decoded)      # spazi multipli → uno

        return decoded.strip()

    # ── Generazione ───────────────────────────────────────────────────────────

    def generate(self, prompts, max_new_tokens: int = 15) -> List[str]:
        """
        Genera risposte per uno o piu' prompt in modalita' greedy.

        Applica automaticamente il chat template per i modelli chat-only e
        restituisce solo il testo generato (senza il prompt originale).

        Args:
            prompts: stringa singola o lista di stringhe contenenti i prompt.
            max_new_tokens: numero massimo di nuovi token da generare per ogni
                prompt (default 15, adatto a risposte estrattive brevi).

        Returns:
            Lista di stringhe contenenti solo le risposte generate, una per
            ciascun prompt fornito.
        """
        # Normalizza l'input: se e' una stringa singola, la avvolge in una lista
        if isinstance(prompts, str):
            prompts = [prompts]

        # Applica chat template per modelli chat-only: avvolge ogni prompt
        # nel formato conversazionale richiesto dal modello
        if self._needs_chat_template():
            prompts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in prompts
            ]

        # Tokenizzazione batch con padding a sinistra e troncamento
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_max_length,
        ).to(self.model.device)

        # Generazione greedy (do_sample=False) con repetition_penalty
        # per prevenire loop degenerativi. Minerva usa un valore piu' alto (1.3)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
               # repetition_penalty=1.1,
                repetition_penalty=1.3 if "minerva" in self._model_lower else 1.1,

                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decodifica ciascuna risposta: estrae solo i token generati,
        # rimuovendo il prompt originale dall'output
        results = []
        for i, output in enumerate(outputs):
            decoded = self._decode_output(
                output_ids=output,
                input_ids=inputs["input_ids"][i],
            )
            results.append(decoded)

        return results
