import re
import torch
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM


# Modelli con thinking mode attivo di default
THINKING_MODE_MODELS = ["qwen3.5", "qwen3-14"]

# Modelli che richiedono chat template obbligatorio (chat-only)
CHAT_TEMPLATE_MODELS = ["gemma-4", "granite-4.1", "ministral", "fastweb", "minerva","villanova"]

# Modelli che richiedono una classe specifica invece di AutoModelForCausalLM
MISTRAL3_MODELS = ["ministral-3"]


class LLM:
    """
    Drop-in replacement per modelli HuggingFace in locale.
    - Raw completion per la maggior parte dei modelli
    - Chat template obbligatorio per modelli chat-only
    - repetition_penalty=1.1 per tutti (previene loop degenerativi)
    - Output: solo la risposta generata, senza il prompt
    """

    def __init__(
        self,
        model_id: str,
        device=None,
        quantization_bits: Optional[int] = None,
        stop_list: Optional[List[str]] = None,
        model_max_length: int = 8192,
    ):
        self.model_id = model_id
        self.model_max_length = model_max_length
        self._model_lower = model_id.lower()

        print(f"Caricamento tokenizer: {model_id}")
        tokenizer_kwargs = dict(
            padding_side="left",
            truncation_side="left",
            model_max_length=model_max_length,
            trust_remote_code=True,
        )
        if self._is_mistral3():
            tokenizer_kwargs["fix_mistral_regex"] = True

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Tokenizer caricato.")

        print(f"Caricamento modello: {model_id}")
        model_kwargs = dict(
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        if self._is_mistral3():
            from transformers.models.mistral3 import Mistral3ForConditionalGeneration
            self.model = Mistral3ForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

        self.model.eval()
        print(f"Modello caricato su: {next(self.model.parameters()).device}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_mistral3(self) -> bool:
        return any(m in self._model_lower for m in MISTRAL3_MODELS)

    def _has_thinking_mode(self) -> bool:
        return any(m in self._model_lower for m in THINKING_MODE_MODELS)

    def _needs_chat_template(self) -> bool:
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
        Genera risposte per uno o più prompt.
        Output: lista di stringhe con SOLO la risposta generata (senza prompt).
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        # Applica chat template per modelli chat-only
        if self._needs_chat_template():
            prompts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in prompts
            ]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_max_length,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
               # repetition_penalty=1.1,
                repetition_penalty=1.3 if "minerva" in self._model_lower else 1.1,

                pad_token_id=self.tokenizer.eos_token_id,
            )

        results = []
        for i, output in enumerate(outputs):
            decoded = self._decode_output(
                output_ids=output,
                input_ids=inputs["input_ids"][i],
            )
            results.append(decoded)

        return results
