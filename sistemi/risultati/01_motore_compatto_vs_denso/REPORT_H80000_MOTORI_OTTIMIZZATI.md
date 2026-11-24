# H80000 Dense e STERA · Generazione con entrambi i motori ottimizzati

Entrambi i percorsi eseguono la funzione del checkpoint H80000 su MacBook Air M1, GPU Metal con MLX, pesi FP32 già caricati, attenzione SDPA, cache KV FP32 da 512 token, decode compilato e batch 1. Il Dense carica il file Safetensors verificato e usa QKV fusa e moltiplicazioni MLX. STERA usa le proiezioni Metal sulle coordinate compatte. Entrambi elaborano il prompt in parallelo.

Sono state verificate sei configurazioni di prefill per motore su un input di taratura, da sequenziale a blocchi da 256. La selezione usa il tempo totale a caldo e richiede parità greedy. Configurazioni selezionate: Dense a blocchi da 256 token e STERA a blocchi da 128 token. Sono state poi congelate e misurate su sei altri input da 256 token, con ordine alternato. Le sei coppie coincidono per tutti i 101 ID osservati, incluso il primo token. I 100 passi successivi sono la finestra di decode.

| Motore | Condizione | Primo token (ms) | Decode (ms/passo) | Decode (token/s) | Totale (s) |
|---|---|---:|---:|---:|---:|
| H80000 Dense FP32 | Prima richiesta | 64.2 | 9.07 | 110.25 | 0.971 |
| H80000 Dense FP32 | A caldo | 32.4 | 9.49 | 105.42 | 0.985 |
| STERA H80000 FP32 | Prima richiesta | 87.6 | 5.54 | 180.52 | 0.644 |
| STERA H80000 FP32 | A caldo | 71.5 | 6.65 | 150.80 | 0.736 |

I valori sono mediane. Sul decode a caldo, la mediana dei rapporti appaiati STERA/Dense equivale a una variazione del tempo per passo pari a **-29.4%**. La scelta riguarda le varianti MLX/Metal provate. Le misure includono l'esecuzione e la sincronizzazione del passo, con costruzione del modello e caricamento esclusi. Il download di Qwen era attivo sul Mac durante la campagna. L'alternanza riduce il rischio di favorire sistematicamente un motore, mentre resta da ripetere la misura con il sistema inattivo per verificare il margine.

Questo esperimento quantifica il costo dell'implementazione sullo stesso modello durante la generazione completa. Il confronto pubblico con TinyStories resta una misura distinta fra modelli e runtime completi.

[Misure grezze appaiate](misure_appaiate.jsonl) · [Risultati della taratura](taratura_completa.json) · [Riepilogo](riepilogo_appaiato.json) · [Script di misura](../../benchmark/misura_dense_stera.py) · [Prefill Dense](../../motori/prefill_dense.py)
