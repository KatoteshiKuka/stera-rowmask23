# Verifica di riproducibilità (24 novembre 2026)

Ogni passo della tesi è stato rieseguito partendo solo dal repository e da fonti pubbliche: il corpus e il tokenizer scaricati da Hugging Face alle revisioni fissate, i due TinyStories alle revisioni di `sistemi/modelli/manifest_tinystories.json` e il checkpoint finale di STERA 30M. La tabella riporta l'esito di ogni voce.

| Voce | Dove è stata eseguita | Esito |
|---|---|---|
| Download di corpus, tokenizer, TinyStories-8M e 33M | Mac | SHA-256 identiche a quelle attese (`6418d412…`, `f6ed3d30…`, `22c355bf…`, `41316d3b…`) |
| Tokenizzazione e split (`prepara_dati.py`) | Mac, CPU | `MANIFEST.json` identico in tutti i campi (pack `5ae58177…`, indice `bbadffc0…`, conteggi per split, token bersaglio) tranne percorso e durata |
| Contratto dei dati (`sigilla_dati.py`) | Mac, CPU | `CONTRACT` identico (`b8a68cad…`); il checkpoint finale lo accetta |
| NLL DEV del checkpoint finale (`ts68_v3.py --steps 0`) | Mac, CPU | 1,152155, identica al log |
| Motore di addestramento: 20 passi da zero | Mac, CPU | NLL DEV al passo 0 (10,927510), perdita al passo 20 (9,34693), norma del gradiente (5,7446), tasso di apprendimento e numero di etichette identici al log; log in `addestramento/verifica_20261124/` |
| Valutazione TEST300, TinyStories-8M | Mac, CPU | 300/300 continuazioni identiche; NLL 1,515349 (tesi 1,515349) |
| Valutazione TEST300, STERA 30M | Mac, CPU (originale su P100) | 300/300 continuazioni identiche; NLL 1,173975 (tesi 1,173997) |
| Valutazione TEST300, TinyStories-33M | Mac, CPU (originale su P100) | 299/300 continuazioni identiche; NLL 1,324450 (tesi 1,324456). L'unica differenza (incipit F156) inizia al token 84 ed è compatibile con un diverso arrotondamento fra CPU e GPU nel campionamento |
| Aggregazione del giudizio cieco (`aggrega_e_decodifica.py`) | Mac | `risultati_3modelli.json` identico |
| Esportazione dei pesi (FP32 e INT8) | Mac | payload di 121.384.728 e 74.240.928 byte, come nella tesi; errore relativo INT8 0,01461 |
| Archivio senza perdita | Mac | rifatto sul checkpoint finale, 102.352.624 byte; ripristino bit per bit su tutti i 267 tensori. Un file `.npz` ricostruito da `restore_lossless_archive.py` contiene gli stessi tensori ma può avere un'impronta diversa, perché il contenitore zip registra metadati propri |
| Figure | Mac, matplotlib 3.9.4 | identiche al pixel; con matplotlib 3.11 differiscono di 1–2 pixel per il rendering del testo |
| Tempi e memoria (Capitolo 8) | Mac M1 | rieseguiti sul checkpoint finale, vedi sotto |

## Tempi, archivio e prototipo F10 sul checkpoint finale

Tutti i risultati di sistema della tesi si riferiscono al checkpoint finale (122.000 aggiornamenti, pesi `H122000_*`, SHA-256 dei pesi FP32 `864d03dc…`). I motori leggono il checkpoint indicato da `STERA_PESI`, con `H122000` come valore predefinito.

- **Tempi.** Cinque campagne consecutive (`sistemi/benchmark/campagne_tempi.sh`), con una media di carico fra 2,7 e 4,6 registrata in `condizioni.log`. In tutte le 30 coppie i due motori producono gli stessi token, identici fra le campagne; i TinyStories coincidono con l'implementazione eager. La variazione mediana del decode è −35,9%, con mediane per campagna fra −35,3% e −36,4%. Riepilogo: `sistemi/risultati/campagne_20261124/protocollo/riepilogo.json`.
- **Forma densa equivalente.** `esporta_denso.py` produce `H122000_DENSE_FP32.safetensors` (330.885.120 byte di payload, verifica bit per bit dopo il salvataggio).
- **Archivio senza perdita.** 102.352.624 byte sul checkpoint finale, con ripristino bit per bit di tutti i 267 tensori (sui pesi a 80.000 aggiornamenti l'archivio era di 102.298.136 byte).
- **Prototipo F10.** Inizializzato dal checkpoint finale: 9,89 e 9,73 token/s in due esecuzioni (`03_f10_ssd_qwen/evidenze_f10/inferenza_H122000/`). La prova di lettura dall'SSD misura volumi di byte che dipendono solo dalla forma del modello, quindi vale per qualunque checkpoint.

Le campagne precedenti sui pesi a 80.000 aggiornamenti restano in `campagne_20261124/protocollo_H80000/` e `preliminari/`, con −36,1% e fra −31,3% e −36,5% di variazione del decode.

## Voci non rieseguite

- **Addestramento completo e valutazione su GPU.** Richiedono il server con le P100; i job sono pronti ma non ancora eseguiti. La verifica a 20 passi e la NLL DEV su CPU controllano il motore di addestramento e il checkpoint senza riaddestrare.
- **Qwen3.5-9B** (`03_f10_ssd_qwen/`): non rieseguito (richiede il download di circa 18 GB); non dipende da STERA.
