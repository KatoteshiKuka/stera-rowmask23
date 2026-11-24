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
| Archivio senza perdita | Mac | 102.298.136 byte; ripristino bit per bit su tutti i 267 tensori. Il file `.npz` ripristinato ha un'impronta diversa dall'originale perché il contenitore zip registra metadati propri, mentre i tensori coincidono |
| Figure | Mac, matplotlib 3.9.4 | identiche al pixel; con matplotlib 3.11 differiscono di 1–2 pixel per il rendering del testo |
| Tempi e memoria (Capitolo 8) | Mac M1 | token identici in tutte le esecuzioni; tempi da cinque campagne ripetute, vedi sotto |

## Tempi

I token generati sono deterministici e coincidono in ogni esecuzione con quelli registrati. I tempi dipendono dallo stato della macchina: per questo la tesi riporta le mediane di cinque campagne complete (`sistemi/benchmark/campagne_tempi.sh`), ciascuna avviata quando la media di carico a un minuto scende sotto 1,5 e dopo 60 s di pausa. Il registro `condizioni.log` riporta il carico all'avvio effettivo, dopo la pausa (fra 1,7 e 2,3). Il riepilogo è in `sistemi/risultati/campagne_20261124/protocollo/riepilogo.json`.

Le cinque campagne danno una variazione mediana del decode di −36,1%, con mediane per campagna fra −35,8% e −36,5% e tutte le 30 coppie fra −34,3% e −37,8%. Tre campagne preliminari, eseguite senza controllo del carico, con media di carico fra 2,2 e 3,6 e la batteria in ricarica (`preliminari/`), danno −35,8%, −31,3% e −36,5% con gli stessi token. Le misure precedenti in `01_motore_compatto_vs_denso/` e `02_tre_modelli_mac/` (−26,9%, sei sole coppie) sono state sostituite nella tesi da queste.

Le misure di tempo usano i pesi del checkpoint a 80.000 aggiornamenti della stessa corsa (`H80000_*`), che hanno la stessa forma e lo stesso numero di byte del checkpoint finale; i token di riferimento registrati sono quelli prodotti da questi pesi.

## Voci non rieseguite

- **Addestramento completo e valutazione su GPU.** Richiedono il server con le P100; i job sono pronti ma non ancora eseguiti. La verifica a 20 passi e la NLL DEV su CPU controllano il motore di addestramento e il checkpoint senza riaddestrare.
- **Prototipo F10, lettura da SSD e Qwen3.5-9B** (`03_f10_ssd_qwen/`): non rieseguiti (richiedono il download di circa 18 GB del modello Qwen).
