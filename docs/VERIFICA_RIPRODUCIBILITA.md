# Verifica di riproducibilità (24 novembre 2026)

Ogni passo della tesi è stato rieseguito partendo solo dal repository e da fonti pubbliche: il corpus e il tokenizer scaricati da Hugging Face alle revisioni fissate, i due TinyStories alle revisioni di `sistemi/modelli/manifest_tinystories.json` e il checkpoint finale di STERA 30M. La tabella riporta l'esito di ogni voce.

| Voce | Dove è stata eseguita | Esito |
|---|---|---|
| Download di corpus, tokenizer, TinyStories-8M e 33M | Mac | SHA-256 identiche a quelle attese (`6418d412…`, `f6ed3d30…`, `22c355bf…`, `41316d3b…`) |
| Tokenizzazione e split (`prepara_dati.py`) | Mac CPU e server con P100 | `MANIFEST.json` identico in tutti i campi (pack `5ae58177…`, indice `bbadffc0…`, conteggi per split, token bersaglio) tranne percorso e durata |
| Contratto dei dati (`sigilla_dati.py`) | Mac, CPU | `CONTRACT` identico (`b8a68cad…`); il checkpoint finale lo accetta |
| NLL DEV del checkpoint finale (`ts68_v3.py --steps 0`) | Mac CPU e P100 | 1,152155 su entrambi, identica al log |
| Motore di addestramento: 20 passi da zero | Mac, CPU | NLL DEV al passo 0 (10,927510), perdita al passo 20 (9,34693), norma del gradiente (5,7446), tasso di apprendimento e numero di etichette identici al log; log in `addestramento/verifica_20261124/` |
| Motore di addestramento: 200 passi da zero con `sbatch addestra_stera30m.sh` | server Slurm, 2 × P100 | NLL DEV al passo 0 e righe del log identiche fino al passo 120, con le stesse etichette a ogni passo; dal passo 140 differenze dell'ordine di 10⁻⁵ sulla perdita (3,71362 contro 3,71367 al passo 200), dovute all'ordine non deterministico delle somme su GPU. Log in `addestramento/verifica_20261124/` |
| Valutazione TEST300, TinyStories-8M | Mac CPU e server con P100 | 300/300 continuazioni identiche su CPU e su GPU; NLL 1,515349 (CPU) e 1,515352 (GPU), tesi 1,515349 |
| Valutazione TEST300, STERA 30M | Mac CPU e server con P100 | 300/300 continuazioni identiche su CPU e su GPU; NLL 1,173975 (CPU) e 1,173986 (GPU), tesi 1,173997 |
| Valutazione TEST300, TinyStories-33M | Mac CPU e server con P100 | su GPU 300/300 continuazioni identiche, NLL 1,324452 (tesi 1,324456); su CPU 299/300, con l'incipit F156 che diverge dal token 84 per l'arrotondamento della CPU |
| Aggregazione del giudizio cieco (`aggrega_e_decodifica.py`) | Mac | `risultati_3modelli.json` identico |
| Esportazione dei pesi FP32 | Mac | payload di 121.384.728 byte, come nella tesi |
| Archivio senza perdita | Mac | rifatto sul checkpoint finale, 102.352.624 byte; ripristino bit per bit su tutti i 267 tensori. Un file `.npz` ricostruito da `restore_lossless_archive.py` contiene gli stessi tensori ma può avere un'impronta diversa, perché il contenitore zip registra metadati propri |
| Figure | Mac, matplotlib 3.9.4 | identiche al pixel; con matplotlib 3.11 differiscono di 1–2 pixel per il rendering del testo |
| Tempi e memoria (Capitolo 8) | Mac M1 | rieseguiti sul checkpoint finale, vedi sotto |

## Tempi, archivio e prototipo F10 sul checkpoint finale

Tutti i risultati di sistema della tesi si riferiscono al checkpoint finale (122.000 aggiornamenti, pesi `H122000_*`, SHA-256 dei pesi FP32 `864d03dc…`). I motori leggono il checkpoint indicato da `STERA_PESI`, con `H122000` come valore predefinito.

- **Tempi.** Cinque campagne (`sistemi/benchmark/campagne_tempi.sh`), ciascuna avviata quando la media di carico a un minuto è scesa sotto 1,5, seguita da 60 s di pausa; il carico all'avvio effettivo è registrato in `condizioni.log` (fra 1,25 e 1,66). In tutte le 30 coppie i due motori producono gli stessi token, identici fra le campagne; i TinyStories coincidono con l'implementazione eager. La variazione mediana del decode è −34,8%, con mediane per campagna fra −33,7% e −36,0%. Riepilogo: `sistemi/risultati/campagne_20261124/protocollo/riepilogo.json`. Cinque campagne sugli stessi pesi con il sistema carico (media fra 2,7 e 4,6, `protocollo_H122000_sotto_carico/`) danno −35,9% sul decode appaiato ma tempi assoluti più lenti.
- **Forma densa equivalente.** `esporta_denso.py` produce `H122000_DENSE_FP32.safetensors` (330.885.120 byte di payload, verifica bit per bit dopo il salvataggio).
- **Archivio senza perdita.** 102.352.624 byte sul checkpoint finale, con ripristino bit per bit di tutti i 267 tensori (sui pesi a 80.000 aggiornamenti l'archivio era di 102.298.136 byte).
- **Prototipo F10.** Inizializzato dal checkpoint finale: 12,29 e 12,33 token/s in due esecuzioni avviate con media di carico sotto 1,5 (`03_f10_ssd_qwen/evidenze_f10/inferenza_H122000/`). La prova di lettura dall'SSD misura volumi di byte che dipendono solo dalla forma del modello, quindi vale per qualunque checkpoint.

Le campagne precedenti sui pesi a 80.000 aggiornamenti restano in `campagne_20261124/protocollo_H80000/` e `preliminari/`, con −36,1% e fra −31,3% e −36,5% di variazione del decode.

## Voci non rieseguite

- **Addestramento completo.** Non è stato ripetuto (circa 35 ore su due P100). Il motore di addestramento è verificato con 20 passi da zero su CPU, con 200 passi avviati con `sbatch` sulle due P100 e con la NLL DEV del checkpoint finale.
- **Qwen3.5-9B** (`03_f10_ssd_qwen/`): non rieseguito (richiede il download di circa 18 GB); non dipende da STERA.
