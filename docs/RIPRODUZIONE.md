# Riproduzione completa

Questa guida porta da zero ai numeri della tesi. Ogni passo indica cosa produce e come verificarlo.
I file pesanti vanno in una cartella `esterni/` (ignorata da git) nella radice del repository.

## 0. Ambiente

| Parte | Hardware usato nella tesi | Software |
|---|---|---|
| Preparazione dati, training, valutazione | server con 2 × NVIDIA P100 16 GB, Slurm | Python 3.12, PyTorch, tokenizers 0.20, transformers 4.46 (`requisiti_addestramento.txt`) |
| Valutazione di TinyStories-8M, statistiche, figure | Mac con Apple M1, 16 GB | Python 3.9+, PyTorch 2.8 su CPU |
| Misure di sistema | Mac con Apple M1, 16 GB | Python 3.11, MLX 0.32.4, PyTorch 2.15 MPS, Transformers 5 (`requisiti_sistemi.txt`) |
| Figure e tesi | qualunque | matplotlib 3.9.4 (figure identiche al pixel), [Tectonic](https://tectonic-typesetting.github.io) |

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requisiti_addestramento.txt
```

## 1. Dati

Scaricare il corpus e il tokenizer alle revisioni usate nella tesi:

```bash
mkdir -p esterni
curl -L -o esterni/TinyStoriesV2-GPT4-train.txt \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/TinyStoriesV2-GPT4-train.txt
curl -L -o esterni/tokenizer.json \
  https://huggingface.co/roneneldan/TinyStories-33M/resolve/2ad0a164221b7c4d21cac7c46aec74f6f98dbfc8/tokenizer.json
shasum -a 256 esterni/TinyStoriesV2-GPT4-train.txt esterni/tokenizer.json
```

Le impronte attese sono `6418d412…dec90443` (corpus, 2.227.753.162 byte) e `f6ed3d30…cfeff5` (tokenizer).

Tokenizzazione e split:

```bash
export STERA_CORPUS=esterni/TinyStoriesV2-GPT4-train.txt
export STERA_TOKENIZER=esterni/tokenizer.json
export STERA_DATI=esterni/corpus_tokenizzato
python addestramento/prepara_dati.py
```

Lo script assegna ogni racconto a TRAIN, DEV o TEST in base all'hash SHA-256 del testo normalizzato (spazi uniformati, minuscole), modulo 10.000: valori 0–99 a TEST, 100–199 a DEV, il resto a TRAIN. I 2.000 racconti di `addestramento/dati/racconti_riservati_dev_2000.jsonl` sono forzati in DEV. Racconti identici finiscono sempre nella stessa partizione.

Poi si scrive il contratto dei dati, che il trainer registra nei checkpoint:

```bash
python addestramento/sigilla_dati.py
```

**Verifica:** `esterni/corpus_tokenizzato/MANIFEST.json` deve coincidere con `addestramento/dati/MANIFEST_DATI.json` nei campi `pack_sha256` (`5ae58177…`), `index_sha256` (`bbadffc0…`), `split_stories` (2.660.865 / 29.279 / 27.351) e `split_target_tokens`. Il file `CONTRACT` deve coincidere con `addestramento/dati/CONTRACT` (lo script lo controlla e stampa `coincide con la tesi: True`).

## 2. Training di STERA 30M

```bash
export STERA_RUN=esterni/checkpoint
sbatch addestramento/addestra_stera30m.sh        # oppure: bash addestramento/addestra_stera30m.sh
```

Configurazione: 6 layer, guadagni per riga e colonna, termine low-rank 96; batch da 64 sequenze di al più 512 token, AdamW ($\beta_1=0{,}9$, $\beta_2=0{,}95$, decadimento 0,1), learning rate con warmup di 300 passi fino a $6\cdot10^{-4}$, costante fino al passo 64.000, discesa lineare fino a $6\cdot10^{-5}$ al passo 80.000 e poi costante. Un checkpoint completo (pesi, momenti di Adam, cursore dei dati) è salvato ogni 2.000 passi; lo script riprende dall'ultimo se viene rilanciato.

Tempo indicativo: circa 12.000 token/s su due P100, cioè circa 35 ore per 122.000 aggiornamenti. La corsa della tesi è stata eseguita in tre job consecutivi dello stesso script.

Al termine, il checkpoint finale `esterni/checkpoint/TS68V2_STEP122000.ckpt.gz` va copiato nel percorso usato da tutti i passi successivi (chi non riaddestra lo scarica invece dalla Release del repository, vedi `pesi/LEGGIMI.md`):

```bash
cp esterni/checkpoint/TS68V2_STEP122000.ckpt.gz pesi/stera30m_step122000.ckpt.gz
shasum -a 256 pesi/stera30m_step122000.ckpt.gz   # 418becfb…6ed0887
```

**Verifica rapida senza GPU.** La NLL DEV del checkpoint finale si ottiene senza addestrare: il trainer valuta prima del primo passo, quindi con `--steps 0` stampa solo `VAL`. Su una macchina senza `torchrun` funzionante basta impostare a mano le variabili del processo unico:

```bash
RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29511 \
  python addestramento/ts68_v3.py --data "$STERA_DATI" --output /tmp/stera_val \
  --ckpt pesi/stera30m_step122000.ckpt.gz --sha256 418becfbcaddcefd56f828f295138bbc865f9405a9060930c060ff7e6ced0887 \
  --layers 6 --gains --rank 96 --steps 0 --ctx 512 --nval 512
```

Il risultato atteso è `VAL step=122000 nll=1.152155`. Allo stesso modo, 20 passi da zero su CPU riproducono esattamente le prime righe del log (perdita 9,34693 al passo 20; vedi `addestramento/verifica_20261124/`).

**Verifica:** le righe `VAL` del log devono seguire `addestramento/log_addestramento_0_122k.txt` (NLL DEV 10,928 al passo 0, 1,172 a 80.000, 1,152 a 122.000). Il checkpoint finale della tesi ha impronta `418becfb…6ed0887`.

## 3. Prompt di prova

Il file `valutazione/protocollo/incipit_test300.json` è già nel repository (SHA-256 `95bcffdb…032ec`). Per rigenerarlo:

```bash
python valutazione/protocollo/seleziona_incipit.py     # usa STERA_DATI e STERA_TOKENIZER
```

## 4. NLL e generazione dei tre modelli

Scaricare i due riferimenti in `esterni/` alle revisioni indicate in `sistemi/modelli/manifest_tinystories.json` (8M: `8612e3b1…`, 33M: `2ad0a164…`), per esempio con `huggingface-cli download roneneldan/TinyStories-33M --revision 2ad0a164… --local-dir esterni/TinyStories-33M`.

```bash
cd valutazione/protocollo
python valuta_modello.py stera30m --checkpoint ../../pesi/stera30m_step122000.ckpt.gz \
       --tokenizer ../../esterni/tokenizer.json --device cuda --out stera30m.jsonl
python valuta_modello.py hf --modello ../../esterni/TinyStories-33M --device cuda --out tinystories33m.jsonl
python valuta_modello.py hf --modello ../../esterni/TinyStories-8M  --device cpu  --out tinystories8m.jsonl
```

Lo stesso script calcola la NLL dei 300 racconti di riferimento (57.841 token target) e genera una continuazione per prompt con top-$k$ 40, temperatura 0,75, al più 180 nuovi token. Il generatore casuale è su CPU con il seme dell'prompt, quindi le continuazioni non dipendono dal dispositivo.

**Verifica:** le continuazioni coincidono token per token con quelle in `valutazione/generazioni/`; rieseguendo i tre modelli su P100 coincidono tutte le 900 continuazioni; su CPU coincidono 899 su 900 (il prompt F156 di TinyStories-33M diverge dal token 84 per l'arrotondamento). Le NLL medie differiscono al più di qualche centomillesimo di nat.

## 5. Giudizio cieco

```bash
python valutazione/giudizio_cieco/prepara_insieme_cieco.py
```

crea `racconti_anonimi_900.json`, i 30 blocchi da 30 testi e `abbinamento_modelli.json`. Il giudice riceve solo i blocchi e `rubrica.txt`; per ogni testo restituisce i voti di grammatica, coerenza e trama (1–10), la categoria S/P/N e una motivazione. Nella tesi il giudice è Claude Opus 5.5, che ha valutato i 900 testi in un'unica sessione senza accesso all'abbinamento; i suoi giudizi sono in `giudizi_blocchi/`.

```bash
python valutazione/giudizio_cieco/aggrega_e_decodifica.py
```

verifica che tutti i 900 testi siano stati giudicati in modo coerente con la rubrica, apre l'abbinamento e calcola quote con intervalli di Wilson, differenze appaiate con bootstrap (25.000 repliche), test di McNemar con correzione di Holm e differenze di NLL con bootstrap (10.000 repliche). Il risultato è `valutazione/risultati_3modelli.json`; l'impronta dei giudizi è `40760aca…a5e8ea`.

## 6. Misure di sistema (Mac con Apple Silicon)

I motori in `sistemi/motori/` leggono i pesi da `sistemi/modelli/`:

| File | Contenuto | Come si ottiene |
|---|---|---|
| `H122000_fp32_weights.npz` | coordinate e tensori FP32 del checkpoint finale | `python sistemi/benchmark/esporta_pesi.py pesi/stera30m_step122000.ckpt.gz 418becfb…6ed0887 H122000` |
| `H122000_DENSE_FP32.safetensors` | matrici dense equivalenti | `python sistemi/benchmark/esporta_denso.py` |

Tutte le misure della tesi usano il checkpoint finale. I motori leggono i pesi indicati dalla variabile `STERA_PESI` (predefinito `H122000`).

```bash
python3.11 -m venv .venv-sistemi && .venv-sistemi/bin/pip install -r requisiti_sistemi.txt
sh sistemi/benchmark/confronto_appaiato.sh                 # una campagna: motore compatto vs denso, 6 input appaiati
sh sistemi/benchmark/TinyStories/esegui_confronto_tesi.sh  # una campagna: STERA 30M e i due TinyStories su MPS
```

I numeri della tesi vengono da cinque campagne complete sui pesi del checkpoint finale, ciascuna avviata quando la media di carico a un minuto scende sotto 1,5 e dopo 60 s di pausa, con il Mac alimentato dalla rete e senza altre applicazioni attive:

```bash
SOGLIA_CARICO=1.5 PAUSA=60 STERA_PESI=H122000 PYTHON=.venv-sistemi/bin/python \
  sh sistemi/benchmark/campagne_tempi.sh sistemi/risultati/campagne_20261124/protocollo 5
.venv-sistemi/bin/python sistemi/benchmark/riepiloga_campagne.py sistemi/risultati/campagne_20261124/protocollo
```

Il riepilogo verifica prima che i token coincidano fra i due motori, con l'implementazione eager dei TinyStories e fra tutte le campagne, poi calcola le mediane. I token prodotti non dipendono dalla macchina; i tempi sì, e su un altro Mac ci si aspettano valori diversi.

I risultati della tesi, con i report che li descrivono, sono in `sistemi/risultati/`:

- `campagne_20261124/protocollo/`: le cinque campagne sul checkpoint finale usate nella tesi e `riepilogo.json` (decode appaiato −34,8%, tempi e picchi dei tre modelli); `protocollo_H122000_sotto_carico/` contiene cinque campagne sugli stessi pesi eseguite con il sistema carico, `protocollo_H80000/` e `preliminari/` campagne precedenti sui pesi a 80.000 aggiornamenti;
- `01_motore_compatto_vs_denso/`: misure appaiate precedenti e attribuzione del tempo per operatore;
- `02_tre_modelli_mac/`: TTFT, decode e picco di memoria dei tre modelli;
- `03_f10_ssd_qwen/`: lettura validata dall'SSD e Qwen3.5-9B; le prove F10 qui contenute sono precedenti alla sezione 6.1 e non sono usate nella tesi (motore `sistemi/motori/stera_independent_fp32_transposed_jit_v28.py`, lettura dal disco `evidenze_f10/ssd_validato/bench_ssd_f10.py`) (pesi compatti 4,003 GB contro 18,597 GB densi), lettura validata dall'SSD, Qwen3.5-9B con caricamento dei layer dal disco, eseguito con [SWLP](https://github.com/Rishav-Upadhaya/SWLP) al commit `3941963812850f9ad6821f3bec78754d695143ac` (script `prepara_swlp.py`, `misura_swlp.py`, `riepiloga_swlp.py`); modello `Qwen/Qwen3.5-9B` scaricato da Hugging Face (revisione registrata nei metadati di download: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`).

L'archivio senza perdita dei pesi FP32 del checkpoint finale (102.352.624 byte, non usato nella tesi) si ottiene con `sistemi/benchmark/archivio_senza_perdita/steraf32_byteplane.py` e si ripristina bit per bit con `restore_lossless_archive.py`.

### 6.1 Modello fittizio STERA F10 (confronto di risorse con Qwen3.5-9B)

F10 non è addestrato: serve solo a confrontare le risorse di esecuzione. Si genera dal checkpoint finale con uno script deterministico (file identico bit per bit su ogni macchina):

```bash
python sistemi/f10/genera_f10.py sistemi/modelli/H122000_fp32_weights.npz sistemi/modelli/F10_H122000.npz
shasum -a 256 sistemi/modelli/F10_H122000.npz   # c56d2f90fab47efcdb250b231394b531c0b11944faf3dffc5e6e166ba4fa9162
```

Decode al variare del contesto (cache riempita con valori di prova, tre esecuzioni per contesto, ciascuna avviata con carico sotto 1,5), prova di controllo con la cache riempita dal modello e verifica termica:

```bash
cd sistemi/benchmark
CONTESTI="128:fp32 512:fp32 2000:fp32 8000:fp32 16000:fp32 32000:fp32 32000:fp16 64000:fp16" ./campagne_f10_contesto.sh
STERA_PESI=H122000 python f10_controllo_reale.py --token 8000 --out ../risultati/campagne_20261127/f10_contesto/controllo_reale_8000.json
STERA_PESI=H122000 python f10_contesto_prolungato.py --contesto 8000 --secondi 800 --out ../risultati/campagne_20261127/f10_contesto/verifica_termica_8000.json
```

Risultati in `sistemi/risultati/campagne_20261127/f10_contesto/`. Le cartelle `diagnostica_*` contengono misure precedenti non usate nella tesi (modello generato in GPU; cache copiata a ogni passo).

## 7. Figure e tesi

```bash
cd tesi/scripts_figure
python fig_pretraining.py && python fig_narrativa.py && python fig_sistemi.py
cd .. && tectonic -X compile main.tex
```

Per lavorare sul testo con anteprima automatica: `python tesi/anteprima/server.py` e aprire `http://localhost:8765`.

## 8. Componenti delle proiezioni e studi di sistema aggiuntivi

```bash
cd valutazione/protocollo
python valuta_modello.py stera30m --checkpoint ../../pesi/stera30m_step122000.ckpt.gz --tokenizer ../../esterni/tokenizer.json \
       --rango-bit 8 --rango-schema mlx --solo-nll --out /tmp/rango8.jsonl      # low-rank a 8 bit
#   --rango-ridotto K (K = 64 ... 0), --ablazione scale1|guadagni1|scale8|guadagni8|piccole8
```

Giudizi ciechi delle varianti: `valutazione/varianti/` (insiemi anonimizzati, abbinamenti, `aggrega*.py`).
Tempi: `sistemi/benchmark/campagne_mlx.sh` (confronto alla pari), `campagne_contesto.sh` (lunghezza dell'ingresso), `campagne_prefill.sh` (prefill con espansione), `campagne_rango8.sh` (low-rank a 8 bit); motori `sistemi/motori/gptneo_denso_mlx.py`, `prefill_espansione.py`, `stera_rango8_mlx.py`.
