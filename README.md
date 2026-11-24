# STERA – parametrizzazione strutturata RowMask23 per Transformer

Repository della tesi di laurea magistrale in Ingegneria Informatica (Università degli Studi Roma Tre, a.a. 2025/2026)

**STERA: Progettazione e valutazione di una parametrizzazione strutturata dei pesi per modelli linguistici Transformer**
Andrea Nardoni — relatore Prof. Fabio Gasparetti

Il PDF della tesi è in [`tesi/tesi_STERA_Nardoni.pdf`](tesi/tesi_STERA_Nardoni.pdf).

---

## In breve

**RowMask23** è una famiglia di matrici $128\times128$ descritte da **3.176 coordinate** anziché da 16.384 coefficienti indipendenti:

- una **diagonale** libera (128 valori);
- una **base XOR** condivisa lungo le diagonali generalizzate $\{(i, i\oplus s)\}$ (127 valori);
- **23 correzioni per riga** in posizioni fisse $\sigma(i,k) = 1 + ((13k+7i) \bmod 127)$ (2.921 valori).

La mappa coordinate → matrice è lineare e iniettiva, il gradiente rispetto alle coordinate ha forma chiusa e il prodotto con un vettore si calcola con la trasformata di Walsh–Hadamard, senza ricostruire la matrice.

**STERA 30M** è un Transformer decoder-only a 6 strati (dimensione 768, 16 teste, feed-forward 3.072, vocabolario GPT-Neo da 50.257 token) in cui tutte le proiezioni, compresa la matrice di embedding condivisa con la testa di uscita, sono mosaici di blocchi RowMask23, con guadagni per riga e colonna e un termine a basso rango (rango 96). Conta **30.346.182 parametri allenabili** (82.721.280 valori nella forma densa equivalente) ed è addestrato da zero su TinyStoriesV2-GPT4 per **122.000 aggiornamenti** (1.529.514.422 token bersaglio) su due GPU NVIDIA P100.

## Risultati principali

Su 300 racconti TEST, con gli stessi incipit, lo stesso tokenizzatore e la stessa procedura di generazione per i tre modelli:

| Modello | Parametri | NLL TEST | Racconti corretti (giudizio cieco) |
|---|---:|---:|---:|
| TinyStories-8M | 19.702.528 | 1,515 | 65/300 (21,7%) |
| **STERA 30M** | **30.346.182** | **1,174** | **132/300 (44,0%)** |
| TinyStories-33M | 68.514.048 | 1,324 | 145/300 (48,3%) |

- STERA 30M − TinyStories-8M: **+22,3 punti** (IC 95% [+15,3; +29,3], McNemar con Holm p < 10⁻⁸).
- STERA 30M − TinyStories-33M: **−4,3 punti** (IC 95% [−12,0; +3,7], p = 0,32, non significativa).
- Pesi compatti FP32: **121,385 MB**, contro 330,885 MB dell'espansione densa (−63,3%) e 274,056 MB di TinyStories-33M (−55,7%); 74,241 MB con le coordinate in INT8.
- Decode sullo stesso modello: **−36,1%** di tempo mediano per token del motore compatto rispetto al motore denso (Mac M1, MLX/Metal, 30 misure appaiate in cinque campagne), con token identici.

Dettagli e numeri completi: [`docs/RISULTATI.md`](docs/RISULTATI.md).

## Mappa del progetto

```
STERA-RowMask23/
├── README.md                     questa pagina
├── docs/
│   ├── RIPRODUZIONE.md           procedura completa, passo per passo
│   ├── RISULTATI.md              tutti i numeri della tesi e da dove vengono
│   ├── VERIFICA_RIPRODUCIBILITA.md   esito della riesecuzione di ogni passo
│   └── ARTEFATTI.md              impronte SHA-256 di dati, modelli e risultati
├── addestramento/
│   ├── prepara_dati.py           tokenizzazione del corpus e split TRAIN/DEV/TEST per hash
│   ├── sigilla_dati.py           contratto dei dati registrato nei checkpoint
│   ├── ts68_core.py              blocchi RowMask23, modello, lettura e scrittura dei checkpoint
│   ├── ts68_v3.py                trainer distribuito (PyTorch, 2 GPU) con guadagni e basso rango
│   ├── addestra_stera30m.sh      comando di addestramento 0 → 122.000 aggiornamenti
│   ├── log_addestramento_0_122k.txt   perdita TRAIN ogni 20 passi e NLL DEV ogni 2.000
│   └── dati/                     manifesto dello split, contratto dei dati, racconti riservati a DEV
├── valutazione/
│   ├── protocollo/
│   │   ├── protocollo_test300.json    descrizione completa del protocollo
│   │   ├── incipit_test300.json       i 300 incipit con racconti di riferimento e semi
│   │   ├── seleziona_incipit.py       selezione deterministica degli incipit
│   │   ├── valuta_modello.py          NLL + generazione, stesso codice per i tre modelli
│   │   └── esclusioni/                incipit usati durante lo sviluppo, esclusi dal TEST
│   ├── generazioni/              continuazioni dei tre modelli (300 ciascuno), con NLL per racconto
│   ├── giudizio_cieco/
│   │   ├── rubrica.txt                istruzioni e rubrica del giudice
│   │   ├── prepara_insieme_cieco.py   anonimizzazione e mescolamento dei 900 testi
│   │   ├── racconti_anonimi_900.json  i testi giudicati (T000–T899)
│   │   ├── giudizi_blocchi/           giudizi grezzi, 30 blocchi da 30 testi
│   │   ├── giudizi_900.json           tutti i giudizi
│   │   ├── abbinamento_modelli.json   corrispondenza T### → modello e incipit
│   │   └── aggrega_e_decodifica.py    statistiche: Wilson, bootstrap, McNemar, Holm, NLL
│   └── risultati_3modelli.json   risultato finale del confronto
├── sistemi/
│   ├── motori/                   motori MLX/Metal: prodotto diretto sulle coordinate e denso equivalente
│   ├── benchmark/                misure appaiate, attribuzione, TinyStories su MPS, esportazione dei pesi, archivio senza perdita
│   ├── modelli/                  tokenizer e manifesti (i pesi non sono versionati)
│   └── risultati/                dati grezzi e report delle misure del Capitolo 8
├── tesi/
│   ├── main.tex, capitoli/, sommario.tex, bibliografia.bib, TesiDiaUniroma3.cls
│   ├── figure/                   figure (PDF e sorgenti TikZ)
│   ├── scripts_figure/           script che generano i grafici dai dati
│   ├── dati_figure/              dati usati dai grafici
│   ├── anteprima/                server di anteprima che ricompila il PDF a ogni salvataggio
│   └── tesi_STERA_Nardoni.pdf    PDF compilato
├── pesi/                         (non versionata) checkpoint finale di STERA 30M
├── requisiti_addestramento.txt
└── requisiti_sistemi.txt
```

## Dalla tesi al codice

| Capitolo della tesi | Dove si trova nel repository |
|---|---|
| 4 – La parametrizzazione RowMask23 | `addestramento/ts68_core.py` (ricostruzione, gradiente), `sistemi/motori/stera4b_wht_*` (prodotto con Walsh–Hadamard) |
| 5 – STERA 30M | `addestramento/ts68_v3.py`, `addestramento/addestra_stera30m.sh` |
| 6 – Metodologia di validazione | `addestramento/prepara_dati.py`, `valutazione/protocollo/` |
| 7 – Apprendimento e qualità dei racconti | `addestramento/log_addestramento_0_122k.txt`, `valutazione/` |
| 8 – Memoria e tempi di esecuzione | `sistemi/` |
| Figure | `tesi/scripts_figure/` (grafici), `tesi/figure/tikz/` (schemi) |

## Riproduzione rapida

```bash
# 1. dati: scaricare il corpus e il tokenizer (vedi docs/RIPRODUZIONE.md), poi
export STERA_CORPUS=esterni/TinyStoriesV2-GPT4-train.txt
export STERA_TOKENIZER=esterni/tokenizer.json
export STERA_DATI=esterni/corpus_tokenizzato
python addestramento/prepara_dati.py && python addestramento/sigilla_dati.py

# 2. addestramento (2 GPU)
export STERA_RUN=esterni/checkpoint
bash addestramento/addestra_stera30m.sh

# 3. valutazione dei tre modelli
python valutazione/protocollo/valuta_modello.py stera30m --checkpoint esterni/checkpoint/TS68V2_STEP122000.ckpt.gz \
       --tokenizer $STERA_TOKENIZER --device cuda --out valutazione/generazioni/stera30m.jsonl
python valutazione/protocollo/valuta_modello.py hf --modello esterni/TinyStories-33M --out valutazione/generazioni/tinystories33m.jsonl

# 4. statistiche sul giudizio cieco
python valutazione/giudizio_cieco/aggrega_e_decodifica.py

# 5. tesi
cd tesi && tectonic -X compile main.tex
```

La procedura completa, con i tempi indicativi e le verifiche intermedie, è in [`docs/RIPRODUZIONE.md`](docs/RIPRODUZIONE.md).

## Licenze dei dati e dei modelli di riferimento

- Corpus TinyStories (`roneneldan/TinyStories`): licenza CDLA-Sharing-1.0.
- Checkpoint di riferimento `roneneldan/TinyStories-8M` e `roneneldan/TinyStories-33M`: scaricati da Hugging Face, con le licenze indicate nelle rispettive schede.
- La Figura 2.1 della tesi riproduce la Figura 1 di Vaswani et al. (2017), con attribuzione (vedi `tesi/FONTI_E_ATTRIBUZIONE_FIGURE.txt`).
