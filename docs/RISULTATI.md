# Risultati della tesi e loro provenienza

## Addestramento (Capitolo 7, Figura 7.1)

| Grandezza | Valore | Fonte |
|---|---|---|
| Aggiornamenti | 122.000 | `addestramento/log_addestramento_0_122k.txt` |
| Token bersaglio | 1.529.514.422 | idem, campo `labels` |
| NLL DEV iniziale | 10,928 | riga `VAL step=0` |
| NLL DEV al passo 80.000 | 1,172 | riga `VAL step=80000` |
| NLL DEV finale | 1,152 (perplessità 3,17) | riga `VAL step=122000` |

## Verosimiglianza sui 300 racconti TEST (Tabella 7.1)

57.841 token bersaglio, stesso tokenizzatore. Fonte: `valutazione/risultati_3modelli.json`, campo `nll`.

| Modello | NLL | Perplessità | Differenza da STERA 30M (IC 95%) |
|---|---:|---:|---|
| TinyStories-8M | 1,515 | 4,55 | +0,341 [+0,319; +0,365] |
| STERA 30M | 1,174 | 3,23 | — |
| TinyStories-33M | 1,324 | 3,76 | +0,150 [+0,115; +0,186] |

## Giudizio cieco (Tabelle 7.2–7.3, Figure 7.2–7.3)

900 continuazioni anonime, un giudice (Claude Opus 5.5), rubrica in `valutazione/giudizio_cieco/rubrica.txt`. Fonte: `valutazione/risultati_3modelli.json`.

| Modello | Corretti | IC Wilson 95% | S / P / N | Grammatica | Coerenza | Trama |
|---|---:|---|---|---:|---:|---:|
| TinyStories-8M | 65/300 (21,7%) | [17,4; 26,7] | 65 / 200 / 35 | 7,43 | 4,74 | 4,96 |
| STERA 30M | 132/300 (44,0%) | [38,5; 49,7] | 132 / 152 / 16 | 7,67 | 5,26 | 5,37 |
| TinyStories-33M | 145/300 (48,3%) | [42,7; 54,0] | 145 / 140 / 15 | 7,91 | 5,54 | 5,49 |

| Confronto appaiato | Differenza (punti) | IC 95% | Solo A / solo B | p Holm |
|---|---:|---|---|---:|
| STERA 30M − TinyStories-8M | +22,3 | [+15,3; +29,3] | 98 / 31 | 5,4·10⁻⁹ |
| STERA 30M − TinyStories-33M | −4,3 | [−12,0; +3,7] | 66 / 79 | 0,32 |
| TinyStories-33M − TinyStories-8M | +26,7 | [+19,7; +33,7] | 109 / 29 | 1,3·10⁻¹¹ |

## Memoria (Capitolo 8)

| Rappresentazione | Byte dei pesi |
|---|---:|
| TinyStories-8M FP32 | 78.810.112 |
| STERA 30M, basso rango U, V a 8 bit (formato MLX), resto FP32 | 86.096.664 |
| STERA 30M FP32 | 121.384.728 |
| TinyStories-33M FP32 | 274.056.192 |
| STERA 30M, espansione densa FP32 | 330.885.120 |

Fonti: conteggio dei parametri (`addestramento/ts68_v3.py`) e file in `sistemi/modelli/`.

Bilancio (Capitolo 4, Sezione 4.14): i blocchi RowMask23 garantiscono −80,61% sulla parte strutturata; per uno strato con rango $r=\kappa d$ il risparmio è $80{,}61\% - 1{,}5\kappa - 1{,}5/d$; STERA 30M ($\kappa = 1/8$) risparmia il 63,32% (30.346.182 valori contro 82.721.280). Componenti: coordinate 15.721.200, scale 4.950, guadagni 134.016, basso rango 12.865.536, posizioni/normalizzazioni/bias 1.620.480.

## Componenti delle proiezioni (Capitoli 4 e 7)

NLL sui 300 racconti TEST del modello finale con una componente modificata a posteriori (`valutazione/varianti/`):

| Variante | NLL | Racconti corretti |
|---|---:|---:|
| Modello della tesi (FP32) | 1,174 | 44,0% |
| Basso rango a 8 bit (gruppi da 32, formato del motore) | 1,174056 (+0,007%) | — |
| Basso rango a 8 bit (una scala per colonna) | 1,174302 (+0,03%) | 41,3% (−2,7 punti, p = 0,52) |
| Scale a 8 bit / guadagni a 8 bit | +0,02% / +0,17% | — |
| Scale, guadagni e basso rango a 8 bit | +0,21% | — |
| Scale riportate a 1 / guadagni riportati a 1 | 4,42 / 11,38 | — |
| Basso rango ridotto a 64 / 48 / 32 / 16 / 8 / 0 | 1,474 / 1,981 / 2,893 / 4,236 / 4,980 / 5,813 | 20,0% / 0% / 0% / 0% / 0% / 0% |

Errore relativo della matrice densa equivalente con U, V a 8 bit: 0,61% (una scala per colonna). Giudizi delle varianti: sessioni separate, stesso giudice e stessa rubrica.

## Tempi di esecuzione (Capitolo 8)

Mac con Apple M1, 16 GB, MLX/Metal, FP32, 256 token in ingresso, 100 passi di decode, pesi del checkpoint finale. Cinque campagne di misura, ciascuna avviata con media di carico a un minuto sotto 1,5 (`sistemi/benchmark/campagne_tempi.sh`); mediane a caldo su 30 coppie (6 input × 5 campagne). Fonte: `sistemi/risultati/campagne_20261124/protocollo/riepilogo.json`, prodotto da `sistemi/benchmark/riepiloga_campagne.py`.

| Motore | Primo token | Decode | Totale |
|---|---:|---:|---:|
| Denso equivalente | 28,8 ms | 130,26 token/s | 0,797 s |
| Compatto (coordinate) | 61,0 ms | 200,58 token/s | 0,559 s |

Variazione mediana appaiata del tempo per token in decode: −34,8% (da −18,0% a −39,6% sulle 30 coppie; mediane per campagna da −33,7% a −36,0%). Il denso emette prima il primo token e il compatto termina prima la richiesta in tutte le 30 coppie. I due motori producono gli stessi 101 token su tutti gli input, identici in tutte le campagne.

Confronto alla pari sullo stesso motore MLX (TinyStories nel motore denso `sistemi/motori/gptneo_denso_mlx.py`, token identici all'implementazione di riferimento), 3 input × 5 campagne (`sistemi/risultati/campagne_20261124/confronto_mlx/`):

| Modello | Primo token | Decode | Totale | Picco RAM |
|---|---:|---:|---:|---:|
| TinyStories-8M | 13,3 ms | 311,9 token/s | 0,334 s | 511 MiB |
| STERA 30M compatto | 100,9 ms | 201,0 token/s | 0,599 s | 413 MiB |
| TinyStories-33M | 20,8 ms | 158,0 token/s | 0,654 s | 879 MiB |
| STERA 30M denso | 29,5 ms | 130,1 token/s | 0,800 s | 569 MiB |

Lunghezza dell'ingresso da 32 a 384 token (`contesto_mlx/`): decode di STERA compatto 191–198 token/s, riduzione rispetto al denso costante fra −35,4% e −36,3%. Prefill (`prefill_mlx/`): con l'espansione al bisogno strato per strato il primo token su 256 token scende da 65,6 a 41,8 ms (denso 30,7). Basso rango a 8 bit (`rango8_mlx/`): decode da 198,9 a 241,9 token/s (−17,8% di tempo per token), primo token invariato; con espansione e basso rango a 8 bit la richiesta da 256 + 100 token dura 0,47 s.

Tre modelli con il motore standard dei TinyStories (STERA 30M su MLX/Metal, TinyStories su Transformers/PyTorch 2.15 MPS), 3 input × 5 campagne, mediane a caldo:

| Modello | Primo token | Decode | Totale | Picco RAM |
|---|---:|---:|---:|---:|
| TinyStories-8M | 22,9 ms | 149,84 token/s | 0,691 s | 1.526,1 MiB |
| STERA 30M | 99,4 ms | 202,41 token/s | 0,593 s | 413,1 MiB |
| TinyStories-33M | 31,7 ms | 90,56 token/s | 1,136 s | 1.493,3 MiB |

I token dei due TinyStories coincidono con quelli dell'implementazione eager in ogni campagna.

Prototipo F10 e Qwen3.5-9B: `sistemi/risultati/03_f10_ssd_qwen/` (F10 inizializzato dal checkpoint finale, decode 12,29 e 12,33 token/s in `evidenze_f10/inferenza_H122000/`; pesi compatti 4,003 GB contro 18,597 GB densi; lettura sequenziale dall'SSD 1,685 s contro 7,821 s; Qwen3.5-9B BF16 con caricamento degli strati: 0,153 token/s con prefetch, 0,126 senza).
