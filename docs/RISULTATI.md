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
| STERA 30M, coordinate INT8 (altri tensori FP32) | 74.240.928 |
| TinyStories-8M FP32 | 78.810.112 |
| STERA 30M, archivio senza perdita | 102.352.624 |
| STERA 30M FP32 | 121.384.728 |
| TinyStories-33M FP32 | 274.056.192 |
| STERA 30M, espansione densa FP32 | 330.885.120 |

Fonti: conteggio dei parametri (`addestramento/ts68_v3.py`) e file in `sistemi/modelli/`.

## Tempi di esecuzione (Capitolo 8)

Mac con Apple M1, 16 GB, MLX/Metal, FP32, 256 token in ingresso, 100 passi di decode, pesi del checkpoint finale. Cinque campagne di misura consecutive (`sistemi/benchmark/campagne_tempi.sh`, media di carico fra 2,7 e 4,6); mediane a caldo su 30 coppie (6 input × 5 campagne). Fonte: `sistemi/risultati/campagne_20261124/protocollo/riepilogo.json`, prodotto da `sistemi/benchmark/riepiloga_campagne.py`.

| Motore | Primo token | Decode | Totale |
|---|---:|---:|---:|
| Denso equivalente | 36,7 ms | 102,13 token/s | 1,016 s |
| Compatto (coordinate) | 75,5 ms | 158,83 token/s | 0,704 s |

Variazione mediana appaiata del tempo per token in decode: −35,9% (da −31,3% a −40,5% sulle 30 coppie; mediane per campagna da −35,3% a −36,4%). Il denso emette prima il primo token e il compatto termina prima la richiesta in tutte le 30 coppie. I due motori producono gli stessi 101 token su tutti gli input, identici in tutte le campagne.

Tre modelli sullo stesso Mac (STERA 30M su MLX/Metal, TinyStories su PyTorch 2.15 MPS), 3 input × 5 campagne, mediane a caldo:

| Modello | Primo token | Decode | Totale | Picco RAM |
|---|---:|---:|---:|---:|
| TinyStories-8M | 21,8 ms | 176,55 token/s | 0,591 s | 1.525,2 MiB |
| STERA 30M | 121,3 ms | 160,03 token/s | 0,745 s | 412,9 MiB |
| TinyStories-33M | 31,8 ms | 94,79 token/s | 1,088 s | 1.493,1 MiB |

I token dei due TinyStories coincidono con quelli dell'implementazione eager in ogni campagna.

Prototipo F10 e Qwen3.5-9B: `sistemi/risultati/03_f10_ssd_qwen/` (F10 inizializzato dal checkpoint finale, decode 9,89 e 9,73 token/s in `evidenze_f10/inferenza_H122000/`; pesi compatti 4,003 GB contro 18,597 GB densi; lettura sequenziale dall'SSD 1,685 s contro 7,821 s; Qwen3.5-9B BF16 con caricamento degli strati: 0,153 token/s con prefetch, 0,126 senza).
