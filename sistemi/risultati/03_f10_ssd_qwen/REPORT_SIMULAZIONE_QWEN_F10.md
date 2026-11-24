# Qwen3.5-9B e STERA F10 · RAM, SSD e contesto

La simulazione usa la configurazione e le intestazioni dei quattro shard Qwen3.5-9B disponibili sul Mac. I file pesano **19,306 GB**. Il decoder testuale greedy in BF16 pesa **17,908 GB** al netto di visione e MTP. Di questi **13,839 GB** sono nei 32 blocchi di decoder. Il resto, incluso embedding e testa separati, rimane residente nello scenario. F10 in FP32 ha **4,003 GB** di pesi compatti e un payload Dense equivalente stimato di **18,597 GB**. F10 è un prototipo di scala inizializzato da H80000. Le precisioni e le architetture sono diverse. [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) · [Architettura ibrida Qwen](https://huggingface.co/docs/transformers/model_doc/qwen3_5) · [Evidenza F10](REPORT_F10_SSD_FP32_20261123.md).

## Il vantaggio di residenza sul Mac M1

| Rappresentazione | Pesi | Soglia di 16 GiB |
|---|---:|---|
| F10 compatto FP32 | 3,73 GiB | Pesi residenti nella prova |
| F10 espanso FP32 | 17,32 GiB | Il solo payload supera la RAM fisica |
| Qwen3.5-9B, decoder testuale BF16 | 16,68 GiB | Il solo payload supera la RAM fisica. SWLP legge gli strati da SSD |

Per F10 compatto ed espanso il confronto a stessa precisione quantifica il passaggio di soglia dovuto alla rappresentazione. L'espansione è un conteggio di payload. La generazione reale Qwen/SWLP mostra il costo dello streaming in un modello pubblico a precisione BF16, mentre F10 compatto genera con i pesi residenti. Le architetture, le precisioni e i carichi sperimentali differiscono. [Misure end-to-end Qwen](REPORT_QWEN_SWLP_REALE.md).

## Stima per richiesta singola

| Contesto, token | F10, cache KV FP32, GiB | F10, pesi + KV, GiB | Qwen, cache e stato, GiB | Qwen, s/token di lettura, budget 8 GiB | Qwen, s/token di lettura, budget 12 GiB |
|---:|---:|---:|---:|---:|---:|
| 512 | 0,132 | 3,86 | 0,064 | 4,90 | 3,06 |
| 2.048 | 0,422 | 4,12 | 0,111 | 4,90 | 3,06 |
| 8.192 | 1,450 | 5,18 | 0,298 | 4,90 | 3,07 |
| 32.768 | 5,669 | 9,40 | 1,048 | 5,27 | 3,43 |
| 131.072 | — | — | 4,048 | Oltre budget | 4,90 |

I budget di 8 e 12 GiB sono disponibili al singolo processo. Ogni scenario riserva 1 GiB per runtime e workspace. Qwen riserva anche due buffer grandi quanto lo strato più pesante (0,81 GiB complessivi). Lo spazio rimanente viene usato per tenere in memoria strati interi, scegliendo prima i più grandi. I byte degli strati rimanenti sono letti una volta per token nel modello di simulazione. La banda usata, **2,378 GB/s**, è la mediana della lettura SSD sequenziale validata senza cache. A 32K e 12 GiB rimangono residenti 13 strati Qwen su 32. Il termine di lettura di 3,43 s/token equivale a un tetto I/O condizionato alla banda e al piano ipotizzati di **0,29 token/s**. La contabilità F10 a 32K richiede 9,40 GiB per pesi e KV e supera il budget di 8 GiB anche prima della riserva. [Tracce SSD](evidenze_f10/ssd_validato/ssd_validato.json) · [Conti per scenario](SIMULAZIONE_QWEN_F10.json) · [Script](simula_qwen_f10.py).

Un doppio buffer permette di leggere lo strato successivo mentre si calcola quello corrente. A regime, con sovrapposizione ideale, la durata per token è almeno il maggiore fra tempo di lettura e calcolo. In serie i due termini si sommano. Frammentazione delle letture, riempimento della pipeline, scritture della cache e contesa possono aumentare i tempi. Cache del sistema operativo, quantizzazione, decodifica speculativa o un piano di residenza diverso cambiano la quantità di byte per token. La lettura integrale del volume Dense FP32 equivalente a F10 richiede circa 7,82 s nella prova I/O. Per Qwen il volume dei soli strati dà 5,82 s di lettura teorica quando nessuno strato è già residente.

## Evidenza di esecuzione disponibile

- **F10, richiesta breve.** Due esecuzioni con prompt da 3 token generano circa 11,13–12,02 tok/s, capacità KV 128. [Tracce](evidenze_f10/inferenza_22nov/).
- **F10, cache sintetica piena.** A 8K è stato rilevato un decode di 9,09 tok/s (v30). A 32K, dopo il primo passo, 4,75 tok/s e picco di allocazione MLX di 9.787 MiB (v44, KV FP32). Le due misure usano versioni diverse del runtime. I test riempiono direttamente la cache e fissano l'indice posizionale oltre le 2.048 righe apprese. Il training è avvenuto a 512 token. Le tracce hanno zero token di prompt realmente processati. [Evidenze dei test meccanici](evidenze_contesto/).
- **Qwen/SWLP.** La tabella resta una simulazione a strati con residenza parziale. La successiva campagna end-to-end usa una finestra di due strati e zero strati fissati in RAM. Ha elaborato realmente input da 256 e 4.096 token. [Campagna end-to-end](REPORT_QWEN_SWLP_REALE.md).

```sh
python3 timing_analysis/03_F10_SSD/simula_qwen_f10.py \
  ~/Desktop/TESI_CHIUSURA_20261123/confronto_pubblici/qwen35_9b_source
```
