# H80000 e TinyStories ufficiali · Pesi, addestramento e inferenza

| Checkpoint | Parametri allenabili | Pesi FP32, MB | File ufficiale o di lavoro, MB | Token di training |
|---|---:|---:|---:|---|
| TinyStories-8M | 19.702.528 | 78,810 | 112,405 | Numero di passaggi e token non documentato nelle fonti consultate |
| STERA H80000 | 30.346.182 | 121,385 | 121,455 | **1.002.967.223** token bersaglio effettivi, da log Tesla |
| TinyStories-33M | 68.514.048 | 274,056 | 290,854 | Circa 20 epoche dichiarate, **circa 9,5 miliardi** di token con il tokenizer pubblico come proxy |

MB decimali. I NPZ/BIN comprendono la serializzazione e i BIN ufficiali contengono anche buffer delle maschere. L'archivio lossless STERA occupa 102,298 MB. La rappresentazione densa equivalente di H80000 materializza 82.721.280 valori, 330,885 MB di soli pesi FP32. [Audit dei pesi ufficiali](../../evaluations/audit_tesi/TINYSTORIES_CHECKPOINT_UFFICIALI.md) · [Conti](../../evaluations/audit_tesi/conti_verificati.json).

Per H80000 il campo `labels` conta i bersagli della loss ed esclude il padding. La scheda del corpus indica `TinyStories-train.txt` per i checkpoint pubblici. Per il 33M Eldan dichiara circa 20 epoche, contesto 512. Il proxy applica il BPE pubblico del 33M a 16 campioni distribuiti nel file ufficiale e stima circa 475 milioni di token per passaggio. Il rapporto nominale di circa 9,5 volte rispetto a H80000 descrive questa stima. Il tokenizer in uso durante l'addestramento e il trattamento effettivo delle finestre possono cambiare il totale. Il budget dell'8M rimane indeterminato. [Log Tesla](TESLA_H80000_TARGETS.txt) · [Scheda del corpus](https://huggingface.co/datasets/roneneldan/TinyStories) · [Dichiarazione di Eldan](https://huggingface.co/roneneldan/TinyStories-33M/discussions/8) · [Campionamento e conti](stima_token_corpus.json) · [Script](../../evaluations/audit_tesi/stima_corpus_tinystories.py).

## Inferenza misurata sul Mac M1

| Checkpoint | Primo token a caldo, ms | Decode a caldo, tok/s | Totale a caldo, s | Massimo footprint RAM campionato, MiB |
|---|---:|---:|---:|---:|
| TinyStories-8M | 21,4 | 121,67 | 0,844 | 1.508,30 |
| STERA H80000 | 111,7 | 167,61 | 0,708 | 408,05 |
| TinyStories-33M | 34,6 | 73,35 | 1,398 | 1.474,39 |

MacBook Air M1 da 16 GiB, GPU Apple, FP32, batch 1, tre processi per modello, ordine ruotato a ogni input. Ogni gruppo riceve gli stessi 256 ID in ingresso. La richiesta greedy produce il primo token e poi esegue altri 100 passi, con sincronizzazione GPU. I tempi sono mediane di tre richieste a caldo, dopo una prima richiesta completa con lo stesso input. La cache KV riparte vuota. I tempi escludono il caricamento. Il footprint fisico del processo (`proc_pid_rusage`) è campionato ogni 10 ms durante prefill e decode. La tabella riporta il massimo osservato nella fase a caldo, inclusi processo, pesi residenti e allocazioni trattenute dal runtime. Le tracce registrano anche la prima richiesta e il massimo del processo intero da `time -l`. I picchi campionati possono perdere transitori brevi.

STERA usa MLX/Metal v63 sulle coordinate compatte con prefill a blocchi da 16, scelto nella campagna precedente. Entrambi i GPT-Neo eseguono i pesi ufficiali con Transformers 5.24.0, PyTorch 2.15.0/MPS, adattatore SDPA e proiezione dei soli logits dell'ultimo token in prefill. Per entrambi i TinyStories tutti i 101 ID coincidono con eager e proiezione completa su 3 input su 3. Il confronto riguarda questi runtime e il footprint comprende i loro costi fissi. Anche il TinyStories con meno pesi ha un footprint elevato in questo ambiente. Il protocollo narrativo, con campionamento e giudizio semantico, è documentato nella campagna TEST100.

[Grezzi ottimizzati](ottimizzati.jsonl) · [Baseline eager](eager_parita.jsonl) · [Mediane e intervalli osservati](RIEPILOGO.json) · [Launcher](../../benchmark/TinyStories/esegui_confronto_tesi.sh).

```sh
sh benchmark/TinyStories/esegui_confronto_tesi.sh timing_analysis/nuova_replica
python3 benchmark/TinyStories/riepiloga_confronto_tesi.py timing_analysis/nuova_replica
```

Il vantaggio di throughput o di memoria a egual numero di scalari allenabili richiede una propria misura. A parità di conteggio e precisione, infatti, il payload dei soli scalari è uguale. Il confronto con l'espansione dello stesso H80000 isola meglio l'effetto della rappresentazione e dell'implementazione. [Campagna H80000 appaiata](../01_H80000_Dense_vs_STERA/REPORT_H80000_MOTORI_OTTIMIZZATI.md).
