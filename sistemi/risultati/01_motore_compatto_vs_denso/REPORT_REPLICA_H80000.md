# Replica H80000 a download Qwen terminato

MacBook Air M1, alimentato da rete, MLX/Metal, FP32, batch 1, 256 token in ingresso, cache KV da 512 posizioni e 100 passi greedy dopo il primo token. I pesi sono già caricati. Prefill a blocchi da 128 per STERA e 256 per Dense. Sei processi separati per motore, con ordine alternato. Su ciascun input i due motori producono gli stessi 101 ID sia alla prima richiesta sia a caldo. Il desktop era attivo e lo swap allocato di sistema era circa 398 MiB prima della replica. Il download Qwen era terminato.

| Motore | Primo token, mediana a caldo | Decode, mediana a caldo | Tempo totale, mediana a caldo |
|---|---:|---:|---:|
| H80000 Dense | 31,3 ms | 115,26 tok/s | 0,899 s |
| H80000 compatto | 86,3 ms | 156,24 tok/s | 0,719 s |

| Input | Variazione appaiata del tempo di decode per passo, STERA rispetto a Dense | Totale Dense (s) | Totale STERA (s) |
|---:|---:|---:|---:|
| 10 | −28,08% | 0,885 | 0,705 |
| 25 | −30,41% | 0,945 | 0,698 |
| 40 | −25,64% | 0,902 | 0,732 |
| 55 | −1,34% | 0,896 | 0,954 |
| 70 | −34,15% | 0,996 | 0,705 |
| 85 | −11,41% | 0,889 | 0,848 |

Il decode compatto richiede meno tempo in tutte le coppie. La mediana delle variazioni appaiate è **−26,86%** rispetto a −29,4% nella campagna precedente. Sul tempo totale STERA è avanti in cinque casi su sei. Dense emette prima il primo token in tutti i casi. La replica verifica la direzione sui sei carichi fissati, con una variazione appaiata del decode che va da −34,15% a −1,34%. Le misure riguardano questi due motori e la medesima funzione appresa.

Nella precedente prova con decoder ibrido, a cache iniziale condivisa, alternare blocchi interni, MLP e proiezione di uscita tra i due percorsi ha permesso di attribuire la differenza agli operatori. La mediana appaiata del passaggio da tutto Dense a tutto compatto era 2,75 ms per passo. Il rapporto di quella campagna usa un prefill comune. [Attribuzione per operatore](REPORT_ATTRIBUZIONE_3MS.md) · [Campagna precedente](REPORT_H80000_MOTORI_OTTIMIZZATI.md).

[Condizioni prima dell'avvio](../../runs/replica_h80000_condizioni.txt) · [12 log della replica e riepilogo](../../runs/confronto.O9WJK4/) · [Launcher](../../benchmark/confronto_appaiato.sh).
