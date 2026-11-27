# Qwen3.5-9B · Inferenza reale con SWLP sul Mac M1

Pesi Qwen3.5-9B originali, revisione `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Gli shard per strato sono prodotti dal convertitore upstream SWLP, commit `3941963812850f9ad6821f3bec78754d695143ac`, in BF16. Il runtime è SWLP su PyTorch 2.15.0/MPS e Transformers 5.24.0. Esegue generazione greedy, batch 1, con lettura diretta da SSD, finestra di due strati e zero strati fissati in RAM. Embedding e testa rimangono residenti. I prompt sono prefissi del manoscritto già presente nel workspace, tokenizzati con il tokenizer Qwen. Il primo token misura il tempo dall'inizio della generazione, dopo caricamento e tokenizzazione. Il decode esclude il primo token. [Configurazione e script](misura_swlp.py) · [Preparazione degli shard](prepara_swlp.py) · [Log di preparazione](../../runs/qwen_swlp_prepare.log).

Il picco RAM è il footprint del processo intero da `/usr/bin/time -l`. I byte letti provengono dai contatori macOS dei dispositivi e includono caricamento, prefill e decode. Il contatore può includere letture di altri processi. Il backend è verificato e la modalità strict disabilita i ripieghi di SWLP. Transformers usa le implementazioni PyTorch di riferimento per convoluzione causale e Gated DeltaNet su questo ambiente. I numeri riguardano quindi questa configurazione e precisione, senza quantizzazione e senza decodifica speculativa.

## Risultati con quattro token generati

| Input | Prefetch | Primo token | Decode | Totale inferenza | Picco processo | Letture incl. caricamento |
|---:|---|---:|---:|---:|---:|---:|
| 256 token | No | 11,30 s | 0,126 tok/s | 35,15 s | 5,83 GiB | 56,89 GB |
| 256 token | Sì | 12,06 s | 0,153 tok/s | 31,72 s | 7,63 GiB | 57,15 GB |
| 4.096 token | Sì | 95,55 s | 0,147 tok/s | 115,91 s | 7,45 GiB | 154,50 GB |

Una prova per condizione, in processi separati. Su 256 token è verificata l'identità di prompt e testo generato con e senza prefetch. Nella coppia, i tre passi di decode passano da 23,85 s a 19,65 s, ossia −17,6% di tempo, a fronte di letture simili e maggior footprint. Per stimare la variabilità del margine servono repliche appaiate. A 4.096 token, il prefill a blocchi da 512 richiede otto passaggi sugli strati. Includendo i tre passi di decode, la crescita osservata delle letture è coerente con il maggiore numero di passaggi. Le misure di footprint per 256 e 4.096 includono le differenze di allocazione e di avvio dei due processi. [Riepilogo verificato](RISULTATI_QWEN_SWLP_REALE.json) · [256 con prefetch](../../runs/qwen_swlp_256_replica.json) · [256 senza prefetch](../../runs/qwen_swlp_256_serial.json) · [4.096 con prefetch](../../runs/qwen_swlp_4096.json) · [Script del riepilogo](riepiloga_swlp.py).

## Prova iniziale

Con 256 token in ingresso e 8 generati, SWLP ha prodotto il primo token in **10,92 s** e il decode in **6,11 s/passo** (**0,164 tok/s**). Il picco del processo intero è **6,69 GiB** e i contatori hanno registrato **113,02 GB** di letture. Il profilo della pipeline mostra 30 attese su 32 strati nella traccia mostrata dal profiler. Il PDF è stato compilato durante questa prova iniziale, perciò la prova è conservata come controllo di funzionamento. [Output e profilo](../../runs/qwen_swlp_256.json) · [Log e footprint](../../runs/qwen_swlp_256.stderr).

## Riproduzione

Impostare `PYTHONPATH` sul checkout SWLP indicato e sulle sue dipendenze. Le dipendenze aggiuntive si rigenerano con `.venv-tinystories/bin/python -m pip install --no-deps --target runs/qwen_swlp_deps accelerate==1.16.0 psutil==7.2.2 python-json-logger==4.2.0`. Usare gli shard ufficiali già scaricati come sorgente. La preparazione scrive circa 18,4 GB aggiuntivi in una cartella temporanea.

```sh
.venv-tinystories/bin/python timing_analysis/03_F10_SSD/prepara_swlp.py "$sorgente_qwen" "$shard_temporanei"
/usr/bin/time -l .venv-tinystories/bin/python timing_analysis/03_F10_SSD/misura_swlp.py \
  "$sorgente_qwen" "$shard_temporanei" 256 --tokens 4
```

Per isolare il costo del prefill si varia la lunghezza dell'input. Per la prova senza prefetch si aggiunge `--serial`. I modelli Qwen e F10 hanno precisioni, architetture e tokenizer diversi. Le misure illustrano strategie di esecuzione e regimi di memoria. [Simulazione con strati residenti](REPORT_SIMULAZIONE_QWEN_F10.md).

[Ultima traccia per strato](../../runs/qwen_swlp_ultima_traccia_strati.json) · [Log delle dipendenze](../../runs/qwen_swlp_dependencies.log).
