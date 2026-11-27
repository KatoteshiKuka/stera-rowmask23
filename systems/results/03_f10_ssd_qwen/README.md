# F10 e SSD interno

[Report FP32 e lettura SSD](REPORT_F10_SSD_FP32_20261123.md) · [Misura SSD verificata](evidenze_f10/ssd_validato/ssd_validato.json).

Il [log di download](QWEN35_DOWNLOAD.log) registra 13/13 file di Qwen3.5-9B, inclusi i quattro shard originali e la configurazione. Il [protocollo storico](PROTOCOLLO_QWEN_SWLP_STORICO.md) descrive la scelta del runtime e il limite di spazio iniziale. La [campagna Qwen/SWLP end-to-end](REPORT_QWEN_SWLP_REALE.md) è stata eseguita dopo aver verificato lo spazio per gli shard temporanei. I 7,82 s appartengono alla lettura sequenziale di un volume equivalente a 18,60 GB.

F10 è un prototipo di scala. Il confronto di qualità narrativa in tesi usa H80000 e i TinyStories ufficiali.

La [simulazione Qwen/F10](REPORT_SIMULAZIONE_QWEN_F10.md) usa le intestazioni degli shard Qwen locali e la banda SSD misurata. Include le prove F10 a cache KV sintetica e dichiara il costo di lettura stimato per contesto e budget RAM.
