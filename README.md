# Human Activity Recognition using Mobile Phone IMU Sensors

Progetto per il corso **"Machine Learning e Data Mining"** (a.a. **2023/2024**).

## Obiettivo
Realizzare un sistema di **Human Activity Recognition (HAR)** a partire dai dati di **sensori IMU** (accelerometro/giroscopio) di uno smartphone, confrontando diversi approcci di preprocessing, feature extraction e modelli di machine learning.

## Contenuto della repository
- **Notebook Jupyter**: analisi esplorativa, preprocessing, training e valutazione dei modelli.
- **Codice Python** (se presente): utility e funzioni di supporto.

> Nota: la repository è composta quasi interamente da notebook (≈98%). Per rendere il progetto più riproducibile, è consigliato mantenere i passi principali anche in script/utility riutilizzabili.

## Dataset
Indicare qui:
- **Nome del dataset**
- **Fonte/Link**
- **Formato** (CSV, parquet, ecc.)
- **Attività/label** disponibili
- Eventuali note su licenza o permessi d’uso

*(Suggerimento: aggiungi una cartella `data/` solo se i dati sono redistribuibili; altrimenti inserisci istruzioni per scaricarli.)*

## Requisiti
Consigliato usare un ambiente dedicato.

### Opzione A — con `pip`
```bash
python -m venv .venv
source .venv/bin/activate  # su Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### Opzione B — con `conda`
```bash
conda create -n har-imu python=3.11
conda activate har-imu
pip install -r requirements.txt
```

## Come eseguire
1. Clona la repository:
   ```bash
   git clone https://github.com/mattiafornari/Human-activity-recognition-using-mobile-phone-IMU-sensors.git
   cd Human-activity-recognition-using-mobile-phone-IMU-sensors
   ```
2. Installa i requisiti (vedi sezione sopra).
3. Avvia Jupyter:
   ```bash
   jupyter lab
   ```
4. Apri ed esegui i notebook nell’ordine suggerito (vedi sezione *Struttura*).

## Struttura (suggerita)
Se vuoi, puoi organizzare la repo con una struttura simile:

```text
.
├─ notebooks/
├─ src/
├─ data/                 # opzionale, se redistribuibile
├─ models/               # opzionale: modelli salvati
├─ reports/              # opzionale: figure, tabelle, risultati
├─ requirements.txt
└─ README.md
```

## Risultati
Aggiungi:
- Metriche principali (accuracy, F1, confusion matrix)
- Confronto tra modelli
- Grafici rilevanti

## Riproducibilità
Per migliorare la riproducibilità:
- Fissa le versioni in `requirements.txt` o `environment.yml`.
- Imposta seed (NumPy / scikit-learn / framework usato).
- Specifica chiaramente il workflow (preprocessing → training → valutazione).

## Roadmap / TODO
- [ ] Aggiungere `requirements.txt` (o `environment.yml`).
- [ ] Documentare il dataset e come ottenerlo.
- [ ] Rendere i notebook più modulari (funzioni in `src/`).
- [ ] Salvare e versionare i risultati principali (tabelle/figure).

## Autore
- Mattia Fornari

## Licenza
Specifica qui la licenza del progetto (es. MIT) oppure indica che è un progetto accademico senza licenza esplicita.
