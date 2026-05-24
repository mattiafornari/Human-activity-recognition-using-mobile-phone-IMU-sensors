# Human Activity Recognition using Mobile Phone IMU Sensors

Progetto per il corso **Machine Learning e Data Mining** (a.a. **2023/2024**).

Questo repository contiene un workflow end-to-end per il riconoscimento di attività umane (HAR) a partire dai segnali IMU di uno smartphone (**accelerometro** e **giroscopio**).

---

## Obiettivo

Realizzare un sistema di **Human Activity Recognition (HAR)** utilizzando dati da sensori IMU di smartphone e confrontare diversi modelli di machine learning e deep learning.

---

## Dataset

Nel notebook viene utilizzato il dataset **“Accelerometer Gyro Mobile Phone”** (UCI ML Repository, ID: **755**).

**Feature principali** (come nel notebook):
- `accX`, `accY`, `accZ` (accelerometro)
- `gyroX`, `gyroY`, `gyroZ` (giroscopio)
- `timestamp` (categorica / tempo)
- `Activity` (target / label)

> Nota: per dettagli su sorgente, licenza e modalità di caricamento fare riferimento ai notebook.

---

## Struttura della repository

```text
.
├─ README.md
├─ Update_Finale_Progetto_MLDM.ipynb
├─ data_analysis_and_preprocessing/
│  ├─ Data_Analysis_and_PreProcessing.ipynb
│  └─ readme.txt
├─ dataset/
└─ development/
```

---

## Come eseguire

Eseguire i notebook nell’ordine seguente:

1. `data_analysis_and_preprocessing/Data_Analysis_and_PreProcessing.ipynb`
2. `Update_Finale_Progetto_MLDM.ipynb`

### Ambiente

I notebook sono compatibili con **Google Colab**. Nel notebook finale sono presenti celle per installare le dipendenze necessarie (es. `ucimlrepo`, `optuna`, `scikeras`, `ydata-profiling`, …).

---

## Modelli addestrati

Nel progetto vengono addestrati e confrontati i seguenti modelli:

- **SVM**
- **Random Forest**
- **Extra Trees**
- **AdaBoost**
- **CNN 1D** (deep learning su sequenze/segnali)

---

## Metriche di valutazione

Per la valutazione dei modelli sono utilizzate:

- **Confusion Matrix**
- **Classification Report**
- **Accuracy**
- **Precision** / **Recall**
- **F1-score**
- **F2-score**

---

## Risultati

I risultati (metriche, tabelle e grafici) sono riportati direttamente nei notebook.

Se vuoi includere nel README una tabella comparativa “finale” (per modello: Accuracy / Precision / Recall / F1 / F2), dimmi i valori oppure indicami dove sono stampati nel notebook e la aggiungo in modo ordinato.

---

## Autore

- Mattia Fornari

---

## Licenza

Progetto accademico. Se vuoi renderlo riutilizzabile da terzi, valuta di aggiungere un file `LICENSE` (es. MIT).
