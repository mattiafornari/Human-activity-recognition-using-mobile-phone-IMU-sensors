from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import os
import uvicorn

app = FastAPI(title="IMU ML API", description="API per predizioni basate su dati IMU")

# Abilita CORS per consentire le richieste dall'app frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In produzione, limita agli URL effettivi del frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definisci la struttura dei dati di input
class IMUData(BaseModel):
    accX: float
    accY: float
    accZ: float
    gyroX: float
    gyroY: float
    gyroZ: float

# Variabile globale per il modello
model = None
model_type = "sklearn"  # Impostato su sklearn poiché useremo un modello .joblib

@app.on_event("startup")
async def load_model():
    """Carica il modello durante l'avvio dell'applicazione"""
    global model
    
    # Configura il percorso del modello (nella stessa directory di app.py)
    model_path = os.getenv("MODEL_PATH", "classification_pipeline_last.joblib")
    
    try:
        # Carica il modello .joblib
        print(f"Caricamento del modello da {model_path}")
        model = joblib.load(model_path)
        print(f"Modello {model.__class__.__name__} caricato correttamente")
        
    except Exception as e:
        print(f"Errore nel caricamento del modello: {str(e)}")
        raise RuntimeError(f"Impossibile caricare il modello: {str(e)}")

@app.get("/")
async def root():
    """Endpoint di base che conferma che l'API è in esecuzione"""
    if model is None:
        model_status = "non caricato"
    else:
        model_status = model.__class__.__name__
        
    return {
        "message": "IMU ML API operativa",
        "model_type": model_type,
        "model_status": model_status,
        "endpoints": {
            "POST /predict": "Effettua una predizione con dati IMU",
            "GET /model-info": "Restituisce informazioni sul modello caricato"
        }
    }

@app.post("/predict")
async def predict(data: IMUData):
    """Effettua una predizione utilizzando i dati IMU"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modello non caricato")
    
    try:
        # Prepara i dati di input come DataFrame pandas
        import pandas as pd
        
        # Crea un DataFrame con i nomi delle colonne appropriati
        input_data = pd.DataFrame([[
            data.accX, data.accY, data.accZ,
            data.gyroX, data.gyroY, data.gyroZ
        ]], columns=["accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ"])
        
        # Effettua la predizione
        prediction = model.predict(input_data)
        
        # Crea il risultato base
        result = {"prediction": prediction.tolist()}
        
        # Aggiungi probabilità se disponibili (per modelli di classificazione)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)
            result["probabilities"] = probabilities.tolist()
            
            # Se abbiamo le classi, includiamole
            if hasattr(model, "classes_"):
                result["classes"] = model.classes_.tolist()
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nella predizione: {str(e)}")
    
@app.get("/model-info")
async def model_info():
    """Restituisce informazioni sul modello caricato"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modello non caricato")
    
    try:
        info = {
            "model_type": model_type,
            "model_class": model.__class__.__name__,
        }
        
        # Aggiungi dettagli del modello sklearn in modo sicuro
        if hasattr(model, "get_params"):
            try:
                params = model.get_params()
                # Filtra eventuali parametri non serializzabili
                serializable_params = {}
                for key, value in params.items():
                    try:
                        # Verifica se è serializzabile convertendolo in JSON
                        import json
                        json.dumps({key: str(value)})
                        serializable_params[key] = str(value)
                    except:
                        serializable_params[key] = f"<non-serializable: {type(value).__name__}>"
                
                info["parameters"] = serializable_params
            except Exception as e:
                info["parameters_error"] = f"Errore nell'estrazione dei parametri: {str(e)}"
            
        if hasattr(model, "feature_importances_"):
            try:
                importances = model.feature_importances_
                if isinstance(importances, np.ndarray):
                    info["feature_importances"] = importances.tolist()
                else:
                    info["feature_importances"] = list(importances)
            except Exception as e:
                info["feature_importance_error"] = f"Errore nell'estrazione delle feature importance: {str(e)}"
        
        # Aggiungi informazioni sulle classi per modelli di classificazione
        if hasattr(model, "classes_"):
            try:
                classes = model.classes_
                if isinstance(classes, np.ndarray):
                    info["classes"] = classes.tolist()
                else:
                    info["classes"] = list(classes)
                info["n_classes"] = len(classes)
            except Exception as e:
                info["classes_error"] = f"Errore nell'estrazione delle classi: {str(e)}"
        
        # Se il modello è una pipeline, aggiungi informazioni sugli step
        if hasattr(model, "steps"):
            try:
                steps_info = []
                for name, step in model.steps:
                    steps_info.append({
                        "name": name,
                        "class": step.__class__.__name__
                    })
                info["pipeline_steps"] = steps_info
            except Exception as e:
                info["pipeline_error"] = f"Errore nell'estrazione degli step della pipeline: {str(e)}"
        
        return info
    
    except Exception as e:
        # Usa un formato di risposta semplificato in caso di errore
        return {
            "model_type": model_type,
            "model_class": model.__class__.__name__,
            "error": f"Errore nella generazione delle informazioni complete: {str(e)}"
        }

if __name__ == "__main__":
    # Porta configurabile tramite variabile d'ambiente
    port = int(os.getenv("PORT", 8000))
    
    # Avvia il server
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)