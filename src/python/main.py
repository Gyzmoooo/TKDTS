import sys
import requests
import datetime
import time
import re
import traceback
import string
import pandas as pd
import numpy as np
from joblib import load

# --- Costanti e Configurazioni 
URL_FETCH = "http://192.168.4.1/"
URL_DELETE = f"{URL_FETCH}delete"
MODEL_PATH = 'C:\\Users\\giuli\\Desktop\\Code\\TKDTS\\model\\rf_model.sav'

TIMESTEPS = 20 # numero di prese dati per calcio
NUM_DATA_PER_TIMESTEP = 24 # totale dati per presa
EXPECTED_DATA_COLUMNS = NUM_DATA_PER_TIMESTEP * TIMESTEPS # colonne csv
EXPECTED_ESP_IDS = [str(i + 1) for i in range(4)] # ids esp
MAX_FETCH_ATTEMPTS = 2
RETRY_DELAY_SECONDS = 2
NUM_BOARDS = 4
SENSORS = ['A', 'G']
AXES = ['x', 'y', 'z']
N_DATA = TIMESTEPS * NUM_BOARDS * len(SENSORS) * len(AXES)

# --- Funzioni di Elaborazione Dati 
def generate_column_names(num_timesteps):
    column_names = []
    for i in range(num_timesteps):
        suffix = f"_{i + 1}"
        for sensor_type in SENSORS:
            for board_id in range(1, NUM_BOARDS + 1):
                for axis in AXES:
                    col_name = f"{sensor_type}{board_id}{axis}{suffix}"
                    column_names.append(col_name)
    return column_names

def parse_data(raw_data, expected_ids_list, target_timesteps):
    number_pattern = r'-?\d+\.?\d*'

    # Perchè i dati registrati dal master sono sempre di meno di quelli registrati dagli slave?
    data = {esp_id: [] for esp_id in expected_ids_list}
    
    for esp_id in expected_ids_list:
        last_start_idx = raw_data.rfind(f"Start{esp_id};")
        last_end_idx = raw_data.find(f"End{esp_id};", last_start_idx + len(f"Start{esp_id};"))
        if last_end_idx != -1 and last_start_idx != -1:
            esp_block = raw_data[last_start_idx + len(f"Start{esp_id};") : last_end_idx]
            esp_block = esp_block.replace(f"ID{esp_id}", "")
            data[esp_id] = [float(num_str) for num_str in re.findall(number_pattern, esp_block)]
    
    active_esp_ids = [esp_id for esp_id in expected_ids_list if data.get(esp_id)]
    samples_counts = [(len(data[esp_id]) / 6) for esp_id in active_esp_ids]
    print(samples_counts)
    min_samples = min(samples_counts)
    
    if target_timesteps > 0 and len(samples_counts) != 4: 
        raise Exception(f"{4 - len(samples_counts)} of the ESP32 didn't send any data")

    elif min_samples < target_timesteps:
            raise Exception(f"Samples available ({min_samples}) < target ({target_timesteps})")
    
    return data
    
def delete_data_on_master():
    try:
        print("Sending DELETE request at", URL_DELETE)
        response = requests.get(URL_DELETE, timeout=5)
        if response.status_code == 200 and "OK" in response.text:
             return True
        else:
             print(f"Error sending DELETE command: Status {response.status_code}, Response: {response.text}")
             return False
    except requests.exceptions.Timeout:
        print(f"Timeout during DELETE request at {URL_DELETE}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Error during communication with Master (DELETE): {e}")
        return False
    except Exception as e:
        print(f"Generic error during DELETE: {e}")
        return False

# --- Classe Worker per il Background Processing
class Worker:
    def __init__(self, model, columns):
        self.model = model
        self.dataframe_columns = columns
        self._is_running = True

    def run(self):
        print("Avvio thread worker...") # Log Console
        cycle_counter = 0

        while self._is_running:
            cycle_counter += 1
            print(f"\n===== Ciclo {cycle_counter}: Inizio Acquisizione =====")
            parsed_data = None; fetch_successful = False
            sufficient_data_received = False; delete_attempted = False

            # 1. Fetch Loop
            for attempt in range(MAX_FETCH_ATTEMPTS):
                if not self._is_running: break

                print(f"Tentativo Fetch {attempt + 1}/{MAX_FETCH_ATTEMPTS}...")

                try:
                    response = requests.get(URL_FETCH, timeout=10)
                    response.raise_for_status()
                    aggregated_data = response.text

                    if not aggregated_data or not aggregated_data.strip():
                        print("Buffer Master vuoto.")
                        if attempt < MAX_FETCH_ATTEMPTS - 1: time.sleep(RETRY_DELAY_SECONDS)
                        continue
                    else:
                        print(f"Dati ricevuti (Len: {len(aggregated_data)}). Parsing...")
                        parsed_data = parse_aggregated_data(aggregated_data, EXPECTED_ESP_IDS)

                        if parsed_data:
                            fetch_successful = True
                            active_ids = [k for k, v in parsed_data.items() if v]
                            if active_ids:
                                min_samples = min(len(parsed_data[eid]) for eid in active_ids) if active_ids else 0
                                print(f"Parsing OK. Campioni min: {min_samples}/{TIMESTEPS}")  
                                all_active_have_enough = all(len(parsed_data[eid]) >= TIMESTEPS for eid in active_ids)
                                if all_active_have_enough:
                                    print("--> Dati sufficienti ricevuti! Procedo.")
                                    sufficient_data_received = True
                                    break
                                else:
                                    print("Dati insufficienti. Attendo/Riprovo...")
                            else:
                                print("Parsing OK, ma nessun ESP attivo con dati.")
                        else:
                            print("Errore durante il parsing dei dati ricevuti.")

                        if not sufficient_data_received and attempt < MAX_FETCH_ATTEMPTS - 1:
                            time.sleep(RETRY_DELAY_SECONDS)

                except requests.exceptions.Timeout:
                    print(f"Timeout (Tentativo {attempt + 1}).")
                    if attempt < MAX_FETCH_ATTEMPTS - 1: time.sleep(RETRY_DELAY_SECONDS)
                except requests.exceptions.RequestException as e:
                    print(f"Errore connessione/HTTP (Tentativo {attempt + 1}): {e}")
                    if attempt < MAX_FETCH_ATTEMPTS - 1: time.sleep(RETRY_DELAY_SECONDS)
                except Exception as e:
                    print(f"Errore imprevisto fetch (Tentativo {attempt + 1}): {e}")
                    traceback.print_exc()
                    self._is_running = False
                    break

            if not self._is_running: break

            # 2. Processa SOLO se dati sufficienti
            if sufficient_data_received and parsed_data:
                print("Formattazione dati per DataFrame...")
                formatted_row_data = format_data_for_row(parsed_data, TIMESTEPS, EXPECTED_ESP_IDS)

                if formatted_row_data is not None:
                    print("Creazione DataFrame...")
                    try:
                        kick_dataframe = pd.DataFrame([formatted_row_data], columns=self.dataframe_columns)
                        if kick_dataframe.isnull().values.any():
                             nan_count = kick_dataframe.isnull().sum().sum()
                             print(f"ATTENZIONE: {nan_count} valori NaN nel DataFrame prima della predizione!")

                        # --- PULIZIA DATI ESP ---
                        print("Pulizia buffer dati sul Master ESP...")
                        delete_attempted = True
                        if not delete_data_on_master():
                             print("WARN: Fallita pulizia buffer Master.")

                        # 4. Esegui Predizione
                        print("Esecuzione predizione modello...")
                        x_numpy = kick_dataframe.values
                        y_pred = self.model.predict(x_numpy)
                        prediction_result = y_pred[0]
                        print(f"---> Predizione Ciclo {cycle_counter}: {prediction_result} <---")
                         # Log console

                    except Exception as e_df_pred:
                        print(f"\nERRORE durante creazione DataFrame o Predizione: {e_df_pred}")
                        traceback.print_exc()
                        self._is_running = False
                else:
                    print("Formattazione dati fallita. Impossibile procedere.")
                    if fetch_successful and not delete_attempted:
                        print("Tento pulizia buffer Master nonostante errore formattazione...")
                        delete_data_on_master()

            # Gestione casi fallimento Fetch / dati insufficienti
            elif fetch_successful and parsed_data:
                print(f"Dati insufficienti raccolti dopo {MAX_FETCH_ATTEMPTS} tentativi. Riprovo.")
                if not delete_attempted:
                    print("Tento pulizia buffer Master (dati insufficienti)...")
                    delete_data_on_master()
            elif fetch_successful and not parsed_data:
                print("Fetch OK ma parsing fallito. Riprovo.")
                if not delete_attempted:
                    print("Tento pulizia buffer Master (parsing fallito)...")
                    delete_data_on_master()
            elif not fetch_successful:
                print(f"Fetch fallito o interrotto dopo {MAX_FETCH_ATTEMPTS} tentativi. Riprovo.")


            if self._is_running:
                time.sleep(1)
                
        print("--- Thread Worker Terminato ---")

    def stop(self):
        print("Richiesta di arresto per il thread worker...") # Log Console
        self._is_running = False


# --- Esecuzione Principale ---
if __name__ == "__main__":
    print("Loading the model...")
    try:
        model = load(MODEL_PATH)
        print("Model loaded succesfully.")
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Model's file not found in {MODEL_PATH}")
        sys.exit(1)
    except Exception as e:
        print(f"CRITICAL ERROR: Unable to upload the model: {e}")
        traceback.print_exc()
        sys.exit(1)

    df_columns = generate_column_names(TIMESTEPS)
    if not df_columns or len(df_columns) != EXPECTED_DATA_COLUMNS:
        print("CRITICAL ERROR: Invalid column names or wrong number.")
        sys.exit(1)

    print("Starting the process of acquisition and prediction...")
    worker = Worker(model, df_columns)
    try:
        worker.run()
    except KeyboardInterrupt:
        print("\nKeyboard interruption detected.")
        worker.stop()
    except Exception as e:
        print(f"Critical error in the execution of the main worker: {e}")
        traceback.print_exc()