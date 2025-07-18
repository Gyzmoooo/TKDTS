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
EXPECTED_ESP_IDS = [str(i + 1) for i in range(4)] # ids esp
MAX_FETCH_ATTEMPTS = 2
RETRY_DELAY_SECONDS = 2
NUM_BOARDS = 4
SENSORS = ['A', 'G']
AXES = ['x', 'y', 'z']
NUM_DATA_PER_TIMESTEP = len(SENSORS) * len(AXES) * NUM_BOARDS # totale dati per presa
EXPECTED_DATA_COLUMNS = NUM_DATA = TIMESTEPS * NUM_DATA_PER_TIMESTEP

# --- Funzioni di Elaborazione Dati 
def generate_column_names(num_timesteps):
    column_names = []
    for i in range(num_timesteps):
        suffix = f"_{i + 1}"
        for board_id in range(1, NUM_BOARDS + 1):
            for sensor_type in SENSORS:
                for axis in AXES:
                    col_name = f"{sensor_type}{board_id}{axis}{suffix}"
                    column_names.append(col_name)
    return column_names

def parse_data(raw_data, expected_ids_list):
    number_pattern = r'-?\d+\.?\d*'
    esp_data = []
    print(raw_data)
    
    # Parses all data and stores them in a list
    for esp_id in expected_ids_list:
        last_start_idx = raw_data.rfind(f"Start{esp_id};")
        last_end_idx = raw_data.find(f"End{esp_id};", last_start_idx + len(f"Start{esp_id};"))
        if last_end_idx != -1 and last_start_idx != -1:
            esp_block = raw_data[last_start_idx + len(f"Start{esp_id};") : last_end_idx]
            esp_block = esp_block.replace(f"ID{esp_id}", "")
            esp_data.append([float(num_str) for num_str in re.findall(number_pattern, esp_block)])
        else: esp_data.append([])

    return esp_data

def format_data(esp_data, target_timesteps, num_boards):
    # Verifies data completeness and number of samples 
    active_esp_ids = [str(esp_id + 1) for esp_id in range(len(esp_data)) if esp_data[esp_id] != []]
    print(active_esp_ids)
    samples_counts = [(len(esp_data[int(esp_id) - 1]) / 6) for esp_id in active_esp_ids]
    print(samples_counts)
    min_samples = int(min(samples_counts))
    if target_timesteps > 0 and len(samples_counts) != 4: 
        raise Exception(f"{4 - len(samples_counts)} of the ESP32 didn't send any data")
    elif min_samples < target_timesteps:
        raise Exception(f"Samples available ({min_samples}) < target ({target_timesteps})")

    # Eliminates excessive data
    for esp in range(num_boards):
        data_to_eliminate = int(-6 * (samples_counts[int(esp)] - min_samples) - 1)
        del esp_data[esp][-1:data_to_eliminate:-1]

    # Conversion in list of lists format, where each sublist contains a single sample
    unfiltered_data = []
    for i in range(min_samples):
        sample = []
        for j in range(num_boards):
            start = i * len(SENSORS) * len(AXES)
            end = start + len(SENSORS) * len(AXES)
            sample.extend(esp_data[j][start:end])
        unfiltered_data.append(sample)
    
    # Sample reduction to 20
    data = [[]]
    step = (len(unfiltered_data) - 1) / (target_timesteps - 1)
    for i in range(target_timesteps):
        index = round(i * step)
        for j in unfiltered_data[min(index, len(unfiltered_data) - 1)]: data[0].append(j)

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


class Predictor:
    def __init__(self, model, columns):
        self.model = model
        self.columns = columns
        self._is_running = True
    """
    def extract_data():
        raw = 0
        return raw

    def split():
        return 0"""

    def predict(self, kick_df):
        try:
            x_numpy = kick_df.values
            y_pred = self.model.predict(x_numpy)
            prediction_result = y_pred[0]
        except Exception as e_df_pred:
            print(f"\Prediction Error: {e_df_pred}")
            traceback.print_exc()

        return prediction_result

    def run(self):
        cycle_counter = 0

        while self._is_running:
            cycle_counter += 1
            print(f"\n===== Ciclo {cycle_counter}: Inizio Acquisizione =====")
            parsed_data = None; fetch_successful = False
            delete_attempted = False; sufficient_data_received = False
            data = []

            # 1. Fetch Loop
            for attempt in range(MAX_FETCH_ATTEMPTS):
                if not self._is_running: break

                print(f"Tentativo Fetch {attempt + 1}/{MAX_FETCH_ATTEMPTS}...")

                try:
                    response = requests.get(URL_FETCH, timeout=10)
                    response.raise_for_status()
                    raw = response.text
                    print(f"QUESTI SONO I DATI RAW RICEVUTI DA RESPONSE.TEXT: {raw}")


                    if not raw or not raw.strip():
                        print("Buffer Master vuoto.")
                        if attempt < MAX_FETCH_ATTEMPTS - 1: time.sleep(RETRY_DELAY_SECONDS)
                        continue
                    else:
                        print(f"Dati ricevuti (Len: {len(raw)}). Parsing...")
                        parsed_data = parse_data(raw, EXPECTED_ESP_IDS)

                        if parsed_data:
                            fetch_successful = True
                            active_ids = [str(esp + 1) for esp in range(4) if parsed_data[esp]]
                            if active_ids:
                                #min_samples = min(len(parsed_data[esp_id]) for esp_id in active_ids) if active_ids else 0
                                all_active_have_enough = all(len(parsed_data[int(esp_id) - 1]) >= TIMESTEPS for esp_id in active_ids)
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

            if not self._is_running: break

            # 2. Processa SOLO se dati sufficienti
            if sufficient_data_received and parsed_data:
                data = format_data(parsed_data, TIMESTEPS, NUM_BOARDS)
                
                if data is not None:
                    print("Creazione DataFrame...")
                    try:
                        kick_df = pd.DataFrame(data, columns=self.columns)

                        # --- PULIZIA DATI ESP ---
                        print("Pulizia buffer dati sul Master ESP...")
                        delete_attempted = True
                        if not delete_data_on_master():
                                print("WARN: Fallita pulizia buffer Master.")

                        # 4. Esegui Predizione
                        result = self.predict(kick_df)
                        print(f"---> Predizione Ciclo {cycle_counter}: {result} <---")
                    
                    except Exception as e_df:
                        print(f"\nERRORE durante creazione DataFrame: {e_df}")
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
    worker = Predictor(model, df_columns)
    try:
        worker.run()
    except KeyboardInterrupt:
        print("\nKeyboard interruption detected.")
        worker.stop()
    except Exception as e:
        print(f"Critical error in the execution of the main worker: {e}")
        traceback.print_exc()