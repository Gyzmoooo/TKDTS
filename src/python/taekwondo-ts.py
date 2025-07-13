import sys
import requests
import time
import re
import traceback
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
        for sensor_type in SENSORS:
            for board_id in range(1, NUM_BOARDS + 1):
                for axis in AXES:
                    col_name = f"{sensor_type}{board_id}{axis}{suffix}"
                    column_names.append(col_name)
    return column_names

def parse_data(raw_data):
    try:    
        number_pattern = r'-?\d+\.?\d*'
        data = [float(num_str) for num_str in re.findall(number_pattern, raw_data)]

        if not data: 
            print("No ESP32 active in parsed data")
            return None
        if len(data) > N_DATA: 
            raise Exception("More data than expected got parsed")
        if len(data) < N_DATA: 
            print("Less data than expected got parsed")

    except Exception as e:
        print(f"Critical error during parsing: {e}"); traceback.print_exc(); return None
    
    return data

#df = pd.DataFrame(parse_data(), columns=generate_column_names(TIMESTEPS))

def parse_aggregated_data(aggregated_text, expected_ids_list):
    try:
        data_by_esp = {esp_id: [] for esp_id in expected_ids_list}
        processed_ids = set()
        for esp_id in expected_ids_list:
            start_marker = f"Start{esp_id};"
            end_marker = f"End{esp_id};"
            last_start_idx = aggregated_text.rfind(start_marker)
            last_end_idx = aggregated_text.find(end_marker, last_start_idx + len(start_marker))
            if last_end_idx != -1 and last_start_idx != -1:
                esp_block = aggregated_text[last_start_idx + len(start_marker) : last_end_idx]
                processed_ids.add(esp_id)
                readings = esp_block.split(f'ID{esp_id};')
                if readings and readings[0] == '': readings = readings[1:]
                for reading_set in readings:
                    parts = reading_set.strip().split(';')
                    accel_data = None
                    gyro_data = None
                    for part in parts:
                        if part.startswith("A:"): accel_data = part
                        if part.startswith("G:"): gyro_data = part
                    if accel_data and gyro_data:
                        data_by_esp[esp_id].append((accel_data, gyro_data))
        if not processed_ids: return None
        print(data_by_esp)
        return data_by_esp
    except Exception as e:
        print(f"Errore critico durante parsing: {e}"); traceback.print_exc(); return None

def format_data_for_row(data_by_esp, target_timesteps, expected_ids_list):
    try:
        active_esp_ids = [esp_id for esp_id in expected_ids_list if data_by_esp.get(esp_id)]
        samples_counts = [len(data_by_esp[esp_id]) for esp_id in active_esp_ids]
        min_samples = min(samples_counts)
        
        if target_timesteps > 0 and len(samples_counts) != 4: 
            raise Exception(f"{4 - len(samples_counts)} of the ESP32 didn't send any data")

        if min_samples >= target_timesteps:
            indices_to_select = list(range(target_timesteps))
        elif min_samples > 0 :
             raise Exception(f"Samples available ({min_samples}) < target ({target_timesteps})")

        row_data_values = []; contains_real_errors = False
        num_actual_indices = len(indices_to_select)

        for timestep_index_target in range(target_timesteps):
            time_index_real = -1; use_padding = False
            if timestep_index_target < num_actual_indices:
                time_index_real = indices_to_select[timestep_index_target]
            else:
                use_padding = True

            for sensor_type in SENSORS:
                for board_id_str in expected_ids_list:
                    data_str_pair = None; is_active_esp = board_id_str in active_esp_ids
                    if is_active_esp and not use_padding and time_index_real != -1 and time_index_real < len(data_by_esp[board_id_str]):
                        data_str_pair = data_by_esp[board_id_str][time_index_real]

                    if data_str_pair:
                        target_str = data_str_pair[0] if sensor_type == 'A' else data_str_pair[1]
                        match = re.search(r"[AG]:([-\d.]+),([-\d.]+),([-\d.]+)", target_str)
                        if match:
                            values = [match.group(1), match.group(2), match.group(3)]
                            for axis_index, axis_val_str in enumerate(values):
                                try:
                                    numerical_value = float(axis_val_str)
                                    row_data_values.append(numerical_value)
                                except ValueError:
                                    print(f"   ERR Format: Conversione float fallita '{axis_val_str}' (ESP {board_id_str}, Sens {sensor_type}, TimeReal {time_index_real})")
                                    num_nan_to_add = 3 - axis_index
                                    row_data_values.extend([np.nan] * num_nan_to_add)
                                    contains_real_errors = True; break
                            if contains_real_errors: break
                        else:
                            print(f"   ERR Format: Regex fallita su '{target_str}' (ESP {board_id_str}, Sens {sensor_type}, TimeReal {time_index_real})")
                            row_data_values.extend([np.nan, np.nan, np.nan])
                            contains_real_errors = True; break
                    else:
                        row_data_values.extend([np.nan, np.nan, np.nan])
                if contains_real_errors: break
            if contains_real_errors: break

        if contains_real_errors:
            print("ERRORE REALE durante formattazione dati. Riga invalida generata."); return None
        if len(row_data_values) != EXPECTED_DATA_COLUMNS:
            print(f"ERRORE CRITICO LUNGHEZZA DATI: Generati {len(row_data_values)} valori != {EXPECTED_DATA_COLUMNS} attesi."); return None
        print(row_data_values)
        return row_data_values
    except Exception as e: traceback.print_exc() ; return None

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