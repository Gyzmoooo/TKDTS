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

#df = pd.DataFrame(data, columns=generate_column_names(TIMESTEPS))

# --- Costanti e Configurazioni 
URL_FETCH = "http://192.168.4.1/"
URL_DELETE = f"{URL_FETCH}delete"
MODEL_PATH = 'C:\\Users\\simon\\Desktop\\Taekwondo-TS\\Codice\\modello\\modello.sav'

TIMESTEPS = 20 # numero di prese dati per calcio
NUM_DATA_PER_TIMESTEP = 24 # totale dati per presa
EXPECTED_DATA_COLUMNS = NUM_DATA_PER_TIMESTEP * TIMESTEPS # colonne csv
EXPECTED_ESP_IDS = [str(i + 1) for i in range(4)] # ids esp
MAX_FETCH_ATTEMPTS = 2
RETRY_DELAY_SECONDS = 2
NUM_BOARDS = 4
SENSORS = ['A', 'G']
AXES = ['x', 'y', 'z']

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

'''
def check_data_completeness(aggregated_text, expected_ids_list):
    if not aggregated_text or not aggregated_text.strip(): return False
    missing_markers = []
    all_found = True
    for esp_id in expected_ids_list:
        start_marker = f"Start{esp_id};"; end_marker = f"End{esp_id};"
        start_found = start_marker in aggregated_text; end_found = end_marker in aggregated_text
        if not start_found or not end_found:
            all_found = False
            if not start_found and not end_found: missing_markers.append(f"ESP {esp_id} (Start & End mancanti)")
            elif not start_found: missing_markers.append(f"ESP {esp_id} (Start mancante)")
            else: missing_markers.append(f"ESP {esp_id} (End mancante)")
    if not all_found:
        print(f"WARN: Dati potenzialmente incompleti. Marcatori mancanti: {', '.join(missing_markers)}")
    return all_found
'''

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
                valid_readings_count = 0
                for reading_set in readings:
                    if not reading_set.strip(): continue
                    parts = reading_set.strip().split(';')
                    accel_data = None; gyro_data = None
                    for part in parts:
                        if part.startswith("A:"): accel_data = part
                        elif part.startswith("G:"): gyro_data = part
                    if accel_data and gyro_data:
                        data_by_esp[esp_id].append((accel_data, gyro_data))
                        valid_readings_count += 1
        if not processed_ids: return None
        return data_by_esp
    except Exception as e:
        print(f"Errore critico durante parsing: {e}"); traceback.print_exc(); return None

def format_data_for_row(data_by_esp, target_timesteps, expected_ids_list):
    try:
        active_esp_ids = [esp_id for esp_id in expected_ids_list if data_by_esp.get(esp_id)]
        if not active_esp_ids: print("Format ERR: Nessun ESP attivo nei dati parsati."); return None
        samples_counts = [len(data_by_esp[esp_id]) for esp_id in active_esp_ids]
        min_samples = min(samples_counts) if samples_counts else 0
        if min_samples == 0 and target_timesteps > 0: print(f"Format ERR: Minimo campioni comuni tra ESP attivi è 0."); return None

        num_to_select = target_timesteps; indices_to_select = []
        if min_samples >= num_to_select:
            indices_to_select = list(range(num_to_select))
        elif min_samples > 0 :
             indices_to_select = list(range(min_samples))
             print(f"Format WARN: Campioni disponibili ({min_samples}) < target ({target_timesteps}). Uso tutti i campioni disponibili.")

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
        return row_data_values
    except Exception as e:
        print(f"Errore critico durante formattazione dati per riga DF: {e}"); traceback.print_exc(); return None

def delete_data_on_master():
    try:
        print("Invio richiesta DELETE a", URL_DELETE)
        response = requests.get(URL_DELETE, timeout=5)
        if response.status_code == 200 and "OK" in response.text:
             print("   -> DELETE confermato dal Master.")
             return True
        else:
             print(f"Errore invio comando DELETE: Status {response.status_code}, Risposta: {response.text}")
             return False
    except requests.exceptions.Timeout:
        print(f"Timeout durante richiesta DELETE a {URL_DELETE}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Errore comunicazione Master (DELETE): {e}")
        return False
    except Exception as e:
        print(f"Errore generico durante DELETE: {e}")
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
            status_msg = f"\n===== Ciclo {cycle_counter}: Inizio Acquisizione ====="
            print(status_msg) # Log Console

            parsed_data = None; fetch_successful = False
            sufficient_data_received = False; delete_attempted = False

            # 1. Fetch Loop
            for attempt in range(MAX_FETCH_ATTEMPTS):
                if not self._is_running: break

                status_msg = f"Tentativo Fetch {attempt + 1}/{MAX_FETCH_ATTEMPTS}..."
                print(status_msg) # Log Console

                try:
                    response = requests.get(URL_FETCH, timeout=10)
                    response.raise_for_status()
                    current_aggregated_data = response.text

                    if not current_aggregated_data or not current_aggregated_data.strip():
                        msg = "   Buffer Master vuoto."
                        print(msg)
                        if attempt < MAX_FETCH_ATTEMPTS - 1: time.sleep(RETRY_DELAY_SECONDS)
                        continue
                    else:
                        print(f"   Dati ricevuti (Len: {len(current_aggregated_data)}). Parsing...")
                        current_parsed_data = parse_aggregated_data(current_aggregated_data, EXPECTED_ESP_IDS)

                        if current_parsed_data:
                            parsed_data = current_parsed_data
                            fetch_successful = True
                            active_ids = [k for k, v in parsed_data.items() if v]
                            if active_ids:
                                min_samples = min(len(parsed_data[eid]) for eid in active_ids) if active_ids else 0
                                msg = f"   Parsing OK. Campioni min: {min_samples}/{TIMESTEPS}"
                                print(msg)
                                all_active_have_enough = all(len(parsed_data[eid]) >= TIMESTEPS for eid in active_ids)
                                if all_active_have_enough:
                                    msg = "   --> Dati sufficienti ricevuti! Procedo."
                                    print(msg)
                                    sufficient_data_received = True
                                    break
                                else:
                                    msg = "   Dati insufficienti. Attendo/Riprovo..."
                                    print(msg)
                            else:
                                msg = "   Parsing OK, ma nessun ESP attivo con dati."
                                print(msg)
                        else:
                            msg = "   Errore durante il parsing dei dati ricevuti."
                            print(msg)

                        if not sufficient_data_received and attempt < MAX_FETCH_ATTEMPTS - 1:
                            time.sleep(RETRY_DELAY_SECONDS)

                except requests.exceptions.Timeout:
                    msg = f"   Timeout (Tentativo {attempt + 1})."
                    print(msg)
                    if attempt < MAX_FETCH_ATTEMPTS - 1: time.sleep(RETRY_DELAY_SECONDS)
                except requests.exceptions.RequestException as e:
                    msg = f"   Errore connessione/HTTP (Tentativo {attempt + 1}): {e}"
                    print(msg)
                    if attempt < MAX_FETCH_ATTEMPTS - 1: time.sleep(RETRY_DELAY_SECONDS)
                except Exception as e:
                    msg = f"   Errore imprevisto fetch (Tentativo {attempt + 1}): {e}"
                    print(msg); traceback.print_exc()
                    self._is_running = False
                    break

            if not self._is_running: break

            # 2. Processa SOLO se dati sufficienti
            if sufficient_data_received and parsed_data:
                msg = "Formattazione dati per DataFrame..."
                print(msg)
                formatted_row_data = format_data_for_row(parsed_data, TIMESTEPS, EXPECTED_ESP_IDS)

                if formatted_row_data is not None:
                    msg = "Creazione DataFrame..."
                    print(msg)
                    try:
                        kick_dataframe = pd.DataFrame([formatted_row_data], columns=self.dataframe_columns)
                        if kick_dataframe.isnull().values.any():
                             nan_count = kick_dataframe.isnull().sum().sum()
                             msg = f"ATTENZIONE: {nan_count} valori NaN nel DataFrame prima della predizione!"
                             print(msg)

                        # --- PULIZIA DATI ESP ---
                        msg = "Pulizia buffer dati sul Master ESP..."
                        print(msg)
                        delete_attempted = True
                        if not delete_data_on_master():
                             msg = "WARN: Fallita pulizia buffer Master."
                             print(msg)

                        # 4. Esegui Predizione
                        msg = "Esecuzione predizione modello..."
                        print(msg)
                        x_numpy = kick_dataframe.values
                        y_pred = self.model.predict(x_numpy)
                        prediction_result = y_pred[0]

                        msg = f"---> Predizione Ciclo {cycle_counter}: {prediction_result} <---"
                        print(msg) # Log console

                    except Exception as e_df_pred:
                        msg = f"\nERRORE durante creazione DataFrame o Predizione: {e_df_pred}"
                        print(msg); traceback.print_exc()
                        self._is_running = False
                else:
                    msg = "Formattazione dati fallita. Impossibile procedere."
                    print(msg)
                    if fetch_successful and not delete_attempted:
                        msg = "Tento pulizia buffer Master nonostante errore formattazione..."
                        print(msg)
                        delete_data_on_master()

            # Gestione casi fallimento Fetch / dati insufficienti
            elif fetch_successful and parsed_data:
                msg = f"Dati insufficienti raccolti dopo {MAX_FETCH_ATTEMPTS} tentativi. Riprovo."
                print(msg)
                if not delete_attempted:
                    msg = "Tento pulizia buffer Master (dati insufficienti)..."
                    print(msg)
                    delete_data_on_master()
            elif fetch_successful and not parsed_data:
                msg = "Fetch OK ma parsing fallito. Riprovo."
                print(msg)
                if not delete_attempted:
                    msg = "Tento pulizia buffer Master (parsing fallito)..."
                    print(msg)
                    delete_data_on_master()
            elif not fetch_successful:
                msg = f"Fetch fallito o interrotto dopo {MAX_FETCH_ATTEMPTS} tentativi. Riprovo."
                print(msg)

            if self._is_running:
                time.sleep(1)

        msg = "--- Thread Worker Terminato ---"
        print(msg) # Log console

    def stop(self):
        print("Richiesta di arresto per il thread worker...") # Log Console
        self._is_running = False


# --- Esecuzione Principale ---
if __name__ == "__main__":
    print("Caricamento modello...")
    try:
        modello_caricato = load(MODEL_PATH)
        print("Modello caricato con successo.")
    except FileNotFoundError:
        print(f"ERRORE CRITICO: File modello non trovato in {MODEL_PATH}")
        sys.exit(1)
    except Exception as e:
        print(f"ERRORE CRITICO: Impossibile caricare il modello: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("Generazione nomi colonne DataFrame...")
    colonne_df = generate_column_names(TIMESTEPS)
    if not colonne_df or len(colonne_df) != EXPECTED_DATA_COLUMNS:
        print("ERRORE CRITICO: Nomi colonne non validi o numero errato.")
        sys.exit(1)
    print(f"Nomi colonne generati ({len(colonne_df)}).")

    print("Avvio del processo di acquisizione e predizione (senza GUI)...")
    worker = Worker(modello_caricato, colonne_df)
    try:
        worker.run()
    except KeyboardInterrupt:
        print("\nInterruzione da tastiera rilevata. Arresto del worker.")
        worker.stop()
    except Exception as e:
        print(f"Errore critico nell'esecuzione del worker principale: {e}")
        traceback.print_exc()
    finally:
        print("Processo principale terminato.")