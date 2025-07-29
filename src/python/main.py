import sys
import time
import re
import traceback
import os

import requests
import pandas as pd
from joblib import load

# --- Costanti e Configurazioni 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = BASE_DIR.strip("src\python") + '\model\\rf_model.sav'

URL_FETCH = "http://192.168.4.1/"
URL_DELETE = f"{URL_FETCH}delete"

TIMESTEPS = 20
NUM_BOARDS = 4
SENSORS = ['A', 'G']
AXES = ['x', 'y', 'z']

EXPECTED_ESP_IDS = [str(i + 1) for i in range(NUM_BOARDS)]
NUM_DATA_PER_TIMESTEP = len(SENSORS) * len(AXES) * NUM_BOARDS
EXPECTED_DATA_COLUMNS = TIMESTEPS * NUM_DATA_PER_TIMESTEP

MAX_FETCH_ATTEMPTS = 2
RETRY_DELAY_SECONDS = 2

class UnsufficientSamples(Exception):
    """Exception raised when the number of samples received is less than expected"""
    def __init__(self, e_samples, r_samples, msg="Samples received is less than expected"):
        self.message = (
            f"{msg}: Received {r_samples}, Expected {e_samples}. "
        )
        super().__init__(self.message)

class UncompleteData(Exception):
    """Exception raised when at least one of the ESP32 didn't send any data"""
    def __init__(self, e_ids, r_ids, msg=" of the ESP32 didn't send any data"):
        self.empty_ids = [esp_id for esp_id in e_ids if esp_id not in r_ids]
        self.message = (
            f"{len(self.empty_ids)}{msg}. IDs of the boards: {self.empty_ids}. "
        )
        super().__init__(self.message)

class DataProcessor:
    def __init__(self, url_delete, timesteps, num_boards, sensors, axes, expected_esp_ids):
        self.url_delete = url_delete
        self.timesteps = timesteps
        self.num_boards = num_boards
        self.sensors = sensors
        self.axes = axes
        self.expected_esp_ids = expected_esp_ids


    def generate_column_names(self):
        column_names = []
        for i in range(self.timesteps):
            suffix = f"_{i + 1}"
            for board_id in range(1, self.num_boards + 1):
                for sensor_type in self.sensors:
                    for axis in self.axes:
                        col_name = f"{sensor_type}{board_id}{axis}{suffix}"
                        column_names.append(col_name)
        return column_names

    def parse(self, raw_data):
        number_pattern = r'-?\d+\.?\d*'
        esp_data = []
        
        # Parses all data and stores them in a list
        for esp_id in self.expected_esp_ids:
            last_start_idx = raw_data.rfind(f"Start{esp_id};")
            last_end_idx = raw_data.find(f"End{esp_id};", last_start_idx + len(f"Start{esp_id};"))
            if last_end_idx != -1 and last_start_idx != -1:
                esp_block = raw_data[last_start_idx + len(f"Start{esp_id};") : last_end_idx]
                esp_block = esp_block.replace(f"ID{esp_id}", "")
                esp_data.append([float(num_str) for num_str in re.findall(number_pattern, esp_block)])
            else: esp_data.append([])

        return esp_data

    def format(self, esp_data):
        # Verifies data completeness and number of samples
        #print(esp_data)
        active_esp_ids = [str(esp_id + 1) for esp_id in range(len(esp_data)) if esp_data[esp_id] != []]
        samples_counts = [(len(esp_data[int(esp_id) - 1]) / 6) for esp_id in active_esp_ids]
        min_samples = int(min(samples_counts))
        #print(f"QUESTO è len(samples_counts): {len(samples_counts)}")
        #print(f"QUESTO è self.num_boards: {self.num_boards}")
        if self.timesteps > 0 and len(samples_counts) != self.num_boards: 
            raise UncompleteData(e_ids=self.expected_esp_ids, r_ids=active_esp_ids)
        elif min_samples < self.timesteps:
            raise UnsufficientSamples(e_samples=self.timesteps, r_samples=min_samples)

        # Eliminates excessive data
        for esp in range(self.num_boards):
            data_to_eliminate = int(-6 * (samples_counts[int(esp)] - min_samples) - 1)
            del esp_data[esp][-1:data_to_eliminate:-1]

        # Conversion in list of lists format, where each sublist contains a single sample
        unfiltered_data = []
        for i in range(min_samples):
            sample = []
            for j in range(self.num_boards):
                start = i * len(self.sensors) * len(self.axes)
                end = start + len(self.sensors) * len(self.axes)
                sample.extend(esp_data[j][start:end])
            unfiltered_data.append(sample)
        
        # Sample reduction to 20
        data = [[]]
        step = (len(unfiltered_data) - 1) / (self.timesteps - 1)
        for i in range(self.timesteps):
            index = round(i * step)
            for j in unfiltered_data[min(index, len(unfiltered_data) - 1)]: data[0].append(j)

        return data
    
    def correct(self, stringa_dati: str, num_sensori: int) -> str:
        # Pulisce e divide i dati grezzi in singole voci, rimuovendo spazi e voci vuote.
        voci = [v.strip() for v in stringa_dati.strip().split(';') if v.strip()]

        # Un dizionario per contenere la lista delle misurazioni per ogni sensore.
        # Una misurazione è una lista di parti di dati (es. ['A:...', 'G:...'])
        misure_sensori = {f'ID{i}': [] for i in range(1, num_sensori + 1)}

        id_corrente = None
        misura_corrente = []

        # Pattern Regex per identificare i marcatori ID (es. ID1, ID2, etc.)
        id_pattern = re.compile(r'^ID(\d+)$')
        # Pattern Regex per identificare i marcatori Start/End
        marker_pattern = re.compile(r'^(Start|End)(\d+)$')

        for voce in voci:
            id_match = id_pattern.match(voce)
            marker_match = marker_pattern.match(voce)

            if id_match:
                # Se troviamo un nuovo ID, la misurazione precedente è completa.
                if id_corrente and misura_corrente:
                    misure_sensori[id_corrente].append(misura_corrente)

                # Inizia una nuova misurazione
                id_corrente = voce
                misura_corrente = []
            elif marker_match:
                # Se troviamo un marcatore Start/End, la misurazione precedente è completa.
                # Questi marcatori verranno ignorati e ricostruiti da zero.
                if id_corrente and misura_corrente:
                    misure_sensori[id_corrente].append(misura_corrente)

                # Resetta lo stato, poiché i marcatori originali indicano un'interruzione.
                id_corrente = None
                misura_corrente = []
            else:
                # Questa è una parte di dati (come 'A:...' o 'G:...')
                if id_corrente:
                    misura_corrente.append(voce)

        # Assicura che l'ultima misurazione nella stringa venga aggiunta.
        if id_corrente and misura_corrente:
            misure_sensori[id_corrente].append(misura_corrente)

        # Ricostruisce la stringa finale
        blocchi_output = []
        for i in range(1, num_sensori + 1):
            sensor_id = f'ID{i}'
            misure = misure_sensori.get(sensor_id)

            # Procede solo se ci sono dati effettivi per questo sensore
            if misure:
                # Inizia il blocco per questo sensore
                parti_blocco = [f'Start{i}']
                for misura in misure:
                    parti_blocco.append(sensor_id)
                    parti_blocco.extend(misura)
                # Termina il blocco
                parti_blocco.append(f'End{i}')
                blocchi_output.append(';'.join(parti_blocco))

        # Unisce tutti i blocchi dei sensori con un punto e virgola e ne aggiunge uno
        # finale per coerenza con il formato di input.
        return ';'.join(blocchi_output) + ';' if blocchi_output else ''


    def delete_data_on_master(self):
        try:
            print("Sending DELETE request at", self.url_delete)
            response = requests.get(self.url_delete, timeout=5)
            if response.status_code == 200 and "OK" in response.text:
                return True
            else:
                print(f"Error sending DELETE command: Status {response.status_code}, Response: {response.text}")
                return False
        except requests.exceptions.Timeout:
            print(f"Timeout during DELETE request at {self.url_delete}")
            return False
        except requests.exceptions.RequestException as e:
            print(f"Error during communication with Master (DELETE): {e}")
            return False
        except Exception as e:
            print(f"Generic error during DELETE: {e}")
            return False


class Predictor:
    def __init__(self, model, url_fetch, max_fetch_attempts, retry_delay_seconds, data_processor: DataProcessor):
        self.model = model
        self.data_processor = data_processor
        self.columns = self.data_processor.generate_column_names()
        self._is_running = True
        self.url_fetch = url_fetch
        self.max_fetch_attempts = max_fetch_attempts
        self.retry_delay_seconds = retry_delay_seconds

    """
    def extract_data(self):
        return 0

    def split():
        return 0
    """

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
        while self._is_running:
            data = []
            try:
                response = requests.get(self.url_fetch, timeout=10)
                response.raise_for_status()
                raw = response.text
                if raw == "aspettaciola":
                    print("Waiting for the data...")
                    #time.sleep(RETRY_DELAY_SECONDS)
                    continue
                elif raw == "":
                    print("Nothing there! :(")
                else:
                    corrected = self.data_processor.correct(raw, 4)
                    print(corrected)
                    parsed = self.data_processor.parse(corrected)
                    data = self.data_processor.format(parsed)

            except requests.exceptions.Timeout:
                print(f"Timeout di requests.")
                break
            except requests.exceptions.RequestException as e:
                print(f"Errore connessione/HTTP: {e}")
                time.sleep(self.retry_delay_seconds)
                continue
            except UncompleteData as e:
                print(f"Error raised while formatting: {e}")
                time.sleep(self.retry_delay_seconds) 
                continue
            except UnsufficientSamples as e: 
                print(f"Error raised while formatting: {e}")
                time.sleep(self.retry_delay_seconds)
                continue
            except ValueError as e:
                print(e)
                time.sleep(self.retry_delay_seconds)
                continue

            except Exception as e:
                print(f"Unexpected error: {e}")
                traceback.print_exc()
                self._is_running = False
            
            if data:
                print(data)
                try:
                    # data = self.extract_data()
                    kick_df = pd.DataFrame(data, columns=self.columns)
                    # Nel codice finale, qui ci va la funzione che splitta e un ciclo for per iterare in ogni calcio

                    print("Pulizia buffer dati sul Master ESP...")
                    if not self.data_processor.delete_data_on_master():
                        print("WARN: Fallita pulizia buffer Master.")

                    result = self.predict(kick_df)
                    print(f"---> Predizione calcio: {result} <---")

                except Exception as e_df:
                    print(f"\nERRORE durante creazione DataFrame o predizione: {e_df}")
                    traceback.print_exc()
                    self._is_running = False

        if self._is_running:
            time.sleep(1)

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

    data_processor = DataProcessor(
        url_delete=URL_DELETE,
        timesteps=TIMESTEPS,
        num_boards=NUM_BOARDS,
        sensors=SENSORS,
        axes=AXES,
        expected_esp_ids=EXPECTED_ESP_IDS
    )

    df_columns = data_processor.generate_column_names()
    if not df_columns or len(df_columns) != EXPECTED_DATA_COLUMNS:
        print("CRITICAL ERROR: Invalid column names or wrong number.")
        sys.exit(1)

    print("Starting the process of acquisition and prediction...")
    predictor = Predictor(
        model=model, 
        data_processor=data_processor,
        url_fetch=URL_FETCH,
        max_fetch_attempts=MAX_FETCH_ATTEMPTS,
        retry_delay_seconds=RETRY_DELAY_SECONDS
    )

    try:
        predictor.run()
    except KeyboardInterrupt:
        print("\nKeyboard interruption detected.")
        predictor.stop()
    except Exception as e:
        print(f"Critical error in the execution of the main worker: {e}")
        traceback.print_exc()