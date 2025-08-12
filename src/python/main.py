import sys
import time
import re
import traceback
import os

import requests
import pandas as pd
from joblib import load
import numpy as np

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

    def compute_svm(self, df):
        out_array = np.array()
        in_array = df.to_numpy()
        
        for row in in_array:
            temp_list = []
            for column in range(1, len(row), 3):
                smv = np.sqrt(row[column]**2 + row[column+1]**2 + row[column+2]**2)
                temp_list.append(smv)
            out_array = np.append(out_array, temp_list)
        
        return out_array
    
    def classify_sample(self, smv_array, threshold):
        if np.mean(smv_array) > threshold:
            return "Calcio"
        return "Fermo"
    
    def crea_gruppi_(dati_input, dimensione_gruppo=20, min_calcio_len=2):
        """
        Crea gruppi di dimensione fissa garantita, senza sovrapposizioni di indici.
        I gruppi sono centrati solo su blocchi di "Calcio" con una lunghezza minima.

        Args:
            dati_input (np.ndarray): L'array 1D con etichette "Calcio" e "Fermo".
            dimensione_gruppo (int): La dimensione esatta di ogni gruppo.
            min_calcio_len (int): La lunghezza minima di una sequenza di "Calcio"
                                per poter essere considerata come centro di un gruppo.

        Returns:
            list: Una lista di liste, dove ogni lista interna è un gruppo di tuple
                (etichetta, indice_originale).
        """
        
        # 1. Identifica tutti i blocchi (Fermo e Calcio)
        blocchi = []
        if len(dati_input) == 0: return []
        label_corrente, start_index = dati_input[0], 0
        for i in range(1, len(dati_input)):
            if dati_input[i] != label_corrente:
                blocchi.append({'label': label_corrente, 'start': start_index, 'end': i})
                label_corrente, start_index = dati_input[i], i
        blocchi.append({'label': label_corrente, 'start': start_index, 'end': len(dati_input)})

        # 2. Filtra per tenere solo i blocchi di Calcio che superano la lunghezza minima
        blocchi_calcio_validi = [
            b for b in blocchi 
            if b['label'] == 'Calcio' and (b['end'] - b['start']) >= min_calcio_len
        ]

        gruppi_finali = []
        ultimo_indice_usato = -1

        # 3. Itera sui blocchi di Calcio validi per costruire i gruppi sequenzialmente
        for calcio_block in blocchi_calcio_validi:
            calcio_start = calcio_block['start']
            
            # Se questo blocco di Calcio è già stato "consumato" dal padding del gruppo precedente, saltalo
            if calcio_start <= ultimo_indice_usato:
                continue

            calcio_end = calcio_block['end']
            num_calcio = calcio_end - calcio_start

            if num_calcio >= dimensione_gruppo:
                # Se il blocco è già di 20 o più, non c'è spazio per padding.
                # Il gruppo sarà composto solo dal blocco di Calcio.
                start_gruppo, end_gruppo = calcio_start, calcio_end
            else:
                padding_necessario = dimensione_gruppo - num_calcio
                pre_padding_target = padding_necessario // 2
                post_padding_target = padding_necessario - pre_padding_target

                # Calcola i limiti per il prelievo del padding
                # Il padding iniziale può essere preso solo da dopo l'ultimo gruppo creato
                limite_pre = ultimo_indice_usato + 1
                # Il padding finale può essere preso fino alla fine dell'array
                limite_post = len(dati_input)

                # Calcola il padding disponibile e prelevalo
                pre_padding_disponibile = calcio_start - limite_pre
                post_padding_disponibile = limite_post - calcio_end

                pre_da_prendere = min(pre_padding_disponibile, pre_padding_target)
                post_da_prendere = min(post_padding_disponibile, post_padding_target)
                
                # Logica di compensazione
                mancanti = padding_necessario - (pre_da_prendere + post_da_prendere)
                if mancanti > 0:
                    extra_post = min(mancanti, post_padding_disponibile - post_da_prendere)
                    post_da_prendere += extra_post
                    mancanti -= extra_post
                if mancanti > 0:
                    extra_pre = min(mancanti, pre_padding_disponibile - pre_da_prendere)
                    pre_da_prendere += extra_pre
                
                # Se ancora mancano elementi, questo gruppo non può essere formato a 20.
                if pre_da_prendere + post_da_prendere + num_calcio < dimensione_gruppo:
                    print(f"ATTENZIONE: Blocco 'Calcio' (indici {calcio_start}-{calcio_end-1}) non ha abbastanza elementi circostanti per formare un gruppo di {dimensione_gruppo}. Saltato.")
                    continue

                start_gruppo = calcio_start - pre_da_prendere
                end_gruppo = calcio_end + post_da_prendere

            # Estrai il gruppo e aggiorna l'ultimo indice usato
            indici_gruppo = np.arange(start_gruppo, end_gruppo)
            etichette_gruppo = dati_input[indici_gruppo]
            gruppo = list(zip(etichette_gruppo, indici_gruppo))
            gruppi_finali.append(gruppo)
            
            ultimo_indice_usato = end_gruppo - 1
                
        return np.array(gruppi_finali)
    
    def split(self, smv_matrix):
        samples_array = np.array()
        for i in range(len(smv_matrix)):
            samples_array = np.append(samples_array, self.classify_sample(smv_matrix[i]))
        
        
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
                    parsed = self.data_processor.parse(raw)
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