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
    
    # Parses all data and stores them in a list
    for esp_id in expected_ids_list:
        last_start_idx = raw_data.rfind(f"Start{esp_id};")
        last_end_idx = raw_data.find(f"End{esp_id};", last_start_idx + len(f"Start{esp_id};"))
        if last_end_idx != -1 and last_start_idx != -1:
            esp_block = raw_data[last_start_idx + len(f"Start{esp_id};") : last_end_idx]
            esp_block = esp_block.replace(f"ID{esp_id}", "")
            esp_data.append([float(num_str) for num_str in re.findall(number_pattern, esp_block)])
        else: esp_data.append([])

def format_data(esp_data, target_timesteps, num_boards):
    # Verifies data completeness and number of samples 
    active_esp_ids = [str(esp_id + 1) for esp_id in range(len(esp_data)) if esp_data[esp_id] != []]
    samples_counts = [(len(esp_data[int(esp_id) - 1]) / 6) for esp_id in active_esp_ids]
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
    def __init__(self, model, kick_df):
        self.model = model
        self.kick_df = kick_df
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

raw = "Start1;ID1;A:-3.8895,-0.1964,-8.1383;G:-0.0511,-0.5678,-0.0048;ID1;A:-3.2620,0.1868,-9.0915;G:-0.0830,-0.0958,0.0442;ID1;A:-3.1902,-0.1198,-9.3118;G:0.0021,-0.6635,-0.0138;ID1;A:-3.8751,0.1245,-8.8951;G:0.2496,-0.4241,0.2453;ID1;A:-3.9374,-0.2060,-9.0484;G:0.0442,-0.3102,0.2065;ID1;A:-4.5889,0.1629,-9.7142;G:0.1485,-0.1431,0.2815;ID1;A:-3.3913,0.1341,-8.5742;G:0.7849,-0.6183,0.4315;ID1;A:-4.3541,-0.1629,-8.1431;G:0.5444,-0.5917,0.2123;ID1;A:-5.2499,0.0240,-8.1335;G:0.2857,-0.1362,0.2496;ID1;A:-4.7326,-0.5461,-8.1095;G:1.2249,-0.7785,0.6540;ID1;A:-4.8667,-0.1724,-8.4544;G:0.2916,-0.1751,0.5401;ID1;A:-5.1062,-0.3066,-8.0521;G:-0.4363,0.3102,0.2251;ID1;A:-4.9625,-0.1820,-8.4784;G:-0.1873,0.0548,0.0984;ID1;A:-5.0535,0.0766,-7.8078;G:-0.2054,-0.0676,0.0149;ID1;A:-5.2355,0.0240,-8.4257;G:-0.1586,0.1016,-0.0724;ID1;A:-5.0774,-0.1820,-8.2293;G:-0.0138,-0.3927,0.1437;ID1;A:-5.2738,0.0814,-7.8365;G:-0.0665,-0.1612,0.0681;ID1;A:-5.3026,0.1150,-7.9610;G:-0.0532,-0.0782,0.1336;ID1;A:-5.2882,0.2156,-8.0760;G:-0.0958,-0.1703,0.1426;ID1;A:-5.4559,0.3880,-7.7886;G:-0.0814,-0.1080,0.1277;ID1;A:-5.3984,0.4407,-7.9371;G:-0.0750,-0.0575,0.1128;ID1;A:-5.4511,0.4694,-7.8892;G:-0.0952,-0.1362,0.0527;ID1;A:-5.5756,0.5269,-7.6114;G:-0.0394,-0.0692,0.0591;ID1;A:-5.5996,0.5269,-7.6305;G:-0.0442,0.0952,0.0623;ID1;A:-5.2307,0.6850,-8.3155;G:-0.2059,0.0639,-0.1703;ID1;A:-5.7624,0.7999,-7.5826;G:-0.2474,0.0027,-0.2762;ID1;A:-5.7385,0.7568,-7.6018;G:0.0740,0.1229,-0.0521;ID1;A:-5.5469,0.5604,-7.8700;G:-0.0032,0.0660,-0.1011;ID1;A:-5.7193,0.6898,-7.7072;G:-0.0330,0.0660,-0.0591;ID1;A:-5.5948,0.7377,-7.7072;G:0.0495,0.1368,0.0234;ID1;A:-5.4894,0.6083,-7.9179;G:0.1171,0.0410,0.0074;ID1;A:-5.5517,0.7808,-7.7311;G:0.1054,0.1075,0.0474;ID1;A:-5.4223,0.6323,-7.9898;G:0.1426,0.0080,0.0553;ID1;A:-5.1301,0.1916,-7.8988;G:0.0543,0.1203,-0.0410;ID1;A:-5.6139,0.6850,-7.7503;G:0.1708,0.0628,0.0080;ID1;A:-5.2690,0.5413,-8.1766;G:0.1926,-0.2602,0.1399;End1;Start2;ID2;A:-3.5003,0.1437,-9.1506;G:-0.0027,-0.0750,-0.0218;ID2;A:-3.0550,0.1389,-9.0214;G:-0.0798,0.4778,-0.0277;ID2;A:-2.6432,-0.1772,-8.9208;G:0.0585,0.9094,-0.0788;ID2;A:-2.3750,0.1389,-9.5289;G:-0.1101,0.7226,-0.0521;ID2;A:-2.2266,0.1437,-9.3230;G:0.0261,0.0341,-0.0239;ID2;A:-2.2649,0.1772,-9.2847;G:0.0250,0.3863,-0.0298;ID2;A:-2.0159,0.0239,-9.5241;G:-0.0314,-0.1383,-0.0128;ID2;A:-1.9489,0.0718,-9.3565;G:-0.0495,0.0926,-0.0197;ID2;A:-2.1596,0.2155,-9.1602;G:-0.0399,0.3054,-0.0591;ID2;A:-2.1117,0.1341,-9.2751;G:-0.0255,0.3182,-0.0447;ID2;A:-1.6664,0.0670,-9.5146;G:0.0415,0.0649,-0.0282;ID2;A:-1.9297,0.1484,-9.3518;G:-0.0500,0.0090,-0.0160;ID2;A:-2.0255,0.0431,-9.7971;G:-0.0197,-0.1139,-0.0080;ID2;A:-2.0973,0.1053,-9.6486;G:0.0032,0.0160,-0.0165;ID2;A:-1.9441,0.2251,-9.6295;G:-0.0170,0.0835,-0.0314;ID2;A:-1.8292,0.1149,-9.5002;G:-0.0208,0.1336,-0.0298;ID2;A:-1.8004,0.0479,-9.7396;G:-0.0255,0.0724,-0.0234;ID2;A:-1.7957,0.2059,-9.5337;G:-0.0202,0.0074,-0.0234;ID2;A:-1.6328,0.1628,-9.4044;G:-0.0101,0.1410,-0.0266;ID2;A:-1.6712,0.1341,-9.6295;G:-0.0223,0.0500,-0.0181;ID2;A:-1.7717,0.1341,-9.5193;G:-0.0255,-0.0165,-0.0208;ID2;A:-1.6472,0.1245,-9.6007;G:-0.0250,0.0420,-0.0266;ID2;A:-1.6807,0.1389,-9.6438;G:-0.0234,0.0282,-0.0266;ID2;A:-1.6664,0.1149,-9.5577;G:-0.0229,0.0144,-0.0181;ID2;A:-1.6089,0.1867,-9.5768;G:-0.0213,0.0410,-0.0250;ID2;A:-1.6568,0.1101,-9.5529;G:-0.0202,0.0197,-0.0218;ID2;A:-1.6568,0.1532,-9.5864;G:-0.0181,0.0208,-0.0186;ID2;A:-1.6137,0.1532,-9.5768;G:-0.0192,0.0335,-0.0255;ID2;A:-1.5802,0.1341,-9.5385;G:-0.0223,0.0101,-0.0160;ID2;A:-1.6664,0.1676,-9.5050;G:-0.0085,0.0043,-0.0170;ID2;A:-1.6281,0.1437,-9.6007;G:-0.0202,0.0090,-0.0186;ID2;A:-1.6424,0.1101,-9.6151;G:-0.0176,0.0181,-0.0245;ID2;A:-1.6185,0.1006,-9.5002;G:-0.0170,0.0059,-0.0255;ID2;A:-1.6520,0.1580,-9.5481;G:-0.0170,0.0170,-0.0234;ID2;A:-1.6424,0.1341,-9.5864;G:-0.0208,0.0186,-0.0202;ID2;A:-1.6472,0.1628,-9.6534;G:-0.0218,0.0021,-0.0176;ID2;A:-1.6568,0.0622,-9.5624;G:-0.0154,0.0053,-0.0234;ID2;A:-1.7190,0.1484,-9.6630;G:-0.0234,0.0176,-0.0255;ID2;A:-1.6520,0.1293,-9.5624;G:-0.0080,0.0170,-0.0239;ID2;A:-1.6424,0.1580,-9.6103;G:-0.0138,0.0069,-0.0229;End2;Start3;ID3;A:1.1827,0.3735,-9.9742;G:0.1639,0.3948,-0.0245;ID3;A:0.4597,0.0670,-9.6965;G:0.3038,0.8423,-0.1293;ID3;A:1.7765,-0.1867,-10.4435;G:0.2868,0.6045,-0.0915;ID3;A:1.7095,-0.0527,-9.9455;G:-0.0351,0.0266,-0.0532;ID3;A:0.7326,0.3400,-8.7053;G:0.4108,1.0674,-0.0319;ID3;A:4.2952,-1.7957,-13.3453;G:0.1027,0.7838,0.2496;ID3;A:2.6528,-0.8236,-10.7356;G:0.0298,0.3698,-0.0495;ID3;A:2.3607,-0.5602,-9.8928;G:-0.0756,0.0761,-0.0665;ID3;A:2.4948,-0.2490,-9.6869;G:-0.0490,0.0958,-0.1022;ID3;A:2.4852,-0.5267,-9.8737;G:0.0585,0.0750,-0.0298;ID3;A:2.6480,-0.5938,-9.7636;G:-0.0511,0.0793,-0.0644;ID3;A:2.4804,-0.3879,-9.6103;G:-0.0431,0.0378,-0.0516;ID3;A:2.4900,-0.5842,-9.7252;G:0.0862,0.1261,-0.0223;ID3;A:2.4804,-0.6033,-9.7348;G:-0.0048,0.0766,-0.0442;ID3;A:2.4565,-0.5507,-9.7348;G:-0.0032,0.0537,-0.0314;ID3;A:2.5666,-0.5363,-9.7683;G:-0.0160,0.0926,-0.0144;ID3;A:2.4612,-0.5555,-9.7540;G:-0.0122,0.0894,-0.0484;ID3;A:2.4421,-0.5602,-9.8497;G:-0.0101,0.0851,-0.0527;ID3;A:2.4612,-0.5555,-9.8210;G:-0.0122,0.0894,-0.0484;ID3;A:2.4086,-0.5459,-9.7923;G:-0.0080,0.0835,-0.0527;ID3;A:2.4421,-0.5315,-9.7971;G:-0.0074,0.0830,-0.0506;ID3;A:2.4660,-0.5459,-9.7013;G:-0.0106,0.0851,-0.0521;ID3;A:2.4852,-0.5076,-9.8067;G:-0.0080,0.0846,-0.0495;ID3;A:2.4421,-0.5650,-9.7588;G:-0.0138,0.0841,-0.0506;ID3;A:2.4565,-0.5890,-9.7300;G:-0.0069,0.0851,-0.0490;ID3;A:2.4660,-0.5602,-9.7731;G:-0.0197,0.0841,-0.0543;ID3;A:2.4804,-0.5986,-9.7731;G:-0.0170,0.0819,-0.0452;ID3;A:2.4421,-0.5459,-9.7827;G:-0.0165,0.0867,-0.0527;ID3;A:2.4517,-0.5459,-9.8162;G:-0.0117,0.0867,-0.0506;ID3;A:2.4708,-0.5698,-9.7540;G:-0.0176,0.0921,-0.0511;ID3;A:2.4421,-0.5890,-9.7971;G:-0.0128,0.0766,-0.0506;ID3;A:2.4612,-0.5746,-9.8019;G:-0.0128,0.0782,-0.0511;ID3;A:2.4660,-0.5315,-9.7971;G:-0.0128,0.0851,-0.0511;ID3;A:2.4660,-0.5555,-9.6774;G:-0.0144,0.0883,-0.0532;ID3;A:2.4948,-0.5698,-9.7588;G:-0.0149,0.0793,-0.0516;ID3;A:2.4900,-0.5650,-9.7636;G:-0.0144,0.0867,-0.0553;ID3;A:2.4277,-0.5363,-9.8162;G:-0.0149,0.0921,-0.0500;ID3;A:2.4852,-0.5698,-9.7252;G:-0.0080,0.0841,-0.0527;ID3;A:2.4612,-0.5602,-9.7588;G:-0.0170,0.0883,-0.0521;ID3;A:2.4373,-0.5171,-9.9072;G:-0.0192,0.0873,-0.0479;End3;Start4;ID4;A:1.5227,-1.0104,-10.3382;G:-0.0335,0.0729,-0.0234;ID4;A:1.6616,-1.2785,-10.5824;G:-0.0580,0.0639,-0.0170;ID4;A:1.5706,-1.1013,-10.3477;G:-0.0506,0.0399,-0.0133;ID4;A:1.6759,-1.1827,-10.5010;G:-0.0607,0.0809,-0.0261;ID4;A:1.6281,-1.0151,-10.5297;G:-0.0607,0.0495,-0.0176;ID4;A:1.6807,-1.1636,-10.3908;G:-0.0495,0.0910,-0.0208;ID4;A:1.5419,-1.1636,-10.4148;G:-0.0596,-0.0016,-0.0080;ID4;A:1.5850,-1.1253,-10.3190;G:-0.0537,0.0059,-0.0144;ID4;A:1.5467,-1.1492,-10.5249;G:-0.0660,0.0984,-0.0255;ID4;A:1.5323,-1.2498,-10.5010;G:-0.0415,-0.0053,-0.0149;ID4;A:1.6185,-1.1684,-10.5441;G:-0.0383,0.0564,-0.0149;ID4;A:1.5467,-1.1588,-10.4196;G:-0.0500,0.0229,-0.0144;ID4;A:1.5562,-1.1779,-10.4531;G:-0.0553,0.0399,-0.0165;ID4;A:1.5706,-1.1301,-10.4866;G:-0.0564,0.0479,-0.0218;ID4;A:1.6137,-1.1349,-10.5249;G:-0.0490,0.0420,-0.0160;ID4;A:1.5754,-1.1827,-10.5105;G:-0.0569,0.0436,-0.0234;ID4;A:1.6185,-1.1827,-10.4914;G:-0.0532,0.0319,-0.0202;ID4;A:1.6137,-1.1732,-10.5393;G:-0.0575,0.0388,-0.0218;ID4;A:1.5945,-1.1875,-10.5680;G:-0.0532,0.0495,-0.0181;ID4;A:1.6089,-1.1540,-10.5010;G:-0.0543,0.0484,-0.0181;ID4;A:1.6185,-1.1636,-10.5058;G:-0.0521,0.0569,-0.0202;ID4;A:1.6424,-1.1396,-10.4627;G:-0.0575,0.0596,-0.0192;ID4;A:1.6089,-1.2115,-10.5201;G:-0.0596,0.0367,-0.0165;ID4;A:1.5802,-1.1732,-10.5632;G:-0.0564,0.0479,-0.0197;ID4;A:1.5706,-1.1923,-10.5153;G:-0.0516,0.0527,-0.0154;ID4;A:1.6041,-1.1827,-10.4531;G:-0.0548,0.0548,-0.0176;ID4;A:1.5945,-1.1971,-10.4914;G:-0.0495,0.0527,-0.0176;ID4;A:1.6233,-1.1971,-10.5297;G:-0.0511,0.0585,-0.0165;ID4;A:1.5897,-1.1492,-10.4483;G:-0.0548,0.0463,-0.0197;ID4;A:1.5610,-1.1827,-10.5058;G:-0.0543,0.0458,-0.0192;ID4;A:1.6089,-1.1492,-10.4052;G:-0.0548,0.0569,-0.0176;ID4;A:1.6281,-1.1971,-10.5153;G:-0.0569,0.0553,-0.0186;ID4;A:1.6089,-1.1732,-10.4914;G:-0.0569,0.0458,-0.0176;ID4;A:1.6328,-1.1827,-10.5728;G:-0.0532,0.0506,-0.0229;ID4;A:1.6233,-1.1732,-10.5680;G:-0.0521,0.0596,-0.0165;ID4;A:1.6424,-1.1827,-10.4914;G:-0.0532,0.0442,-0.0170;ID4;A:1.6041,-1.1732,-10.4962;G:-0.0495,0.0564,-0.0181;ID4;A:1.5371,-1.2402,-10.5153;G:-0.0516,0.0410,-0.0144;ID4;A:1.6328,-1.1779,-10.4627;G:-0.0521,0.0724,-0.0181;ID4;A:1.4844,-1.1923,-10.5489;G:-0.0585,0.0596,-0.0192;End4;"

df = pd.DataFrame(parse_data(raw, EXPECTED_ESP_IDS, TIMESTEPS, NUM_BOARDS), columns=generate_column_names(TIMESTEPS))
