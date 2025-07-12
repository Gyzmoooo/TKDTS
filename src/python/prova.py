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

print(N_DATA)