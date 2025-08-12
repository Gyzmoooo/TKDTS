import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

TIMESTEPS = 20
NUM_BOARDS = 4
SENSORS = ['A', 'G']
AXES = ['x', 'y', 'z']

df = pd.read_csv("kicks_classification.csv")
#df = df.loc[0:100]

def classifica_con_soglia(registrazione_smv, soglia=7.0):
    """
    Classifica una registrazione come 'Calcio' o 'Fermo' basandosi su una semplice soglia.

    :param registrazione_smv: Una lista o array di 8 valori SMV.
    :param soglia: Il valore di SMV oltre il quale si considera un calcio.
    :return: La stringa 'Calcio' o 'Fermo'.
    """
    # Controlla se QUALSIASI valore nella lista supera la soglia
    if np.mean(registrazione_smv) > soglia:
        return "Calcio"
    
    # Se il ciclo finisce senza aver trovato valori sopra la soglia
    return "Fermo"

def compute_svm(df):
    out_list = []
    in_array = df.to_numpy()
    
    for row in in_array:
        temp_list = []
        for column in range(1, len(row), 3):
            smv = np.sqrt(row[column]**2 + row[column+1]**2 + row[column+2]**2)
            temp_list.append(smv)
        out_list.append(temp_list)
    
    return np.array(out_list)

def generate_column_names():
    column_names = []
    for i in range(TIMESTEPS):
        suffix = f"_{i + 1}"
        for board_id in range(1, NUM_BOARDS + 1):
            for sensor_type in SENSORS:
                for axis in AXES:
                    col_name = f"{sensor_type}{board_id}{axis}{suffix}"
                    column_names.append(col_name)
    return column_names

lista2 = []
for i in range(len(compute_svm(df)[0:50])):
    array = compute_svm(df)[i]
    counter = 0
    counter2 = 0
    lista = []
    for j in range(len(array)):
        if counter % 8 == 0:
            start = counter2 * 8
            end = (counter2 + 1) * 8
            lista = array[start:end]
            counter2 += 1 
            lista2.append(lista)
            #print(f"Questa è la registrazione numero {(counter // 8) + 1}")
            #print(lista)
            #print(classifica_con_soglia(lista, soglia=14.0))
        counter += 1

#print(np.array(lista2))

lista3 = np.array([])
for i in range(len(np.array(lista2))):
    lista3 = np.append(lista3, classifica_con_soglia(lista2[i]))

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


#print(crea_gruppi_(lista3))
new_df = pd.DataFrame(generate_column_names())
lista = []
for i in range(len(crea_gruppi_(lista3))):
    start = crea_gruppi_(lista3)[i][0][1]
    end = crea_gruppi_(lista3)[i][-1][1]
    for j in range(int(end) - int(start)):
        lista.append(crea_gruppi_(lista3)[i][j][1])
    print(lista)
    #new_df = pd.concat([new_df, pd.DataFrame(lista)], ignore_index=True)

#print(new_df)
#print(df)