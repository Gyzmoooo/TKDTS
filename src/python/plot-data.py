import os
import re
import traceback
import matplotlib.pyplot as plt


# Quanti sensori (A+G) e assi per sensore ci sono in un blocco?
# A1-A4, G1-G4 -> 8 sensori. x,y,z -> 3 assi.
LINES_PER_TIMESTEP = 8 * 3  # 24 righe per ogni istante temporale completo

# Indici (basati su 0) delle righe relative all'inizio di un timestep (blocco di 24)
# A1x=0, A1y=1, A1z=2
# A2x=3, A2y=4, A2z=5
# A3x=6, A3y=7, A3z=8 <-- Queste ci interessano
# ...
A2_X_OFFSET = 3
A2_Y_OFFSET = 4
A2_Z_OFFSET = 5

# --- Funzione per estrarre il valore numerico dalla parentesi ---
def extract_value_from_line(line):
    """Estrae il valore float da una stringa tipo 'ID(valore)' o '(valore)'."""
    try:
        start_paren = line.find('(')
        end_paren = line.find(')')
        if start_paren != -1 and end_paren != -1 and start_paren < end_paren:
            value_str = line[start_paren + 1 : end_paren]
            # Gestisce sia punti che virgole come separatori decimali
            return float(value_str.replace(',', '.'))
        else:
            print(f"  WARN: Formato parentesi non trovato o non valido nella riga: '{line.strip()}'")
            return None
    except ValueError:
        print(f"  WARN: Impossibile convertire in numero il valore estratto dalla riga: '{line.strip()}'")
        return None
    except Exception as e:
        print(f"  WARN: Errore generico estrazione valore da riga '{line.strip()}': {e}")
        return None

for i in os.listdir("C:\\Users\\giuli\\Desktop\\Code\\TKDTS\\data"):
    FILE_PATH = f"C:\\Users\\giuli\\Desktop\\Code\\TKDTS\\data\\{i}"

    # --- Lettura e Parsing del File ---
    a3_x_values = []
    a3_y_values = []
    a3_z_values = []

    print(f"Tentativo di leggere il file: {FILE_PATH}")

    if not os.path.exists(FILE_PATH):
        print(f"ERRORE: File non trovato: '{FILE_PATH}'")
        exit()

    try:
        with open(FILE_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        num_lines = len(lines)
        print(f"Lette {num_lines} righe dal file.")

        if num_lines == 0:
            print("ERRORE: Il file è vuoto.")
            exit()

        if num_lines % LINES_PER_TIMESTEP != 0:
            print(f"ATTENZIONE: Il numero di righe ({num_lines}) non è un multiplo esatto di {LINES_PER_TIMESTEP}.")
            print("          Potrebbero mancare dati o esserci righe extra. Il parsing potrebbe essere incompleto o errato.")

        # Calcola quanti timestep completi ci sono
        num_timesteps = num_lines // LINES_PER_TIMESTEP
        print(f"Numero di timestep completi stimati: {num_timesteps}")

        processed_timesteps = 0
        for i in range(num_timesteps):
            timestep_start_line_index = i * LINES_PER_TIMESTEP

            # Indici assoluti delle righe per A3 in questo timestep
            idx_a2x = timestep_start_line_index + A2_X_OFFSET
            idx_a2y = timestep_start_line_index + A2_Y_OFFSET
            idx_a2z = timestep_start_line_index + A2_Z_OFFSET

            # Verifica se gli indici sono validi (necessario se il file non è multiplo perfetto)
            if idx_a2z >= num_lines:
                print(f"WARN: Timestep {i+1} incompleto, impossibile leggere dati A2. Interruzione.")
                break

            # Estrai le righe per A2
            line_a3x = lines[idx_a2x].strip()
            line_a3y = lines[idx_a2y].strip()
            line_a3z = lines[idx_a2z].strip()

            # Controllo formale (opzionale ma utile)
            if not line_a3x.startswith("A2("):
                print(f"WARN: Timestep {i+1}: Formato riga A2x inatteso (non inizia con 'A2('): '{line_a3x}'. Salto timestep.")
                continue
            if not line_a3y.startswith("("):
                print(f"WARN: Timestep {i+1}: Formato riga A2y inatteso (non inizia con '('): '{line_a3y}'. Salto timestep.")
                continue
            if not line_a3z.startswith("("):
                print(f"WARN: Timestep {i+1}: Formato riga A2z inatteso (non inizia con '('): '{line_a3z}'. Salto timestep.")
                continue

            # Estrai i valori numerici
            val_a3x = extract_value_from_line(line_a3x)
            val_a3y = extract_value_from_line(line_a3y)
            val_a3z = extract_value_from_line(line_a3z)

            # Aggiungi i valori alle liste solo se tutti e tre sono stati estratti correttamente
            if val_a3x is not None and val_a3y is not None and val_a3z is not None:
                a3_x_values.append(val_a3x)
                a3_y_values.append(val_a3y)
                a3_z_values.append(val_a3z)
                processed_timesteps += 1
            else:
                print(f"WARN: Timestep {i+1}: Errore nell'estrazione di uno o più valori per A2. Salto timestep.")


    except FileNotFoundError:
        print(f"ERRORE CRITICO: File non trovato all'indirizzo specificato: '{FILE_PATH}'")
        exit()
    except Exception as e:
        print(f"ERRORE CRITICO durante la lettura o il parsing del file: {e}")
        traceback.print_exc()
        exit()

    # --- Plotting dei Dati Estratti ---
    if not a3_x_values: # Controlla se abbiamo effettivamente estratto dati
        print("\nNessun dato valido per A2 è stato estratto. Impossibile creare il grafico.")
    else:
        num_samples = len(a3_x_values)
        print(f"\nDati A2 estratti con successo per {num_samples} campioni.")
        print("Creazione del grafico...")

        # Crea l'asse X (numero del campione/timestep)
        sample_indices = list(range(num_samples))

        # Crea il grafico
        plt.figure(figsize=(12, 6)) # Imposta dimensioni figura (larghezza, altezza in pollici)

        plt.plot(sample_indices, a3_x_values, label='A2 - Asse X', marker='.', linestyle='-', markersize=4)
        plt.plot(sample_indices, a3_y_values, label='A2 - Asse Y', marker='.', linestyle='-', markersize=4)
        plt.plot(sample_indices, a3_z_values, label='A2 - Asse Z', marker='.', linestyle='-', markersize=4)

        # Aggiungi etichette e titolo
        plt.xlabel("Numero Campione (Quantità di dati)")
        plt.ylabel("Valore Accelerometro")
        plt.title(f"Dati Accelerometro A3 ({num_samples} campioni)\nFile: {os.path.basename(FILE_PATH)}") # Aggiunge nome file al titolo

        # Aggiungi legenda e griglia
        plt.legend() # Mostra le etichette definite nel plt.plot
        plt.grid(True) # Aggiunge una griglia per facilitare la lettura

        # Mostra il grafico
        plt.tight_layout() # Aggiusta layout per evitare sovrapposizioni
        plt.show()

        print("Grafico visualizzato.")