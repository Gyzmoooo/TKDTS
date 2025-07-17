#include <WiFi.h>
#include <Wire.h>      
#include <esp_now.h>
#include <HTTPClient.h> 
#include <esp_wifi.h>   



const int CLIENT_ESP_ID = 4; // CAMBIA A 2, 3, o 4 
const char *ssid_master = "Taekwondo-ts"; 
const char *password_master = "123456789";   
const int WIFI_CHANNEL = 1;                   // Deve corrispondere al canale del Master



uint8_t masterMac[] = {0xE4, 0xB3, 0x23, 0xD3, 0xA4, 0xD4}; //e4:b3:23:d3:a4:d4

// URL del Master per inviare dati (usa l'IP fisso dell'AP del Master: 192.168.4.1)
String masterUrl = "http://192.168.4.1/submit?id=" + String(CLIENT_ESP_ID);

// Struttura per i messaggi ESP-NOW 
typedef struct struct_message {
  char command[10]; // "START" o "STOP"
} struct_message;
struct_message receivedMessage;

// Sensore MPU6050
const int MPU = 0x68; 
const int sensorVccPin = 5; // Pin GPIO usato per alimentare VCC del sensore
const int sensorGndPin = 6; // Pin GPIO usato per collegare GND del sensore
const int i2cSdaPin = 8;    // Pin SDA per I2C
const int i2cSclPin = 7;    // Pin SCL per I2C

// Frequenza Campionamento 
const int SAMPLES_PER_SECOND = 20; 
const unsigned long sampleIntervalMillis = 1000 / SAMPLES_PER_SECOND;
unsigned long lastSampleTime = 0;

// Accumulo Dati Locale & Stato
String accumulatedDataLocal = "";
bool collectingDataClient = false; // Stato raccolta dati locale
bool sendDataPending = false;   // Flag per indicare che i dati devono essere inviati

// Callback ricezione ESP-NOW
void OnDataRecv(const esp_now_recv_info * info, const uint8_t *incomingData, int len) {
  if (len == sizeof(struct_message)) {
    memcpy(&receivedMessage, incomingData, sizeof(struct_message));
    Serial.print("Comando ricevuto via ESP-NOW: ");
    Serial.println(receivedMessage.command);

    if (strcmp(receivedMessage.command, "START") == 0) {
      if (!collectingDataClient) {
         Serial.println("Ricevuto START: Inizio raccolta dati MPU6050.");
         collectingDataClient = true;
         accumulatedDataLocal = ""; // Pulisce buffer da eventuali dati precedenti
         accumulatedDataLocal += "Start" + String(CLIENT_ESP_ID) + ";"; // Marcatore inizio
         lastSampleTime = millis(); // Inizia il timer per il primo campionamento
      } else {
         Serial.println("Ricevuto START, ma la raccolta è già attiva. Ignorato.");
      }
    } else if (strcmp(receivedMessage.command, "STOP") == 0) {
      if (collectingDataClient) {
        Serial.println("Ricevuto STOP: Fine raccolta dati MPU6050.");
        collectingDataClient = false;
        accumulatedDataLocal += "End" + String(CLIENT_ESP_ID) + ";"; // Marcatore fine
        sendDataPending = true; // Imposta flag per inviare i dati nel loop principale
      } else {
         Serial.println("Ricevuto STOP, ma la raccolta era già ferma. Ignorato.");
      }
    }
  } else {
     Serial.printf("Ricevuto pacchetto ESP-NOW di lunghezza errata (%d byte invece di %d).\n", len, sizeof(struct_message));
  }
}

// Funzione per raccogliere dati dal sensore MPU6050 
void collectSensorDataClient() {
  int16_t AcX, AcY, AcZ, GyX, GyY, GyZ; // Variabili per dati grezzi

  Wire.beginTransmission(MPU);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) { 
      Serial.println("Errore I2C endTransmission pre-lettura MPU6050");
      return; // Esce dalla funzione se c'è errore
  }

  int bytesRead = Wire.requestFrom(MPU, 14, true); // true = rilascia il bus dopo lettura
  if (bytesRead == 14) {
      // dati grezzi
      AcX = Wire.read() << 8 | Wire.read(); // Accel X (High byte | Low byte)
      AcY = Wire.read() << 8 | Wire.read(); // Accel Y
      AcZ = Wire.read() << 8 | Wire.read(); // Accel Z
      Wire.read() << 8 | Wire.read();       // Salta i due byte della Temperatura (Tmp)
      GyX = Wire.read() << 8 | Wire.read(); // Gyro X
      GyY = Wire.read() << 8 | Wire.read(); // Gyro Y
      GyZ = Wire.read() << 8 | Wire.read(); // Gyro Z

      // Conversione in unità fisiche
      // Acc: ±16g, Gyro: ±1000°/s
      const float ACCEL_SENSITIVITY = 2048.0; 
      const float GYRO_SENSITIVITY = 32.8;    
      const float G_ACCEL = 9.80665;          

      float accX_mps2 = (AcX / ACCEL_SENSITIVITY) * G_ACCEL;
      float accY_mps2 = (AcY / ACCEL_SENSITIVITY) * G_ACCEL;
      float accZ_mps2 = (AcZ / ACCEL_SENSITIVITY) * G_ACCEL;
      float gyroX_rads = (GyX / GYRO_SENSITIVITY) * (PI / 180.0); // Converti da °/s a rad/s
      float gyroY_rads = (GyY / GYRO_SENSITIVITY) * (PI / 180.0);
      float gyroZ_rads = (GyZ / GYRO_SENSITIVITY) * (PI / 180.0);


      String data = "ID" + String(CLIENT_ESP_ID) + ";";
      data += "A:" + String(accX_mps2, 4) + "," + String(accY_mps2, 4) + "," + String(accZ_mps2, 4) + ";"; // 4 cifre decimali
      data += "G:" + String(gyroX_rads, 4) + "," + String(gyroY_rads, 4) + "," + String(gyroZ_rads, 4) + ";";


      // USA IL BUFFER LOCALE DEL CLIENT
      accumulatedDataLocal += data;

      /*/ Controllo opzionale dimensione buffer (da adattare se necessario)
       if (accumulatedDataLocal.length() > 10000) { // Esempio limite
          Serial.println("WARN: Buffer locale quasi pieno, invio forzato.");
          collectingDataClient = false; // Ferma raccolta
          accumulatedDataLocal += "End" + String(CLIENT_ESP_ID) + "_Forced;"; // Marcatore speciale
          sendDataPending = true;
       }*/

  } else {
      Serial.printf("Errore I2C: Letti %d byte invece di 14 da MPU6050\n", bytesRead);
      
  }
}


// Funzione per inviare dati al Master via HTTP POST
void sendDataToMaster() {
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("Invio dati al Master via HTTP POST a: "); Serial.println(masterUrl);
    // Debug: Stampa solo i primi N caratteri per evitare flood se i dati sono tanti
    // Serial.print("Dati (inizio): "); Serial.println(accumulatedDataLocal.substring(0, 100));

    HTTPClient http;
    http.begin(masterUrl); // Specifica URL completo (include già ?id=X)
    http.addHeader("Content-Type", "text/plain"); // Il Master si aspetta plain text nel corpo
    // Potresti voler aumentare il timeout per l'invio di grandi quantità di dati
    // http.setTimeout(15000); // Esempio: 15 secondi

    int httpCode = http.POST(accumulatedDataLocal); // Invia i dati accumulati nel corpo della richiesta

    if (httpCode > 0) { // Controlla se c'è stata una risposta HTTP valida
      String payload = http.getString(); // Ottieni la risposta dal server (dovrebbe essere "OK")
      Serial.printf("[HTTP] Codice risposta POST: %d\n", httpCode);
      Serial.printf("[HTTP] Risposta dal Master: %s\n", payload.c_str());

      if (httpCode == HTTP_CODE_OK) { // HTTP 200 OK (il Master ha ricevuto e processato correttamente)
        Serial.println("Dati inviati con successo al Master.");
        accumulatedDataLocal = ""; // *** Pulisce buffer locale SOLO dopo invio confermato con successo ***
        sendDataPending = false;   // Resetta il flag dopo l'invio riuscito
      } else {
         Serial.printf("Errore dal server Master durante l'invio (Codice HTTP: %d). I dati NON sono stati cancellati localmente.\n", httpCode);
         // Non puliamo il buffer, i dati verranno ritentati al prossimo comando STOP.
         // Resettiamo comunque il flag per evitare loop di tentativi falliti.
         sendDataPending = false;
      }
    } else {
      // Errore a livello di connessione HTTP (es. server non raggiungibile, timeout, errore DNS)
      Serial.printf("[HTTP] Invio POST fallito, errore HTTPClient: %s\n", http.errorToString(httpCode).c_str());
      // I dati non vengono cancellati, verranno ritentati al prossimo STOP.
      sendDataPending = false; // Resetta il flag per evitare loop di tentativi falliti
    }
    http.end(); // Rilascia le risorse usate da HTTPClient
  } else {
    Serial.println("WiFi non connesso. Impossibile inviare dati ora. Riproverò al prossimo STOP.");
    // I dati rimangono in accumulatedDataLocal.
    // Resetta il flag per evitare tentativi continui senza WiFi.
    sendDataPending = false;
  }
}


void setup() {
  Serial.begin(115200);

  // --- Prima: Configurazione Pin e MPU6050 (come Master) ---
  Serial.println("Configurazione pin sensore...");
  pinMode(sensorVccPin, OUTPUT); digitalWrite(sensorVccPin, HIGH);
  pinMode(sensorGndPin, OUTPUT); digitalWrite(sensorGndPin, LOW);
  delay(100); // Attesa stabilizzazione alimentazione

  Serial.printf("\n--- Avvio Client ESP-NOW con Sensore (ID %d) ---\n", CLIENT_ESP_ID);
  Serial.println("!!! VERIFICA CHE I PIN SENSOR/I2C SIANO CORRETTI PER LA TUA SCHEDA !!!");
  Serial.printf("  SDA: %d, SCL: %d, VCC_PIN: %d, GND_PIN: %d\n", i2cSdaPin, i2cSclPin, sensorVccPin, sensorGndPin);

  Serial.println("Inizializzazione I2C e MPU6050...");
  Wire.begin(i2cSdaPin, i2cSclPin);
  delay(100);

  Wire.beginTransmission(MPU);
  Wire.write(0x6B); // Registro PWR_MGMT_1
  Wire.write(0);    // Sveglia sensore
  if (Wire.endTransmission(true) != 0) {
      Serial.println("ERRORE FATALE: MPU6050 non trovato all'indirizzo I2C! Blocco.");
      // Aggiungi qui il lampeggio LED veloce se usi il debug LED
      while(1) { delay(100); } // Blocco
  } else {
      Serial.println("MPU6050 trovato e svegliato.");
      // Aggiungi qui il lampeggio LED di successo MPU se usi il debug LED
  }

  // Imposta sensibilità (come Master e come Slave originale)
  Wire.beginTransmission(MPU); Wire.write(0x1C); Wire.write(0x18); Wire.endTransmission(true);
  Wire.beginTransmission(MPU); Wire.write(0x1B); Wire.write(0x10); Wire.endTransmission(true);
  Serial.println("MPU6050 configurato (Accel: +/-16g, Gyro: +/-1000 deg/s).");

  // *** AGGIUNGI UN SEGNO CHE L'MPU È OK PRIMA DI WIFI ***
  Serial.println("--- MPU OK! Procedo con WiFi ed ESP-NOW ---");
  // Se usi il LED, fai un lampeggio specifico qui.
  delay(500); // Pausa opzionale

  // --- ORA: Inizializzazione WiFi ed ESP-NOW (come Slave originale, ma alla fine) ---
  Serial.println("Configurazione WiFi (Modalità Station)...");
  WiFi.mode(WIFI_STA);

  esp_err_t channel_result = esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);
  if (channel_result != ESP_OK) {
    Serial.printf("Errore critico impostazione canale WiFi %d: %s. Riavvio...\n", WIFI_CHANNEL, esp_err_to_name(channel_result));
    delay(1000);
    ESP.restart();
  } else {
     Serial.printf("Canale WiFi impostato su %d.\n", WIFI_CHANNEL);
  }

  WiFi.begin(ssid_master, password_master);
  Serial.print("Connessione all'AP "); Serial.print(ssid_master);
  int connect_timeout = 30; // Timeout 15 secondi
  while (WiFi.status() != WL_CONNECTED && connect_timeout > 0) {
    delay(500);
    Serial.print(".");
    connect_timeout--;
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("WiFi Connesso all'AP del Master!");
    Serial.print("  Indirizzo IP ottenuto: "); Serial.println(WiFi.localIP());
    Serial.print("  Client MAC Address: "); Serial.println(WiFi.macAddress());
    Serial.print("  Canale WiFi attuale: "); Serial.println(WiFi.channel());
  } else {
    Serial.println("!!! Connessione WiFi all'AP del Master fallita! Riavvio...");
    delay(1000);
    ESP.restart();
  }

  Serial.println("Inizializzazione ESP-NOW...");
  if (esp_now_init() != ESP_OK) {
    Serial.println("Errore critico inizializzazione ESP-NOW. Riavvio...");
    delay(1000);
    ESP.restart();
  }

  if (esp_now_register_recv_cb(OnDataRecv) != ESP_OK) {
      Serial.println("Errore critico registrazione callback ricezione ESP-NOW. Riavvio...");
      delay(1000);
      ESP.restart();
  } else {
      Serial.println("Callback ricezione ESP-NOW registrata.");
  }

  Serial.println("Registrazione peer Master ESP-NOW...");
  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, masterMac, 6);
  peerInfo.channel = WIFI_CHANNEL;
  peerInfo.encrypt = false;
  peerInfo.ifidx = WIFI_IF_STA;

  if (esp_now_add_peer(&peerInfo) != ESP_OK){
    Serial.println("Fallito aggiunta peer Master. ESP-NOW potrebbe non funzionare correttamente.");
  } else {
     Serial.println("Peer Master aggiunto con successo.");
  }

   Serial.println("--- Client pronto a ricevere comandi START/STOP ---");
   Serial.print("Frequenza di campionamento MPU6050: "); Serial.print(SAMPLES_PER_SECOND); Serial.println(" Hz");
   // Se usi il LED, fai un lampeggio lento qui per indicare setup completo.
}


void loop() {
  // --- Raccolta Dati ---
  // Controlla se dobbiamo raccogliere dati e se è passato l'intervallo di campionamento
  if (collectingDataClient) {
    unsigned long currentTime = millis();
    // Esegue la lettura solo se è passato abbastanza tempo dall'ultima
    if (currentTime - lastSampleTime >= sampleIntervalMillis) {
      lastSampleTime = currentTime; // Aggiorna l'ultimo tempo di campionamento
      collectSensorDataClient();    // *** CHIAMA LA FUNZIONE PER LEGGERE IL SENSORE ***
    }
  }

  // --- Invio Dati ---
  // Controlla se c'è una richiesta di invio pendente (impostata da OnDataRecv su STOP)
  if (sendDataPending) {
     sendDataToMaster(); // Chiama la funzione che tenta l'invio via HTTP POST
     // sendDataPending viene resettato all'interno di sendDataToMaster()
  }

  // Piccolo delay per dare respiro al processore
  delay(1); // Breve delay (1ms)

} 