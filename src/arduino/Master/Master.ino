#include <WiFi.h>
#include <Wire.h>
#include <WebServer.h>
#include <esp_now.h>
#include <esp_wifi.h> 

#include <FS.h>
#include <LittleFS.h>

//CONFIGURAZIONE MASTER
const int MASTER_ESP_ID = 1;
const char *ssid = "Taekwondo-ts";
const char *password = "123456789";
const int WIFI_CHANNEL = 1; // Canale WiFi per AP e ESP-NOW (devono coincidere)

// MAC Address 
uint8_t clientMacs[][6] = {
  {0xDC, 0x1E, 0xD5, 0x1D, 0x27, 0xD8}, // MAC ESP ID 2 DC:1E:D5:1D:27:D8
  {0xDC, 0x1E, 0xD5, 0x1E, 0x6F, 0xEC}, // MAC ESP ID 3 dc:1e:d5:1e:6f:ec
  {0xE4, 0xB3, 0x23, 0xD4, 0xEC, 0x04}  // MAC ESP ID 4 e4:b3:23:d4:ec:04
};
const int numClients = sizeof(clientMacs) / sizeof(clientMacs[0]);

// Struttura per i messaggi ESP-NOW
typedef struct struct_message {
  char command[10]; // "START" o "STOP"
} struct_message;
struct_message commandMessage;

// Server & Sensore 
WebServer server(80);
const int MPU = 0x68;
const int buttonPin = 4; // Pin pulsante sul Master
const int sensorVccPin = 5;
const int sensorGndPin = 6;
const int i2cSdaPin = 8;
const int i2cSclPin = 7;
const int ledGround = 1;
const int ledPin = 0;

// Frequenza Campionamento 
const int SAMPLES_PER_SECOND = 20;
const unsigned long sampleIntervalMillis = 1000 / SAMPLES_PER_SECOND;
unsigned long lastSampleTime = 0;

const int NUM_RETRIES = 3; // Quante volte inviare il comando a *ciascun* client
const int RETRY_DELAY_MS = 20; // Pausa tra i tentativi allo *stesso* client (ms)
const int CLIENT_DELAY_MS = 10; // Pausa *tra client diversi* (ms)

bool collectingDataMaster = false;

void appendDataToFile(const String& path, const String& data) {
  File file = LittleFS.open(path, "a");
  if (!file) {
    Serial.println("Errore apertura file per scrittura");
    return;
  }
  if (!file.print(data)) {
    Serial.println("Errore di scrittura su file");
  }
  file.close();
}

void clearDataFiles(){
  Serial.println("Cancellazione dei file di dati precedenti... ");
  for (int i = 1; i <= (numClients + 1); i++) {
    String path = "/data_esp" + String(i) + ".txt";
    if (LittleFS.exists(path)) {
      LittleFS.remove(path);
    }
  }
}


// Callback invio ESP-NOW
void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  Serial.print("\r\nStato Invio Ultimo Pacchetto a ");
  char macStr[18];
  snprintf(macStr, sizeof(macStr), "%02x:%02x:%02x:%02x:%02x:%02x",
           mac_addr[0], mac_addr[1], mac_addr[2], mac_addr[3], mac_addr[4], mac_addr[5]);
  Serial.print(macStr);
  Serial.print(": ");
  Serial.println(status == ESP_NOW_SEND_SUCCESS ? "Successo" : "Fallito");
}

// Funzione per raccogliere dati 
void collectSensorDataMaster() {

  int16_t AcX, AcY, AcZ, GyX, GyY, GyZ;
  Wire.beginTransmission(MPU); Wire.write(0x3B); Wire.endTransmission(false);
  Wire.requestFrom(MPU, 14, true);
  AcX=Wire.read()<<8|Wire.read(); AcY=Wire.read()<<8|Wire.read(); AcZ=Wire.read()<<8|Wire.read();
  Wire.read(); Wire.read(); // Skip Temp
  GyX=Wire.read()<<8|Wire.read(); GyY=Wire.read()<<8|Wire.read(); GyZ=Wire.read()<<8|Wire.read();

  float accX_mps2 = (AcX/2048.0)*9.81; float accY_mps2 = (AcY/2048.0)*9.81; float accZ_mps2 = (AcZ/2048.0)*9.81;
  float gyroX_rads = (GyX/32.8)*(PI/180.0); float gyroY_rads = (GyY/32.8)*(PI/180.0); float gyroZ_rads = (GyZ/32.8)*(PI/180.0);

  String data = "ID" + String(MASTER_ESP_ID) + ";"; // Aggiungi identificatore all'inizio di ogni lettura
  data += "A:" + String(accX_mps2, 4) + "," + String(accY_mps2, 4) + "," + String(accZ_mps2, 4) + ";";
  data += "G:" + String(gyroX_rads, 4) + "," + String(gyroY_rads, 4) + "," + String(gyroZ_rads, 4) + ";";
  
  appendDataToFile("/data_esp" + String(MASTER_ESP_ID) + ".txt", data);
}

// Funzione per gestire la pressione del pulsante (CON RETRY PER ESP-NOW)
void handleButtonPress() {
  collectingDataMaster = !collectingDataMaster; // Inverti stato locale

  if (collectingDataMaster) {
    // INIZIO RACCOLTA 
    Serial.println("--------------------------------------------------");
    Serial.println("Pulsante Premuto: INIZIO Raccolta Dati Globale.");
    clearDataFiles();

    String startMarker = "Start" + String(MASTER_ESP_ID) + ";";
    appendDataToFile("/data_esp" + String(MASTER_ESP_ID) + ".txt", startMarker);
    lastSampleTime = millis(); // Resetta il tempo per il campionamento del Master

    // Invia comando START via ESP-NOW a tutti i client con retry
    Serial.println("Invio comando START ai client...");
    strcpy(commandMessage.command, "START");
    for (int i = 0; i < numClients; ++i) {
      Serial.printf("  Tentativi per Client MAC: %02X:%02X:%02X:%02X:%02X:%02X\n",
                     clientMacs[i][0], clientMacs[i][1], clientMacs[i][2], clientMacs[i][3], clientMacs[i][4], clientMacs[i][5]);
      for (int retry = 0; retry < NUM_RETRIES; ++retry) {
        esp_err_t result = esp_now_send(clientMacs[i], (uint8_t *) &commandMessage, sizeof(commandMessage));
        Serial.printf("    Tentativo %d: %s\n", retry + 1, (result == ESP_OK) ? "OK (Inviato)" : "FALLITO");
        if (result == ESP_OK) {
          
        }
        delay(RETRY_DELAY_MS); 
      }
      delay(CLIENT_DELAY_MS); // Piccola pausa prima di passare al client successivo
    }
    Serial.println("Comando START inviato a tutti i client.");
    Serial.println("--------------------------------------------------");

  } else {
    // --- FINE RACCOLTA ---
    Serial.println("--------------------------------------------------");
    Serial.println("Pulsante Premuto: FINE Raccolta Dati Globale.");

    String endMarker = "End" + String(MASTER_ESP_ID) + ";";
    appendDataToFile("/data_esp" + String(MASTER_ESP_ID) + ".txt", endMarker);

    // Invia comando STOP via ESP-NOW a tutti i client con retry
    Serial.println("Invio comando STOP ai client...");
    strcpy(commandMessage.command, "STOP");
     for (int i = 0; i < numClients; ++i) {
       Serial.printf("  Tentativi per Client MAC: %02X:%02X:%02X:%02X:%02X:%02X\n",
                      clientMacs[i][0], clientMacs[i][1], clientMacs[i][2], clientMacs[i][3], clientMacs[i][4], clientMacs[i][5]);
      for (int retry = 0; retry < NUM_RETRIES; ++retry) {
        esp_err_t result = esp_now_send(clientMacs[i], (uint8_t *) &commandMessage, sizeof(commandMessage));
        Serial.printf("    Tentativo %d: %s\n", retry + 1, (result == ESP_OK) ? "OK (Inviato)" : "FALLITO");
         if (result == ESP_OK) {
            // break; // Opzionale: esci al primo successo
         }
        delay(RETRY_DELAY_MS);
      }
      delay(CLIENT_DELAY_MS);
    }
    Serial.println("Comando STOP inviato a tutti i client.");
    Serial.println("--------------------------------------------------");
  }
}

void handleRoot() {
  if (collectingDataMaster) {
    server.send(200, "text/plain", "aspettaciola");
  } else {
    // Inizia lo streaming della risposta senza definire una lunghezza totale
    server.setContentLength(CONTENT_LENGTH_UNKNOWN);
    server.send(200, "text/plain", ""); // Invia solo gli header HTTP per aprire la connessione

    // Itera su tutti i file di dati (Master + Slaves)
    for (int i = 1; i <= (numClients + 1); i++) {
      String path = "/data_esp" + String(i) + ".txt";
      
      if (LittleFS.exists(path)) {
        File file = LittleFS.open(path, "r");
        if (file) {
          // Invia il contenuto del file in piccoli blocchi (linea per linea)
          // senza mai caricarlo tutto in memoria.
          while (file.available()) {
            server.sendContent(file.readStringUntil('\n') + "\n");
          }
          file.close();
        }
      }
    }
    
    server.sendContent(""); // Invia un blocco vuoto per segnalare la fine dello stream
  }
}

// Gestore per la sottomissione dati dai client
void handleSubmit() {
  /*
  digitalWrite(ledPin, HIGH);
  delay(1000);
  digitalWrite(ledPin, LOW);*/
  if (server.method() != HTTP_POST) {
    server.send(405, "text/plain", "Method Not Allowed");
    return;
  }

  String clientIdStr = server.arg("id"); // Legge l'ID dall'URL (?id=X)
  String clientData = server.arg("plain"); // Legge i dati dal corpo della POST

  if (clientIdStr.length() == 0 || clientData.length() == 0) {
    server.send(400, "text/plain", "Bad Request: Missing id or data");
    return;
  }

  String path = "/data_esp" + clientIdStr + ".txt";
  appendDataToFile(path, clientData);

  server.send(200, "text/plain", "OK"); // Conferma ricezione al client
}


// Gestore per eliminare tutti i dati
void handleDelete() {
  Serial.println("Richiesta /delete ricevuta. Cancello tutti i buffer.");
  clearDataFiles();
  server.send(200, "text/plain", "OK");
}

void setup() {
  Serial.begin(115200);
  Serial.println("\nAvvio Master ESP (ID 1)...");
  
  // Inizializzazione LittleFS
  if (!LittleFS.begin(true)) { // Il 'true' formatta se non montabile
    Serial.println("Errore montaggio LittleFS! Blocco.");
    while (1);
  }
  Serial.println("File System LittleFS montato.");

  // Configurazione Pin 
  pinMode(buttonPin, INPUT_PULLUP);
  pinMode(sensorVccPin, OUTPUT); digitalWrite(sensorVccPin, HIGH);
  pinMode(sensorGndPin, OUTPUT); digitalWrite(sensorGndPin, LOW);
  pinMode(ledGround, OUTPUT); digitalWrite(ledGround, LOW);
  pinMode(ledPin, OUTPUT); digitalWrite(ledPin, LOW);
  Serial.println("Pin configurati.");

  // Inizializzazione I2C MPU6050
  Wire.begin(i2cSdaPin, i2cSclPin); delay(100);
  Serial.println("Inizializzazione MPU6050...");
  Wire.beginTransmission(MPU); Wire.write(0x6B); Wire.write(0);
  if (Wire.endTransmission(true) != 0) { Serial.println("Errore I2C MPU6050!"); while(1); }
  Wire.beginTransmission(MPU); Wire.write(0x1C); Wire.write(0x18); Wire.endTransmission(true); // Accel ±16g
  Wire.beginTransmission(MPU); Wire.write(0x1B); Wire.write(0x10); Wire.endTransmission(true); // Gyro ±1000 °/s
  Serial.println("MPU6050 Master inizializzato.");

  //Configurazione WiFi
  Serial.println("Configurazione WiFi (Proviamo prima AP)...");
  WiFi.mode(WIFI_AP); // Inizia solo come AP
  // Crea l'AP specificando il canale qui, invece di usare esp_wifi_set_channel prima
  if (WiFi.softAP(ssid, password, WIFI_CHANNEL, 0, numClients + 1)) { // Canale, Hidden=0, MaxConnessioni
       Serial.println("Access Point avviato!");
       Serial.print("SSID: "); Serial.println(ssid);
       Serial.print("IP Address: http://"); Serial.println(WiFi.softAPIP());
       Serial.print("Master MAC: "); Serial.println(WiFi.macAddress());
       Serial.print("Canale AP: "); Serial.println(WiFi.channel());
  } else {
      Serial.println("Errore avvio SoftAP!");
      ESP.restart(); // Riavvia se l'AP non parte
  }

  //inizializione ESP-NOW DOPO l'AP ---
  Serial.println("Inizializzazione ESP-NOW...");

  if (esp_now_init() != ESP_OK) {
    Serial.println("Errore inizializzazione ESP-NOW");
  } else {
      esp_now_register_send_cb(OnDataSent);
      Serial.println("ESP-NOW inizializzato.");

      // Registra i peer client
      Serial.println("Registrazione peers ESP-NOW...");
      for (int i = 0; i < numClients; ++i) {
         esp_now_peer_info_t peerInfo = {};
         memcpy(peerInfo.peer_addr, clientMacs[i], 6);
         peerInfo.channel = WIFI_CHANNEL; // Usa lo stesso canale dell'AP
         peerInfo.encrypt = false;
         peerInfo.ifidx = WIFI_IF_AP; // Specifica l'interfaccia AP per l'invio

         if (esp_now_add_peer(&peerInfo) != ESP_OK){
           Serial.printf("Fallito aggiunta peer %d\n", i+2);
           // return; // O gestisci l'errore
         } else {
           Serial.printf("Peer Client %d aggiunto\n", i+2);
         }
      }
  }

  //Configurazione Web Server
  server.on("/", HTTP_GET, handleRoot);
  server.on("/submit", HTTP_POST, handleSubmit);
  server.on("/delete", HTTP_GET, handleDelete);
  server.begin();
  Serial.println("Server Web avviato!");
  Serial.print("Frequenza di campionamento: "); Serial.print(SAMPLES_PER_SECOND); Serial.println(" Hz");
}


void loop() {
  server.handleClient();

  // Controllo Pulsante Master
  static bool lastButtonState = HIGH;
  bool currentButtonState = digitalRead(buttonPin);
  if (lastButtonState == HIGH && currentButtonState == LOW) {
    delay(50); // Debounce
    if (digitalRead(buttonPin) == LOW) { handleButtonPress(); }
  }
  lastButtonState = currentButtonState;

  // Raccoglie i dati del Master alla frequenza specificata
  if (collectingDataMaster) {
    unsigned long currentTime = millis();
    if (currentTime - lastSampleTime >= sampleIntervalMillis) {
      lastSampleTime = currentTime;
      collectSensorDataMaster();
    }
  } else {
     delay(10); // Mantieni un piccolo delay per stabilità
  }
}