#include "FS.h"
#include "LittleFS.h"

void setup() {
  Serial.begin(115200);

  if(!LittleFS.begin(true)){
    Serial.println("An Error has occurred while mounting LittleFS");
    return;
  }

  // Scrivere un file
  File file = LittleFS.open("/welcome.txt", FILE_WRITE);
  if(!file){
    Serial.println("There was an error opening the file for writing");
    return;
  }
  if(file.print("Welcome to LittleFS!")){
    Serial.println("File was written");
  } else {
    Serial.println("File write failed");
  }
  file.close();

  // Leggere un file
  file = LittleFS.open("/welcome.txt", FILE_READ);
  if(!file){
    Serial.println("Failed to open file for reading");
    return;
  }
  Serial.println("File content:");
  while(file.available()){
    Serial.write(file.read());
  }
  file.close();

  // Elencare i file
  File root = LittleFS.open("/");
  File fileInDir = root.openNextFile();
  while(fileInDir){
      Serial.print("FILE: ");
      Serial.println(fileInDir.name());
      fileInDir = root.openNextFile();
  }
}

void loop() {
}