#include "FS.h"
#include "LittleFS.h"

void setup() {
  Serial.begin(115200);

  if(!LittleFS.begin(true)){
    Serial.println("An Error has occurred while mounting LittleFS");
    return;
  }
  
  const char* filename_write = "/welcome.txt";
  char data_to_write[] = "Start1;ID1;A:-0.3736,9.5897,-1.0251;G:0.0192,0.1389,-0.0090;ID1;A:-0.1629,9..1150;ID1;A:-0.1341,9.6759,-0.9053;G:0.0357,0.0282,0.0011;ID1;A:-0.1198,9.6423,-0.8909;G:0.0346,0.0271,-0.0011;ID1;A:-0.1054,9.6423,-0.9197;G:0.;ID1;A:-0.1868,9.7381,-0.9580;G:-0.0649,-0.5241,0.0399;End1;";
  Serial.println(data_to_write);

  Serial.printf("Writing to file: %s\n", filename_write);
  File file = LittleFS.open(filename_write, FILE_WRITE); // "w" mode: create/overwrite
  if (!file) {
    Serial.println("- Failed to open file for writing");
    return;
  }
  if (file.print(data_to_write)) {
    Serial.println("- File written successfully");
  } else {
    Serial.println("- Write failed");
  }
  file.close(); // Always close the file

  // Leggere un file
  file = LittleFS.open("/welcome.txt", FILE_READ);
  if(!file){
    Serial.println("Failed to open file for reading");
    return;
  }
  Serial.println("File content:");
  String fileContent = file.readString();
  if (fileContent.charAt(fileContent.length() - 3) == 'd') {
    Serial.println("CULOOOOOO");
  }
  
  file.close(); // Chiudi il file appena hai finito di leggere
}
  

void loop() {
}