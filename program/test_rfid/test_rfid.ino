#include <SPI.h>
#include <MFRC522.h>

#define SS_PIN 10
#define RST_PIN 9

MFRC522 rfid(SS_PIN, RST_PIN);

int nomorScan = 0;

void setup() {
  Serial.begin(9600);
  SPI.begin();
  rfid.PCD_Init();

  Serial.println("==============================");
  Serial.println("       CEK UID KARTU");
  Serial.println("==============================");
  Serial.println("Tempelkan kartu satu per satu...");
  Serial.println();
}

void loop() {
  if (!rfid.PICC_IsNewCardPresent()) return;
  if (!rfid.PICC_ReadCardSerial()) return;

  nomorScan++;

  String uid = "";
  for (byte i = 0; i < rfid.uid.size; i++) {
    if (rfid.uid.uidByte[i] < 0x10) uid += "0";
    uid += String(rfid.uid.uidByte[i], HEX);
    if (i < rfid.uid.size - 1) uid += " ";
  }
  uid.toUpperCase();

  Serial.print("Scan ke-");
  Serial.println(nomorScan);
  Serial.print("UID     : ");
  Serial.println(uid);
  Serial.print("Size    : ");
  Serial.print(rfid.uid.size);
  Serial.println(" bytes");
  Serial.println("------------------------------");

  rfid.PICC_HaltA();
  rfid.PCD_StopCrypto1();
  delay(1000);
}