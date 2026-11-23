
#include <SPI.h>
#include <MFRC522.h>

#define SS_PIN 10
#define RST_PIN 9

MFRC522 rfid(SS_PIN, RST_PIN);

int jumlahScan = 0;

void setup() {
  Serial.begin(9600);

  SPI.begin();
  rfid.PCD_Init();

  Serial.println("==============================");
  Serial.println("      RFID UID READER");
  Serial.println("        ARDUINO UNO");
  Serial.println("==============================");
  Serial.println("Tempelkan kartu RFID...");
  Serial.println();
}

void loop() {
  // Cek apakah ada kartu baru
  if (!rfid.PICC_IsNewCardPresent()) {
    return;
  }

  // Baca data kartu
  if (!rfid.PICC_ReadCardSerial()) {
    return;
  }

  jumlahScan++;

  Serial.println("------------------------------");
  Serial.print("Kartu ke-: ");
  Serial.println(jumlahScan);

  // Tampilkan UID dalam format HEX
  Serial.print("UID HEX  : ");

  for (byte i = 0; i < rfid.uid.size; i++) {
    if (rfid.uid.uidByte[i] < 0x10) {
      Serial.print("0");
    }

    Serial.print(rfid.uid.uidByte[i], HEX);

    if (i < rfid.uid.size - 1) {
      Serial.print(" ");
    }
  }

  Serial.println();

  // Tampilkan panjang UID
  Serial.print("UID Size : ");
  Serial.print(rfid.uid.size);
  Serial.println(" bytes");

  Serial.println("------------------------------");
  Serial.println("Silakan scan kartu berikutnya...");
  Serial.println();

  // Hentikan komunikasi dengan kartu
  rfid.PICC_HaltA();
  rfid.PCD_StopCrypto1();

  delay(500);
}
