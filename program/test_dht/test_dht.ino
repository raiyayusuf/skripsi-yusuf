#include "DHT.h"

#define DHTPIN 4        // <--- Pindah ke Pin 4
#define DHTTYPE DHT22   // <--- Tetep DHT22

DHT dht(DHTPIN, DHTTYPE);

void setup() {
  Serial.begin(9600);
  dht.begin();
  
  // Header ringkas
  Serial.println("Suhu(C) | Lembap(%) | HeatIndex(C)");
  Serial.println("----------------------------------");
  delay(2000);
}

void loop() {
  float suhu = dht.readTemperature();
  float kelembapan = dht.readHumidity();

  // Cek gagal baca
  if (isnan(suhu) || isnan(kelembapan)) {
    Serial.println("[ERROR] Gagal baca");
    delay(2000);
    return;
  }

  // Hitung Heat Index
  float heatIndex = dht.computeHeatIndex(suhu, kelembapan, false);

  // Tampilan jejer satu baris
  Serial.print(suhu, 1);       // 1 angka di belakang koma biar rapi
  Serial.print(" | ");
  Serial.print(kelembapan, 1);
  Serial.print(" | ");
  Serial.println(heatIndex, 1);

  delay(2000);
}