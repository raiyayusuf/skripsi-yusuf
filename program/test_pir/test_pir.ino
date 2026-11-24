#define PIR_PIN 2

void setup() {
  Serial.begin(9600);
  pinMode(PIR_PIN, INPUT);
  Serial.println("=== PIR SENSOR TEST ===");
  Serial.println("Tunggu 30-60 detik buat kalibrasi awal...");
  delay(30000); // PIR butuh warm-up time
}

void loop() {
  int gerak = digitalRead(PIR_PIN);

  Serial.println(gerak == HIGH ? "GERAKAN TERDETEKSI!" : "Tidak ada gerakan");

  delay(500);
}