// // VERSI DIGITAL

// #define RAIN_PIN 2

// void setup() {
//   Serial.begin(9600);
//   pinMode(RAIN_PIN, INPUT);
// }

// void loop() {
//   int hujan = digitalRead(RAIN_PIN);

//   Serial.println(hujan == LOW ? "Air Terdeteksi" : "Tidak Ada Air");

//   delay(500);
// }

// VERSI ANALOG

#define RAIN_PIN A0

void setup() {
  Serial.begin(9600);
}

void loop() {
  int hujan = analogRead(RAIN_PIN);

  Serial.print("Nilai Air: ");
  Serial.println(hujan);

  delay(500);
}
