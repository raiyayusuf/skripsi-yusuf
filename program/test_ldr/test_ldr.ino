// #define LDR_PIN 2

// void setup() {
//   Serial.begin(9600);
//   pinMode(LDR_PIN, INPUT);
// }

// void loop() {
//   int cahaya = digitalRead(LDR_PIN);

//   Serial.println(cahaya == HIGH ? "Terang" : "Gelap");

//   delay(500);
// }

#define LDR_PIN A0

void setup() {
  Serial.begin(9600);
}

void loop() {
  int cahaya = analogRead(LDR_PIN);

  Serial.print("Nilai Cahaya: ");
  Serial.println(cahaya);

  delay(500);
}