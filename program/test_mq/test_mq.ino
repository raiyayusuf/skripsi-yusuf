// // VERSI DIGITAL


// #define MQ_PIN 2

// void setup() {
//   Serial.begin(9600);
//   pinMode(MQ_PIN, INPUT);
// }

// void loop() {
//   int gas = digitalRead(MQ_PIN);

//   Serial.println(gas == LOW ? "Gas Terdeteksi" : "Gas Tidak Terdeteksi");

//   delay(500);
// }


// VERSI ANALOG

#define MQ_PIN A0

void setup() {
  Serial.begin(9600);
}

void loop() {
  int gas = analogRead(MQ_PIN);

  Serial.print("Nilai Gas: ");
  Serial.println(gas);

  delay(500);
}
