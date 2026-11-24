
#define IR_PIN 2

void setup() {
  Serial.begin(9600);
  pinMode(IR_PIN, INPUT);
}

void loop() {
  if (digitalRead(IR_PIN) == LOW) {
    Serial.println("Objek Terdeteksi");
  } else {
    Serial.println("Tidak Ada Objek");
  }

  delay(300);
}
