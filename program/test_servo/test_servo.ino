// #include <Servo.h>

// Servo servo;

// void setup() {
//   servo.attach(3);
//   servo.write(0);
// }

// void loop() {
// }

#include <Servo.h>

Servo servo;

void setup() {
  servo.attach(3);
  servo.write(0);
}

void loop() {
  delay(1000);

  for (int sudut = 0; sudut <= 90; sudut++) {
    servo.write(sudut);
    delay(15);
  }
  delay(1000);

  for (int sudut = 91; sudut <= 180; sudut++) {
    servo.write(sudut);
    delay(15);
  }
  delay(1000);

  for (int sudut = 179; sudut >= 90; sudut--) {
    servo.write(sudut);
    delay(15);
  }
  delay(1000);

  for (int sudut = 89; sudut >= 0; sudut--) {
    servo.write(sudut);
    delay(15);
  }
  delay(1000);
}