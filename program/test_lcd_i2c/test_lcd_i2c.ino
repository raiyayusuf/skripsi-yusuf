#include <Wire.h>
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27, 16, 2);

byte heart[8] = {
  0b00000,
  0b01010,
  0b11111,
  0b11111,
  0b11111,
  0b01110,
  0b00100,
  0b00000
};

void setup() {
  lcd.init();
  lcd.backlight();
  lcd.createChar(0, heart);
}

void loop() {
  const char* text = "Trainer Kit Arduino Uno Siap Digunakan  ";
  int len = strlen(text);

  for (int i = 0; i <= len; i++) {
    lcd.clear();
    lcd.setCursor(0, 0);
    for (int j = 0; j < 16; j++) {
      int idx = i + j;
      if (idx < len) {
        lcd.print(text[idx]);
      } else {
        lcd.print(' ');
      }
    }
    lcd.setCursor(0, 1);
    lcd.print("LCD I2C OK ");
    lcd.write(byte(0));
    delay(300);
  }
}