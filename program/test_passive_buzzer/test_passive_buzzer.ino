// Pin buat buzzer
#define BUZZER_PIN 8

// Frekuensi Nada (dalam Hz)
#define NOTE_C4  262
#define NOTE_D4  294
#define NOTE_E4  330
#define NOTE_F4  349
#define NOTE_G4  392
#define NOTE_A4  440
#define NOTE_B4  494
#define NOTE_C5  523

// Array nada (Do Re Mi Fa Sol La Si Do)
int melodi[] = {
  NOTE_C4, NOTE_D4, NOTE_E4, NOTE_F4, 
  NOTE_G4, NOTE_A4, NOTE_B4, NOTE_C5
};

void setup() {
  Serial.begin(9600);
  Serial.println("=== Buzzer Do-Re-Mi ===");
  Serial.println("Mulai...");
}

void loop() {
  // Hitung jumlah nada di dalam array
  int jumlahNada = sizeof(melodi) / sizeof(melodi[0]);

  // Mainkan nada satu per satu
  for (int i = 0; i < jumlahNada; i++) {
    // Bunyikan nada
    tone(BUZZER_PIN, melodi[i]);
    
    // Print ke Serial Monitor biar tau nada apa yang lagi bunyi
    Serial.print("Nada ke-");
    Serial.print(i + 1);
    Serial.print(" (");
    Serial.print(melodi[i]);
    Serial.println(" Hz)");
    
    // Delay 1 detik (sesuai request)
    delay(1000);
    
    // Matikan bunyi sebentar biar antar nada ada jeda
    noTone(BUZZER_PIN);
    delay(50); // jeda 50ms doang biar gak nyambung
  }
  
  Serial.println("--- Ulang dari awal ---");
  delay(1000); // jeda 1 detik sebelum ngulang dari Do lagi
}