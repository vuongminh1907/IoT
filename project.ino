#include <SoftwareSerial.h>
 
#define TX_PIN      7
#define RX_PIN      6
#define LED_PIN 13

// adjust the leds
unsigned long previousMillis = 0;
const long interval = 500; 
const int greenPin = 3;
const int redPin = 4;
const int yellowPin = 5;
bool ledState = false;

// receive data from laptop
/*
state = 0: the led is off
state = 1: the led is on but flickering
state = 2: the led is on
*/
String data = ""; 
int state = 0;    
int volume = 0; 

//adjust rotation motor speed
int in1 = 9;
int in2 = 10;

SoftwareSerial bluetooth(RX_PIN, TX_PIN);
int baudRate[] = {300, 1200, 2400, 4800, 9600, 19200, 38400, 57600, 115200};
 
void setup() {

  pinMode(LED_PIN, OUTPUT);

  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);

  Serial.begin(9600);
  while (!Serial) {}
  
  Serial.println("Configuring, please wait...");
  
  for (int i = 0 ; i < 9 ; i++) {
     bluetooth.begin(baudRate[i]);
     String cmd = "AT+BAUD4";
     bluetooth.print(cmd);
     bluetooth.flush();
     delay(100);
  }
  
  bluetooth.begin(9600);
  Serial.println("Config done");
  while (!bluetooth) {}
  
  Serial.println("Enter AT commands:");
}
 
void loop() {

  if (Serial.available()){
    data = Serial.readStringUntil('\n');
    parseData(data);
    if (state == 0){
      stopMotor();
      if (bluetooth.available()){
        sendDataToPhone(state, volume);
      }
      turnonRed();
    } else if (state == 1){
      unsigned long currentMillis = millis();
      if (currentMillis - previousMillis >= interval) {
        // Đã đến thời điểm cần thay đổi trạng thái đèn
        previousMillis = currentMillis; // Cập nhật thời gian
        flickerYellow();
      }
      while (state == 1){
        data = Serial.readStringUntil('\n');
        parseData(data);
        adjustMotor(volume);
        if (bluetooth.available()){
          sendDataToPhone(state, volume);
        }
        if (currentMillis - previousMillis >= interval) {
          // Đã đến thời điểm cần thay đổi trạng thái đèn
          previousMillis = currentMillis; // Cập nhật thời gian
          flickerYellow();
        }
      }
      
    } else {
      isoSpeed();
      if (bluetooth.available()){
        sendDataToPhone(state, volume);
      }
      turnonGreen();
    }
  }
  
}

// function to process data received from laptop 
void parseData(String input){
  input.trim();
  if (input.startsWith("(") && input.endsWith(")")) {
    input = input.substring(1, input.length() - 1); // Loại bỏ dấu ngoặc
    int commaIndex = input.indexOf(','); // Vị trí dấu phẩy

    if (commaIndex != -1) {
      // Lấy state và volume từ chuỗi
      String stateStr = input.substring(0, commaIndex).trim();
      String volumeStr = input.substring(commaIndex + 1).trim();

      // Chuyển đổi giá trị
      state = stateStr.toInt(); // Chuyển state sang số nguyên
      volume = volumeStr.toInt(); // Chuyển volume sang số nguyên
    }
  }else{
    Serial.println("Error: Data must have format (state, volume)")
  }
}

void stopMotor(){
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  Serial.println("Fan is off")
}

void turnonRed(){
  digitalWrite(redPin, HIGH);
  digitalWrite(yellowPin, LOW);  
  digitalWrite(greenPin, LOW);
}

void adjustMotor(volume){
  analogWrite(in1, volume);
  analogWrite(in2, LOW);
  Serial.print("Motor running at speed: ");
  Serial.println(volume);
}

void flickerYellow(){
  digitalWrite(redPin, LOW);
  digitalWrite(yellowPin, !digitalRead(yellowPin));  
  digitalWrite(greenPin, LOW);
}

void isoSpeed(){
  analogWrite(in1, 255);
  digitalWrite(in2, LOW);
  Serial.println("Fan is on");
}

void turnonGreen(){
  digitalWrite(redPin, LOW);
  digitalWrite(yellowPin, LOW);  
  digitalWrite(greenPin, HIGH);
}

void sendDataToPhone(int state, int volume){
  String message = "State: " + String(state) + ", Volume: " + String(volume);
  bluetooth.println(message);  // Send message to Bluetooth
  Serial.print("Sending to phone: ");
  Serial.println(message);
}