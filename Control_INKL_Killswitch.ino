#include <RCSwitch.h>
#include <Servo.h>


//BUTTONS
const long knapA = 5592569;
const long knapB = 5592570;
const long knapC = 5592571;
const long knapD = 5592572;

RCSwitch mySwitch = RCSwitch();
unsigned long recievedValue;
int buttonstate = 1;

//SERVO CONTROL
const int turnServoPin = 5;
const int motorsPin = 6;
long number = 100101;  //Speed 100 Angle 101
bool step = false;

Servo turnServo;
Servo motors;

void setup() {
  Serial.begin(9600);
  //SETUP RF RECIEVER
  mySwitch.enableReceive(0);  // Receiver on interrupt 0 => that is pin #2

  //SETUP MOTORS
  turnServo.attach(turnServoPin);
  motors.attach(motorsPin);
  turnServo.write(90);
  motors.write(90);
  delay(1000);
  Serial.println("Program starter");
  Serial.setTimeout(5); //200 opdateringer i sekundet max på Serial
}

void loop() {
  if (mySwitch.available()) {
    knapInput(); //Checks what button has been pressed
    mySwitch.resetAvailable(); //Resets rf reciever
  }

  if (buttonstate > 0) {
    motorControl();
  } else {
    turnServo.write(90);  //Stop car immediately
    motors.write(90); //Sets the wheels straight
  }
}



void knapInput() {
  if (mySwitch.getReceivedValue() == knapA) {
    buttonstate = 0;
  } else if (mySwitch.getReceivedValue() == knapB) {
    buttonstate = 1;
  } else if (mySwitch.getReceivedValue() == knapC) {
    buttonstate = 2;
  } else if (mySwitch.getReceivedValue() == knapD) {
    buttonstate = 3;
  }
}

void motorControl() {
  if (Serial.available()) {  // Check if there is data available to read
    number = Serial.readString().toInt();   // Read and convert the input to an integer              parseInt(); på egen computer(arduino IDE)                 readString().toInt(); på jetson nano(python)
    
    if (number != 0) {
      Serial.println("Motorspeed:"); 
      Serial.println(number / 1000);
      motors.write(number / 1000); //Gets the 3 first digits of the 6 numbers

      Serial.println("TurnAngle:");
      Serial.println(number % 1000);
      turnServo.write(number % 1000); //Gets the 3 last digits of the 6 numbers
    }
  }
}