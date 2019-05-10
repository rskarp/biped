/*
 * Riley Karp
 * CS442
 * Stepping
 * 12 April 2019
 */

 #include <avr/io.h>

 // Initialize global variables
 int periodR = 21;
 int periodL = 21;
 bool moveL = false;
 bool moveR = false;
 int target = 0; // start at 90 degrees (leg straight down)

 // Takes in an angle in degrees (-90 to 90) and returns corresponding pulse value
 int deg2pulse( float deg ) {
  return int(float(38*deg/180) + 21);
 }
 
 // move left to given angle
 void moveLeft( float deg ) {
  target = deg2pulse(deg);
  moveL = true;
//  moveR = true; // uncomment this and the corresponding line below to move both legs at the same time
  TIMSK1 |= 0b000000010; // enable OC1A ISR
  _delay_ms(500);
 }

 // move right leg to given angle
 void moveRight( float deg ) {
  target = deg2pulse(deg);
  moveR = true;
//  moveL = true; // uncomment this and the corresponding line above to move both legs at the same time
  TIMSK1 |= 0b000000010; // enable OC1A ISR
  _delay_ms(500);
 }
 
 // Update leg position
 ISR( TIMER1_COMPA_vect ) {
  if(moveL) { // left leg is moving
    if( target > periodL ) {
      periodL += 1;
    }
    else if( target < periodL ) {
      periodL -= 1;
    }
    else { // period == target
      moveL = false;
      TIMSK1 &= 0b11111101; // disable OC1A ISR
    }
  }
  else if(moveR) { // right leg is moving
    if( target > periodR ) {
      periodR += 1;
    }
    else if( target < periodR ) {
      periodR -= 1;
    }
    else { // period == target
      moveR = false;
      TIMSK1 &= 0b11111101; // disable OC1A ISR
    }
  }
 }

 // Update Motor 1 (left) position
 ISR( TIMER0_OVF_vect ) {
  OCR0A = periodL;
 }

 // Update Motor 2 (right) position
 ISR( TIMER2_OVF_vect ) {
  OCR2A = periodR;
 }

 int main() {
  DDRD = 0b01000000; // configure OC0A pin PD[6] as output (Left leg)
  DDRB = 0b00001000; // configure OC2A pin PB[3] as output (Right leg)

  // Configure TC0 for Motor 1 (left)
  TCCR0A = 0b10000011; // clear OC0A on compare match, Fast PWM mode
  TCCR0B = 0b00000101; // Fast PWM mode, prescale by 1024
  OCR0A = periodL;
  TIMSK0 = 0b00000001; // enable overflow interrupt (16.3ms period)

  // Configure TC2 for Motor 2 (right)
  TCCR2A = 0b10000011; // clear OC2A on compare match, Fast PWM mode
  TCCR2B = 0b00000111; // Fast PWM mode, prescale by 1024
  OCR2A = periodR;
  TIMSK2 = 0b00000001; // enable overflow interrupt (16.3ms period)

  // Configure TC1 to update period (motor position) value
  TCCR1A = 0b01000000; // toggle OC1A on compare match, CTC mode
  TCCR1B = 0b00001010; // set timer to CTC mode, prescale by 8
  OCR1AH = (48192 & 0xFF00) >> 8; // set high byte
  OCR1AL = (48192 & 0x00FF) ; // set low byte
  TIMSK1 = 0b000000000; // disable OC1A ISR (period: 1ms)

  SREG |= 0b10000000; // enable global interrupts

  // initialize legs to be straight down
  moveLeft(0);
  moveRight(0);
  _delay_ms(1000);
  
  while(true){
    // Left Leg: straight down = 0, full forward = -90, full reverse = 90
    // Right Leg: straight down = 0, full forward = 90, full reverse = -90

    // Play with angles and when to move each leg to adjust gait pattern
    moveLeft(-10); // left forward
    moveRight(-10); // right backward
//    moveLeft(0); // left straight
    
    moveRight(10); // right forward
    moveLeft(10); // left backward
//    moveRight(0); // right straight
  }
  return 0;
 }
