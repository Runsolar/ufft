/*
 ufft v.1.0
 Uno Fast Fourier Transform for real time spectrum analisys for speech recognition 
 in Arduino (UNO) projects and other low energy systems. 
 This opensource program for engineering and scientific purposes.
 It uses integer rounding and long type in complex computation.
 Tested on Arduino UNO: ATMega 328, 16MHz.
 Results: Elapsed time 16ms for 128 data length, and 31ms for 256.
 Free license.
 Maked by Danijar Wolf, 2016.
 */

#include <avr/pgmspace.h>
#include "math.h"

//Here set your length data
#define L 128

#define K L/4
#define LOG2(x) (log(x)/log(2))

//Dynamic cosine wave data occupies sizeof(int)*(L-K) bytes in SRAM 
int *pCosinewave;

//Uncomment if you want use program memory data (for 128 data length or add yours)
/*
const PROGMEM int Cosinewave[L-K] = {
128,128,127,127,126,124,122,121,
118,116,113,110,106,103,99,
95,91,86,81,76,71,66,
60,55,49,43,37,31,25,
19,13,6,0,-6,-13,-19,
-25,-31,-37,-43,-49,-55,-60,
-66,-71,-76,-81,-86,-91,-95,
-99,-103,-106,-110,-113,-116,-118,
-121,-122,-124,-126,-127,-127,-128,
-128,-128,-127,-127,-126,-124,-122,
-121,-118,-116,-113,-110,-106,-103,
-99,-95,-91,-86,-81,-76,-71,
-66,-60,-55,-49,-43,-37,-31,
-25,-19,-13,-6

//0,6,13,19,25,31,37,43,49,55,
//60,66,71,76,81,86,91,
//95,99,103,106,110,113,116,
//118,121,122,124,126,127,127,
//128
};
*/

//Tested speech vector, sample rate 8192 Hz.
const PROGMEM int Vector[256] = {
2,1,0,0,2,5,3,0,
-2,0,2,3,4,7,6,
-5,-8,0,5,1,-6,-7,
-3,11,24,18,-2,-23,-35,
-37,-16,14,29,27,15,-8,
-28,-30,-22,-8,4,7,0,
2,12,18,11,-2,-12,-10,
12,30,24,13,21,-5,-46,
-22,33,40,-1,-27,-35,-10,
60,91,31,-45,-82,-105,-86,
-1,71,67,36,2,-35,-47,
-39,-33,-30,-6,11,5,13,
42,55,27,-14,-44,-21,46,
84,70,41,-21,-98,-71,40,
106,58,-15,-56,-49,35,128,
99,-31,-125,-154,-130,-32,85,
98,34,-13,-46,-57,-39,-22,
-35,-39,-20,-9,9,45,67,
45,12,-18,-39,-5,66,110,
102,39,-92,-136,-28,98,107,
24,-32,-46,-1,93,117,6,
-112,-155,-137,-53,66,110,50,
-19,-54,-62,-41,-16,-14,-27,
-21,-5,5,17,33,32,12,
1,-11,-1,28,61,75,54,
-30,-99,-57,37,77,40,-7,
-34,-19,39,76,34,-45,-94,
-99,-55,18,63,45,0,-29,
-36,-24,-10,-10,-17,-18,-9,
1,14,21,15,1,-5,-6,
-3,12,25,32,30,1,-36,
-33,3,28,21,5,-6,-7,
7,21,14,-8,-26,-30,-20,
0,17,16,1,-10,-11,-6,
-2,-1,-4,-6,-4,-1,2,
6,6,2,-2,-3,-2,1,
6,9,9};

//long type Complex class
class Complex
{
public:
    long re;
    long im;   
public:
    Complex(const long r=0, const long i=0) : re(r), im(i) {};
    long real() { return re; };
    long imag() { return im; };
            
    ~Complex() {}
 
    //MATH
    Complex operator + (const Complex &c)
    {
      return Complex(re + c.re, im + c.im);
    }
    
    Complex operator - (const Complex &c)
    {
      return Complex(re - c.re, im - c.im);
    }
    
    Complex operator * (const Complex &c)
    {
      long r = re * c.re - im * c.im;
      long i = re * c.im + im * c.re;     
      return Complex(r, i);  
    } 
    
    Complex operator >> (const int c)
    {     
      return Complex(re >> c, im >> c);  
    } 
};

//Init dynamic cosine wave data
static void InitCosinewave()
{
  pCosinewave = (int *)malloc((L-K)*sizeof(int));
  for(int i=0; i<L; i++) 
  {
    pCosinewave[i] = (int) L*(double)cos(-2*PI*i/L);
  }
}

static void DestroyCosinewave()
{
  free(pCosinewave);
}

static Complex w(int z, int p)
{ 
  int m = L / p;
  int i = m * z;
  
  //Uncomment if you want use program memory data
  //int c = (int)pgm_read_word(Cosinewave + i);
  //int s = (int)pgm_read_word(Cosinewave + i + K);
  
  int c = pCosinewave[i];
  int s = pCosinewave[i + K];
  return Complex(c, s);
}

static int Bitreverse(int j, int N)
{
  int nu = 0;
  for (int tmp_size = N; tmp_size > 1; tmp_size /= 2, nu++);
  int ans = 0;

  for (int i = 0; i < nu; i++)
  {
    ans = (ans << 1) | (j & 1);
    j = j >> 1;
  }
  return ans;
}

static void ufft(int *Re, int *Im, int N)
{
  for (int i = 0; i < N; i++)
  {
    int n = Bitreverse( i, N );

    if (i < n)
    {
      int tmp = Re[i];
      Re[i] = Re[n];
      Re[n] = tmp;
    }
  }

  for (int i = 0; i < N - 1; i = i + 2)
  {
    int tmp = Re[i];
    Re[i] = Re[i] + Re[i + 1];
    Re[i + 1] = tmp - Re[i + 1];
  }

  int M = LOG2(N);
  int k = 1;
  int p = 2;
  for (int stage = 0; stage < M-1; stage++)
  { 
    int m = 0;
    int z = 0;
    for (int i = 0; i < N - 2; i = i + 2)
    {
      (z == p) ? z = 0 : 0;
      
      int k2 = k + k;
      int ind0 = i + k2;
      int ind1 = i + 1;
      int ind2 = ind1 + k2;
      
      Complex X_even(Re[i], Im[i]);
      Complex X_odd(Re[ind0], Im[ind0]);

      Complex Y_even(Re[ind1], Im[ind1]);
      Complex Y_odd(Re[ind2], Im[ind2]);

      Complex X_tmp = 0;
      Complex Y_tmp = 0;
      
      X_tmp = X_even;
      X_odd = w(z, p << 1) * X_odd;
      X_odd = X_odd >> M;     
      X_even = X_even + X_odd;
      X_odd = X_tmp - X_odd;

      z++;
      
      Y_tmp = Y_even;
      Y_odd = w(z, p << 1) * Y_odd; 
      Y_odd = Y_odd >> M;      
      Y_even = Y_even + Y_odd;
      Y_odd = Y_tmp - Y_odd;

      z++;
      
      Re[i] = (int) X_even.real();
      Im[i] = (int) X_even.imag();

      Re[ind1] = (int) Y_even.real();
      Im[ind1] = (int) Y_even.imag();
      
      Re[ind0] = (int) X_odd.real();     
      Im[ind0] = (int) X_odd.imag();
    
      Re[ind2] = (int) Y_odd.real();
      Im[ind2] = (int) Y_odd.imag();    

      m++;
      if (m == k)
      {
        i = i + k2;
        m = 0;
      }
    }
    z = 0;
    k = k << 1;
    p = p << 1;
  }
}

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  while (!Serial);
  
  //Init cosine wave data
  InitCosinewave();
}


void loop() {
  // put your main code here, to run repeatedly:

//Example
  int N=L;
  int *Re = (int *)malloc(N*sizeof(int));
  int *Im = (int *)malloc(N*sizeof(int));
     
  unsigned long time1=0, time2=0;
  
  for(int i=0; i<N; i++)
  {
    int v=pgm_read_word(Vector+i);
    Re[i]=(int)v;
    Im[i]=0;  
  }

time1=millis(); 
  ufft(Re, Im, N);
time2=millis();


  for(int i=0; i<N; i++)
  {
    float R=(float)Re[i]*Re[i];
    float I=(float)Im[i]*Im[i];
    Serial.println(sqrt(R+I));
  }

  Serial.print("Elapsed time (ms): ");
  Serial.println(time2-time1); 
 
  delay(1000);
  
  free(Re);
  free(Im);
}
