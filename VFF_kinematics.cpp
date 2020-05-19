// #include <iostream>
// #include <cmath>
// #include "symbolicc++.h"

// using namespace std;

// int main(){
//     // int a, b;
//     // a = 2; b =5;
//     cout<<"HI";
// }


#include <iostream>
#include "symbolicc++.h"

using namespace std;

int main(void)
{
 Symbolic alpha("alpha");
 Symbolic X, XI, dX, result;

 X = ( ( cos(alpha), sin(alpha) ),
       (-sin(alpha), cos(alpha) ) );

 cout << X << endl;

 XI = X[alpha == -alpha]; cout << XI << endl;
 dX = df(X, alpha);       cout << dX << endl;

 result = XI * dX;        cout << result << endl;
 result = result[(cos(alpha)^2) == 1 - (sin(alpha)^2)];
 cout << result << endl;

 return 0;
}
