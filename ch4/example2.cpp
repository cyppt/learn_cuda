#include<iostream>
#include<bitmap.h>

using namespace std;

int main()
{
    unsigned char bitmap[DIM][DIM];
    JuliaKernel(bitmap);
    cout << "--------------------------\n";
    for(int i = 0 ; i < DIM; i++)
    {
        for( int j = 0; j < DIM; j++)
        {
            if(bitmap[i][j] == 1) cout << "*";
            else cout << "_";
        }
        cout << endl;
    }
    cout << "--------------------------\n";
    return 0;
}