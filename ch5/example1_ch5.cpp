#include<iostream>
#include<add1.h>

int main()
{
    int a[N], b[N], c[N];
    bool success = true;
    for (int i = 0; i < N; i++)
    {
        a[i] = i + 1;
        b[i] = (i + 1) * (i + 1);
    }
    addKernel(a,b,c);
    for (int i = 0; i < N && success; i++)
    {
        if(a[i] + b[i] != c[i])
        {
            success = false;
        }
    }
    if(success) std::cout << "We did it \n"; 
    return 0;
}