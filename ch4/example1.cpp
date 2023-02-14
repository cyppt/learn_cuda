#include<iostream>
#include<add.h>

int main()
{
    int a[N], b[N], c[N];
    for (int i = 0; i < N; i++)
    {
        a[i] = -i;
        b[i] = i * i;
    }

    addKernel(a, b, c);

    for (int i =0; i < N; i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }
    return 0;
}