#include<dot_ch5.h>

using namespace std;

int main()
{
    float a[N], b[N], c, partial_c[BLOCKS_PER_GRID];

    for (int i = 1; i <= N ; i++)
    {
        a[i-1] = i;
        b[i-1] = 2 * i;
    }
    c = 0;
    DotKernel(a,b,partial_c);
    for (int i = 0; i < BLOCKS_PER_GRID; i++)
    {
        c += partial_c[i];
        //cout << "check c:" << c << endl;
    }

    #define sum_squares(x) (x * (x + 1) * (2 * x + 1) / 6)
    cout << "dot ans(gpu):" << c << endl;
    
    cout << "true ans:" << 2 * sum_squares((float(N))) << endl;

    // free(a);
    // free(b);
    // free(partial_c);

    return 0;
}