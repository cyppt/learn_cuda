#include<gl_cuda.h>

GLuint bufferObj;
cudaGraphicsResource *resource;

__global__ void kernel(uchar4 *ptr)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float fx = x/(float)DIM - 0.5f;
    float fy = y/(float)DIM - 0.5f;

    unsigned char green = 128 + 127 * sin(abs(fx * 100) - abs(fy * 100));

    ptr[offset].x = 0;
    ptr[offset].y = green;
    ptr[offset].z = 0;
    ptr[offset].w = 255;
}


// key_func uses a switch statement to define the behaviour of given input from the user (key, x, y). 
// If 27 is selected as key(Esc button), GPU resources are unregistered, binds a buffer to 0 and deletes it before the application exits (exit(0);).
static void key_func(unsigned char key, int x, int y)
{
    switch(key)
    {
        case 27:
        cudaGraphicsUnregisterResource(resource);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
        glDeleteBuffers(1, &bufferObj);
        exit(0);
    }
}

static void draw_func()
{   
    // Draw the pixels in the RGB buffer with the given dimensions (DIM) and a format of unsigned bytes
    glDrawPixels(DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    // Exchange the back buffer with the front buffer on the display device to show the image on the screen
    glutSwapBuffers();
    // int gl_error = glGetError();
    // std::cout <<"gl_error num:" << gl_error << std::endl;
}

extern "C" void GlCudaKernel(int *argc, char **argv)
{
    cudaDeviceProp prop;
    int dev;

    // Set properties for a CUDA device
    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1;
    prop.minor = 0;
    // Choose the device with the given properties
    HANDLE_ERROR(cudaChooseDevice(&dev,&prop));
    // Set GL device with the chosen device number
    HANDLE_ERROR(cudaGLSetGLDevice(dev));

    /* Initialize the GLUT library */
    glutInit(argc, argv);
    /* Set up a double buffer and RGBA colors for the display mode */
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    /* Set window size */
    glutInitWindowSize(DIM, DIM);
    /* Create a window with the given title */
    glutCreateWindow("bitmap"); 

    // Generate 1 buffer object, bind it to GL_PIXEL_UNPACK_BUFFER_ARB,
    // and assign dynamic memory (size of DIM x DIM x 4) to it 
    glGenBuffers(1, &bufferObj); 
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj); 
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, DIM * DIM * 4, NULL, GL_DYNAMIC_DRAW_ARB);
    
    // Register buffer object with cuda

    /*  !!!!报错!!!：all CUDA-capable devices are busy or unavailable */
    /*
    查资料发现：  
    在WSL2上是无法使用OpenGL-CUDA的，详见官网：
    https://docs.nvidia.com/cuda/wsl-user-guide/index.html#features-not-yet-supported
    的4.2. Features Not Yet Supported
    OpenGL-CUDA Interop is not yet supported. Applications relying on OpenGL will not work.
    */
    HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone));

    uchar4* devPtr;
    size_t size;
    // Register buffer object with CUDA and map it to resource
    HANDLE_ERROR(cudaGraphicsMapResources(1, &resource, NULL));   
    // Get the pointer to the mapped resource so we can access it in a CUDA kernel
    HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource));

    dim3 grids(DIM/16, DIM/16);
    dim3 threads(16,16);
    kernel<<<grids, threads>>>(devPtr);

    // unmap the CUDA resource from the buffer object
    HANDLE_ERROR(cudaGraphicsUnmapResources(1, &resource, NULL));

    /* Register keyboard callback for the specified key press */
    glutKeyboardFunc(key_func);
    /* Display the current window with OpenGL drawing function */
    glutDisplayFunc(draw_func);
    /* Begins the primary event loop and will not return until after glutLeaveMainLoop has been called */
    glutMainLoop();
}