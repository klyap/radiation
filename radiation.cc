#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <fstream>
#include <iostream>
#include <math.h>

#include <algorithm>
#include <cassert>
/*#include <cuda_runtime.h> */

#include <glm/glm.hpp>
#include <glm/vec3.hpp> // glm::vec3
#include <glm/vec4.hpp> // glm::vec4, glm::ivec4
#include <glm/mat4x4.hpp> // glm::mat4
#include <glm/gtc/matrix_transform.hpp> // glm::translate, glm::rotate, glm::scale, glm::perspective
#include <glm/gtc/type_ptr.hpp> // glm::value_ptr
#include <glm/gtx/transform.hpp>

/*
using std::cerr;
using std::cout;
using std::endl;
*/
const float PI = 3.14159265358979;


/*
 * NOTE: You can use this macro to easily check cuda error codes 
 * and get more information. 
 * 
 * Modified from:
 * http://stackoverflow.com/questions/14038589/
 *   what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 */
/*
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
    bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
        exit(code);
    }
}
*/

/* Checks the passed-in arguments for validity. */
/*
void check_args(int argc, char **argv) {

    if (argc != 3) {
        cerr << "Incorrect number of arguments.\n";
        cerr << "Arguments: <threads per block> <max number of blocks>\n";
        exit(EXIT_FAILURE);
    }
}
*/

/* m x n are dims of a, p x q are dims of matrix b */
/*
double * matrix_multiply(double *a, double *b, int m, int n, int p, int q){
    
    double *c[m][q];
    int i, j, k;
    for(i=0;i<m;++i)
        {
            for(j=0;j<q;++j)
            {
                c[i][j]=0;
                for(k=0;k<n;++k)
                    c[i][j]=c[i][j]+(a[i][k]*b[k][j]);
                cout<<c[i][j]<<" ";
        }
            cout<<"\n";
        }
        
    return NULL;
}
*/

/*
void stuff(){

    // For each output pixel:
    for (int x = 0; x < 100; x++){
        for (int y = 0; y < 100; y++){
            for (int z = 0; z < 100; z++){
                output[x + 100 * y + z * 100]
                for (int b = 0; b < MAX_BEAM_IDX; b++){
                    int kernel_idx = beam_depths[input_beam_idx];

                }

            }

        }

    }
    
}
*/

// Map kernels to beam based on depth
// Each beam_depths[i] is a kernel index, which goes up to 50
// For now, it's just some arbitrary kernel
int * fillBeamDepths(int *beam_depths, int MAX_BEAM_IDX){
    for (int i = 0; i < MAX_BEAM_IDX; i++){
        beam_depths[i] = i % 50;
    }
    return beam_depths;
}

/*
double * fillKernels(double *kernel_data){
    // for each kernel
    
    for (int l = 0; i < 50; l++){
        // Calculate I for all pixels
        int L = -125 + l * 5;
        for (int x = 0; x < 100; x++){
            for (int y = 0; y < 100; y++){
                for (int z = 0; z < 100; z++){
                    float I = exp(-(500 - z) / 400) / 
                        ((500 - L - z) * (500 - L - z) + x * x + y * y);
                    // TODO: Insert this into matrices 
                }
            }
        }
    }
    
    return NULL;
    
}
*/
double * fillTransform(){return NULL;}

int calculate(int argc, char **argv) {
    //check_args(argc, argv);

    /* Form Gaussian blur vector */
    //float mean = 0.0;
    //float std = 5.0;

    int MAX_BEAM_IDX = 36 * 6;

    double *output = (double *) malloc(100 * 100 * 100 * sizeof (double));

    // Rotation matrices for 60 psi and 6 theta angles
    // One per the 36 * 6 beams
    // Each matrix is 4 x 4
    //double *transform = (double *) malloc((36 * 6) * (4 * 4) * sizeof (double));
    glm::mat4 *transform = (glm::mat4 *) malloc((36 * 6) * sizeof (glm::mat4 *));
    
    // Output data storage for GPU implementation (will write to this from GPU)
    int *beam_depths = (int *) malloc(MAX_BEAM_IDX * sizeof (int));

    // 50 input kernels of 250 x 100 x 100 pixels
    double *kernel_data = (double *) malloc(50 * (250 * 100 * 100) * sizeof (double));
    
    // CPU Computation
    //cout << "CPU computation..." << endl;

    //memset(output_data_host, 0, n_frames * sizeof (float));

    // Use the CUDA machinery for recording time
    
    /*
    cudaEvent_t start_cpu, stop_cpu;
    cudaEventCreate(&start_cpu);
    cudaEventCreate(&stop_cpu);
    cudaEventRecord(start_cpu);
    */

    // Populate kernels
    beam_depths = fillBeamDepths(beam_depths, MAX_BEAM_IDX);
    //kernel_data = fillKernels(kernel_data);
    //stuff();
    // Process



    // Stop timer
    /*
    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);
    */

    // Stop timer
    //cudaEventRecord(stop_gpu);
    //cudaEventSynchronize(stop_gpu);
    
    /*
    cout << "Comparing..." << endl;

    float cpu_time_milliseconds = -1;

    cudaEventElapsedTime(&cpu_time_milliseconds, start_cpu, stop_cpu);

    cout << endl;
    cout << "CPU time: " << cpu_time_milliseconds << " milliseconds" << endl;
    */
    

    // Free memory on host
    free(transform);
    free(output);
    free(kernel_data);


    return EXIT_SUCCESS;
}


int main(int argc, char **argv) {

    return calculate(argc, argv);
}


