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
// http://www.thecrazyprogrammer.com/2012/09/c-program-to-multiply-two-matrices.html
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
                //cout<<c[i][j]<<" ";
        }
            //cout<<"\n";
        }
        
    return NULL;
}
*/



/* Map kernels to beam based on depth
 * Each beam_depths[i] is a kernel index, which goes up to 50
 * For now, it's just some arbitrary kernel
 */
int * fillBeamDepths(int *beam_depths, int MAX_BEAM_IDX){
    for (int i = 0; i < MAX_BEAM_IDX; i++){
        beam_depths[i] = i % 50;
    }
    return beam_depths;
}


/* Generate kernel matrices and store in an array */
double ** fillKernelData(double **kernel_data){
    // for each kernel in kernel_data:
    for (int l = 0; i < 50; l++){
        //// Making a new kernel matrix ////
        int L = -125 + l * 5;
        
        float *matrix[x][y][z];
        for (int x = 0; x < 100; x++){
            for (int y = 0; y < 100; y++){
                for (int z = 0; z < 100; z++){
                    // Get value at pixel
                    float I = exp(-(500 - z) / 400) / 
                        ((500 - L - z) * (500 - L - z) + x * x + y * y);
                    // Insert value into matrix
                    *matrix[x][y][z] = I;
                }
            }
        }

        //// Insert matrix into kernel_data ////
        *kernel_data[l] = matrix;
    }
    
    return kernel_data;
}


/*
Makes an array of transform matrices that do all
combinations of theta and phi rotations.
*/
glm::mat4 * fillTransform(glm::mat4 *transform){
    int phi, theta;
    int i = 0;
    for (phi = 0; phi < 360; phi = phi + 10){
        for (theta = 0; theta < 60; theta = theta + 10) {
            // TODO: hardcoding psi for now
            int psi = 0;

            //// Make the transform arrays ////
            /*
            // Let phi be X, theta be Y, psi by Z arbitrarily
            glm::mat4 phi_rotation = glm::eulerAngleX(phi);
            glm::mat4 theta_rotation = glm::eulerAngleY(theta);
            glm::mat4 psi_rotation = glm::eulerAngleY(psi);

            // Combine the rotations into one matrix
            glm::mat4 rotation = 
                phi_rotation * theta_rotation * psi_rotation;
            */

            // Convert to radians
            phi = phi / 360.0 * PI;
            theta = theta / 360.0 * PI;
            psi = psi / 360.0 * PI;

            // Make rotation matrix
            glm::mat4 rotation = 
                glm::eulerAngleYXZ(phi, theta, psi);
            
            //// Add this rotation to array ////
            *(transform[i]) = rotation;

            // Go do next matrix
            i++;
        }
    }

    return transform;
}

/* Linear interpolation 
   Inputs: a, b, t (linear coords)
       a and b are bounds, t is target
*/
double linear(double a, double b, double a_val, double, b_val, double t){
    double t_val = 
        a_val + ((t - a)/(b - a) * (b_val - a_val));

    return t_val;
}

/* Linear interpolation 
   Inputs: a, b, c, d: arrays of size 2
        Each represents a point in kernel
        a[0] = a.x, a[1] = a.y
    
*/
double bilinear(
    double *a, double *b, double *c, double *d,
    double a_val, double b_val, double c_val, double d_val,
    double *t,){
    
    //// Interpolate along x axis ////
    
    // Value of point between top 2 points
    // t1.x = t.x = a.y = b.y
    t1_val = linear(a[0], b[0], a_val, b_val, t[0]);
    // Value of point between bottom 2 points
    // t2.x = t.x, t2.y = c.y = d.y
    t2_val = linear(c[0], d[0], c_val, d_val, t[0]);

    //// Interpolate along y-axis ////
    
    // Value of point between 2 interpolated points
    // Note: t1.y = a[1], t2.y = c[1]
    t_val = linear(a[1], c[1], t1_val, t2_val, t[1]);

    return t_val;
}

double trilinear(kernel_data[kernel_idx], loc_x, loc_y, loc_z){
    //// Find closest 8 points in kernel (whole # coords) to target ////
    
    //// 
    return NULL;
}


void getDose(double *out_data, int beam_depths, glm::mat4 *transform, double **kernel_data){
    // For each beam:
    for (int b = 0; b < MAX_BEAM_IDX; b++){
        int input_beam_idx = b;
        
        //// Get parameters for this beam ////
        
        // Choose kernel matrix index for this beam
        int kernel_idx = beam_depths[input_beam_idx];
        // Get this beam's rotation transform matrix
        glm::mat4 A = transforms[input_beam_idx];

        //// Get this beam's dose for each pixel ////
        for (int x = 0; x < 100; x++){
            for (int y = 0; y < 100; y++){
                for (int z = 0; z < 100; z++){
                    // Get current coordinates
                    glm::vec4 output_pixel_loc = glm::vec4(glm::vec3(x, y, z), 1.0);
                    
                    // Get transformed location
                    glm::vec4 transformed_loc = glm::rotate(A, output_pixel_loc);
                    // glm::vec3 transformed_loc = A * output_pixel_loc;
                    double loc_x = transformed_loc[0];
                    double loc_y = transformed_loc[1];
                    double loc_z = transformed_loc[2];
                    // Get trilinearly interpolated dosage on transformed location
                    dose = trilinear(kernel_data[kernel_idx], loc_x, loc_y, loc_z);
                    // Add to current coordinate's total dosage value
                    output_data[x][y][z] += dose
                }
            }
        }
    }
}






int calculate(int argc, char **argv) {
    //check_args(argc, argv);

    int MAX_BEAM_IDX = 36 * 6;

    // output[x][y][z] returns dosage value at that coordinate
    double *output = (double *) malloc(100 * 100 * 100 * sizeof (double));

    // Rotation matrices for 60 psi and 6 theta angles
    // One per the 36 * 6 beams
    // Each matrix is 4 x 4
    //double *transform = (double *) malloc((36 * 6) * (4 * 4) * sizeof (double));
    glm::mat4 *transform = (glm::mat4 *) malloc((36 * 6) * sizeof (glm::mat4 *));
    
    // beam_depths[beam_idx] returns the corresponding kernel_idx
    int *beam_depths = (int *) malloc(MAX_BEAM_IDX * sizeof (int));

    // 50 input kernels of 250 x 100 x 100 pixels for each beam
    double **kernel_data = (double *) malloc(50 * (250 * 100 * 100) * sizeof (double));
    
    /*
    // CPU Computation
    //cout << "CPU computation..." << endl;

    //memset(output_data_host, 0, n_frames * sizeof (float));

    // Use the CUDA machinery for recording time
    
    cudaEvent_t start_cpu, stop_cpu;
    cudaEventCreate(&start_cpu);
    cudaEventCreate(&stop_cpu);
    cudaEventRecord(start_cpu);
    */


    //// Populate input variables ////
    beam_depths = fillBeamDepths(beam_depths, MAX_BEAM_IDX);
    kernel_data = fillKernelData(kernel_data);
    transform = fillTransform(transform);

    //// Run actual algorithm ////
    out_data = getDose(out_data, beam_depths, transform, kernel_data);

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

    //// Set up input variables and run actual dosage calcutations ///
    return calculate(argc, argv);
}


