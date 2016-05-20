

#define 
GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/vec3.hpp> // glm::vec3
#include <glm/vec4.hpp> // glm::vec4
#include <glm/mat4x4.hpp> // glm::mat4
#include <glm/gtc/matrix_transform.hpp> // glm::translate, glm::rotate, glm::scale, glm::perspective
#include <glm/gtc/type_ptr.hpp>

#include <stdio.h>
#include <stdlib.h>

int main(){
    float aaa[16] = {
       1, 2, 3, 4,
       5, 6, 7, 8,
       9, 10, 11, 12,
       13, 14, 15, 16
    };

    glm::mat4 bbb = glm::make_mat4(aaa);
    
    glm::mat4 *b;
    *b = bbb;
    printf(b);
    return 0;
}