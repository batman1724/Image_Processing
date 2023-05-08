#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define BLOCK_SIZE 32
#define KERNEL_SIZE 3
#define STRIDE 1

typedef struct rgb{
    unsigned int r;
    unsigned int g;
    unsigned int b;
} rgb;

__global__ void denoise_kernel(rgb* input, rgb* output, int width, int height)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > 0 && j > 0 && i < width - 1 && j < height - 1)
    {
        int idx = j * width + i;
        output[idx].r = (2 * input[idx].r + input[idx - 1].r + input[idx + 1].r + input[idx - width].r + input[idx - width+1].r +input[idx - width-1].r+ input[idx + width].r+input[idx + width+1].r+input[idx + width-1].r) / 10;
        output[idx].g = (2 * input[idx].g + input[idx - 1].g + input[idx + 1].g + input[idx - width].g +  input[idx - width+1].g +input[idx - width-1].g+ input[idx + width].g+input[idx + width+1].g+input[idx + width-1].g) / 10;
        output[idx].b = (2 * input[idx].b + input[idx - 1].b + input[idx + 1].b + input[idx - width].b +  input[idx - width+1].b +input[idx - width-1].b+ input[idx + width].b+input[idx + width+1].b+input[idx + width-1].b) / 10;
    }
}

__global__ void contrastEnhancementLuminosity(rgb *input, rgb *output, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height) {
        int index = j * width + i;
        float r = input[index].r;
        float g = input[index].g;
        float b = input[index].b;
        float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
        output[index].r = fminf(fmaxf((r - lum) * 1.2f + lum, 0.0f), 255.0f);
        output[index].g = fminf(fmaxf((g - lum) * 1.2f + lum, 0.0f), 255.0f);
        output[index].b = fminf(fmaxf((b - lum) * 1.2f + lum, 0.0f), 255.0f);
    }
}


__global__ void histogram_equalization_kernel(rgb* input, rgb* output, int width, int height) {

    // Get the index of the current thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    // Check if the thread is within the bounds of the image
    if (x < width && y < height) {
        // Compute the histogram
        int hist[256] = { 0 };
        for (int i = 0; i < width * height; i++) {
            int gray = (input[i].r + input[i].g + input[i].b) / 3;
            hist[gray]++;
        }

        // Compute the cumulative distribution function (CDF) of the histogram
        int cdf[256] = { 0 };
        for (int i = 0; i < 256; i++) {
            cdf[i] = (i > 0) ? cdf[i - 1] + hist[i] : hist[i];
        }

        // Normalize the CDF
        for (int i = 0; i < 256; i++) {
            cdf[i] = (cdf[i] * 255) / (width * height);
        }

        // Apply the transformation to the input pixel
        int gray = (input[idx].r + input[idx].g + input[idx].b) / 3;
        output[idx].r = cdf[gray];
        output[idx].g = cdf[gray];
        output[idx].b = cdf[gray];
    }
}

__global__ void color_correction_kernel(unsigned char *image, int width, int height, float red_factor, float green_factor, float blue_factor)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index = (y * width + x) * 3;
        unsigned char r = image[index];
        unsigned char g = image[index + 1];
        unsigned char b = image[index + 2];

        r = min(255.0f, max(0.0f, r * red_factor));
        g = min(255.0f, max(0.0f, g * green_factor));
        b = min(255.0f, max(0.0f, b * blue_factor));

        image[index] = r;
        image[index + 1] = g;
        image[index + 2] = b;
    }
}


__global__ void image_sharpening_kernel(rgb* input, rgb* output, int width, int height) {
    
    // Get the index of the current thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    // Check if the thread is within the bounds of the image
    if (x < width && y < height) {

        // Define the sharpening filter
       float kernel[3][3] = {
           {-1, -1, -1},
           {-1, 9, -1},
           {-1, -1, -1}
        };

        // Initialize the pixel values for the output image
        float r = 0;
        float g = 0;
        float b = 0;

        // Compute the sharpened pixel values
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int row = y + i;
                int col = x + j;

                // Make sure the pixel is within the bounds of the image
                if (row >= 0 && row < height && col >= 0 && col < width) {
                    int offset = row * width + col;
                    r += input[offset].r * kernel[i + 1][j + 1];
                    g += input[offset].g * kernel[i + 1][j + 1];
                    b += input[offset].b * kernel[i + 1][j + 1];
                }
            }
        }

        // Normalize the pixel values and write to output
        output[idx].r = fminf(fmaxf(r, 0), 255);
        output[idx].g = fminf(fmaxf(g, 0), 255);
        output[idx].b = fminf(fmaxf(b, 0), 255);
    }
}

__global__ void gamma_correction_kernel(rgb* input, rgb* output, int width, int height) {

    float gamma = 1.5;
    // Get the index of the current thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    // Check if the thread is within the bounds of the image
    if (x < width && y < height) {

        // Apply gamma correction to each color channel
        output[idx].r = pow(input[idx].r / 255.0, gamma) * 255;
        output[idx].g = pow(input[idx].g / 255.0, gamma) * 255;
        output[idx].b = pow(input[idx].b / 255.0, gamma) * 255;
    }
}
#define FILTER_WIDTH    3       
#define FILTER_HEIGHT   3       
__global__ void sharpeningFilter(rgb *srcImage, rgb *dstImage, unsigned int width, unsigned int height)
{
   int x = blockIdx.x*blockDim.x + threadIdx.x;
   int y = blockIdx.y*blockDim.y + threadIdx.y;

   float kernel[FILTER_WIDTH][FILTER_HEIGHT] = {-1, -1, -1, -1, 9, -1, -1, -1, -1};
   // only threads inside image will write results
   if((x>=FILTER_WIDTH/2) && (x<(width-FILTER_WIDTH/2)) && (y>=FILTER_HEIGHT/2) && (y<(height-FILTER_HEIGHT/2)))
   {
      // Loop inside the filter to compute the sharpened pixel values
      for(int c=0 ; c<3 ; c++)   
      {
         float sum = 0;
         for(int ky=-FILTER_HEIGHT/2; ky<=FILTER_HEIGHT/2; ky++) {
            for(int kx=-FILTER_WIDTH/2; kx<=FILTER_WIDTH/2; kx++) {
               float fl = 0;
               if (c == 0) {
                   fl = srcImage[(y+ky)*width + (x+kx)].r;
               } else if (c == 1) {
                   fl = srcImage[(y+ky)*width + (x+kx)].g;
               } else if (c == 2) {
                   fl = srcImage[(y+ky)*width + (x+kx)].b;
               }
               sum += fl*kernel[ky+FILTER_HEIGHT/2][kx+FILTER_WIDTH/2];
            }
         }
         // Update the sharpened pixel values in the output image
         if (sum > 255) {
             sum = 255;
         } else if (sum < 0) {
             sum = 0;
         }
         if (c == 0) {
             dstImage[y*width+x].r = sum;
         } else if (c == 1) {
             dstImage[y*width+x].g = sum;
         } else if (c == 2) {
             dstImage[y*width+x].b = sum;
         }
      }
   }
}
int main() {

    FILE* input_file = fopen("input.txt", "r");

    // first line contains width and height
    int width, height;

    fscanf(input_file, "%d %d\n", &width, &height);
    
    printf("%d %d\n", width, height);

    // remaining lines contain rgb values
    rgb *input = (rgb*)malloc(width * height * sizeof(rgb));

    for (int i = 0; i < width * height; i++) {
        fscanf(input_file, "%d %d %d\n", &input[i].r, &input[i].g, &input[i].b);
        //printf("%d %d %d \n", input[i].r, input[i].g, input[i].b);
    }

    fclose(input_file);

    rgb *d_input;
    cudaMalloc((void**)&d_input, width * height * sizeof(rgb));
    cudaMemcpy(d_input, input, width * height * sizeof(rgb), cudaMemcpyHostToDevice);

    rgb *d_output;
    cudaMalloc((void**)&d_output, width * height * sizeof(rgb));

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

    //denoise_kernel<<<dimGrid, dimBlock>>>(d_input, d_output, width, height);
    //contrastEnhancementLuminosity<<<dimGrid, dimBlock>>>(d_input, d_output, width, height);
    //histogram_equalization_kernel<<<dimGrid, dimBlock>>>(d_input, d_output, width, height);
    //image_sharpening_kernel<<<dimGrid, dimBlock>>>(d_input, d_output, width, height);
    //gamma_correction_kernel<<<dimGrid, dimBlock>>>(d_input, d_output, width, height);
    //super_resolution_kernel<<<dimGrid, dimBlock>>>(d_input, d_output, width, height);
    //boxFilter<<<dimGrid,dimBlock>>>(d_input, d_output, width, height);
     sharpeningFilter<<<dimGrid,dimBlock>>>(d_input, d_output, width, height);
    rgb *output = (rgb*)malloc(width * height * sizeof(rgb) );
    cudaMemcpy(output, d_output, width * height * sizeof(rgb), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    FILE* output_file = fopen("output.txt", "w");

    // write width and height 
    fprintf(output_file, "%d %d\n", width, height);

    for (int i = 0; i < width * height; i++) {
        fprintf(output_file, "%d %d %d\n", output[i].r, output[i].g, output[i].b);
    }

    fclose(output_file);

    free(input);
    free(output);

    return 0;
}