#include <windows.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "neural_net.h"
#include "stb_image.h"

#define MAX_EPOCH 501

static inline void setup_console() {
    SetConsoleOutputCP(CP_UTF8);
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD dwMode = 0;
    GetConsoleMode(hOut, &dwMode);
    SetConsoleMode(hOut, dwMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
}

void makeBMP(const char* filename, int width, int height, const unsigned char* rgb) {
    FILE* f = fopen(filename, "wb");
    if (!f) return;

    int rowSize = (3 * width + 3) & ~3;  // pad to 4 bytes
    int dataSize = rowSize * height;
    int fileSize = 54 + dataSize;

    unsigned char header[54] = {
        'B','M',
        0,0,0,0,    // file size
        0,0, 0,0,
        54,0,0,0,   // pixel data offset
        40,0,0,0,   // DIB header size
        0,0,0,0,    // width
        0,0,0,0,    // height
        1,0,        // planes
        24,0,       // bpp
        0,0,0,0,    // compression
        0,0,0,0,    // image size
        0x13,0x0B,0,0,
        0x13,0x0B,0,0,
        0,0,0,0,
        0,0,0,0
    };

    // file size
    header[2] = fileSize;
    header[3] = fileSize >> 8;
    header[4] = fileSize >> 16;
    header[5] = fileSize >> 24;

    // width
    header[18] = width;
    header[19] = width >> 8;
    header[20] = width >> 16;
    header[21] = width >> 24;

    // height
    header[22] = height;
    header[23] = height >> 8;
    header[24] = height >> 16;
    header[25] = height >> 24;

    fwrite(header, 1, 54, f);

    unsigned char* row = malloc(rowSize);

    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            int i = (y * width + x) * 3;
            row[x * 3 + 0] = rgb[i + 2]; // B
            row[x * 3 + 1] = rgb[i + 1]; // G
            row[x * 3 + 2] = rgb[i + 0]; // R
        }
        for (int p = width * 3; p < rowSize; p++) row[p] = 0;
        fwrite(row, 1, rowSize, f);
    }

    free(row);
    fclose(f);
}

static unsigned char* load_image(const char* path, int newW, int newH) {
    int w, h, channels;
    unsigned char* input = stbi_load(path, &w, &h, &channels, 3);
    unsigned char* output = malloc(newW * newH * 3);

    if (!(input && output))exit(1);

    for (int y = 0; y < newH; y++) {
        for (int x = 0; x < newW; x++) {

            int srcX = x * w / newW;
            int srcY = y * h / newH;

            int srcIndex = (srcY * w + srcX) * 3;
            int dstIndex = (y * newW + x) * 3;

            output[dstIndex + 0] = input[srcIndex + 0];
            output[dstIndex + 1] = input[srcIndex + 1];
            output[dstIndex + 2] = input[srcIndex + 2];
        }
    }

    stbi_image_free(input);
    return output;
}

static int load_all_images(const char* folder, unsigned char*** out_imgs, int* count, int width, int height) {
    WIN32_FIND_DATAA fd;
    char search_path[MAX_PATH];
    snprintf(search_path, MAX_PATH, "%s\\*.png", folder);

    HANDLE h = FindFirstFileA(search_path, &fd);
    if (h == INVALID_HANDLE_VALUE) return 0;

    unsigned char** imgs = NULL;
    int img_count = 0;

    do {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            char fullpath[MAX_PATH];
            snprintf(fullpath, MAX_PATH, "%s\\%s", folder, fd.cFileName);

            unsigned char* img = (unsigned char*)load_image(fullpath, width, height);
            imgs = realloc(imgs, sizeof(unsigned char*) * (img_count + 1));
            if (!imgs)exit(1);
            imgs[img_count++] = img;
        }
    } while (FindNextFileA(h, &fd));

    FindClose(h);

    *out_imgs = imgs;
    *count = img_count;
    return 1;
}

static void draw_image(const unsigned char* img, int w, int h, int width, int height) {
    if (w < 0) {
        w = width;
    }
    if (h < 0) {
        h = width;
    }
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int srcX = x * width / w;
            int srcY = y * height / h;
            int idx = (srcY * width + srcX) * 3;

            int r = img[idx + 0];
            int g = img[idx + 1];
            int b = img[idx + 2];
            printf("\033[38;2;%d;%d;%dm██", r, g, b);
        }
        printf("\033[0m\n");
    }
}

static void draw_image_from_path(char* path, unsigned int w, unsigned int h) {
    int width, height, channels;
    unsigned char* img = stbi_load(path, &width, &height, &channels, 3);
    if (!img) exit(1);
    draw_image(img, w, h, width, height);
    stbi_image_free(img);
}

static float* normalize_rgb(unsigned char* img, unsigned int size) {
    float* img_float = (float*)malloc(size * sizeof(float));
    if (!img_float)exit(1);
    for (unsigned int i = 0; i < size; i++) {
        img_float[i] = img[i] / 255.0f;
    }
    return img_float;
}

static unsigned char* unnormalize_rgb(const float* img_float, unsigned int size) {
    unsigned char* out = malloc(size * sizeof(unsigned char));
    if (!out) exit(1);
    for (unsigned int i = 0; i < size; i++) {
        float v = img_float[i];
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        v = v * 255.0f;
        out[i] = (unsigned char)(v + 0.5f);
    }

    return out;
}

static inline void progress_bar(int a, int b, int length) {
    printf("[");
    for (int i = 0; i < length; i++) {
        if ((i + 0.5f) * (float)a / (float)length < b) {
            printf("█");
        }
        else {
            printf(":");
        }
    }
    printf("]");
}

int main() {
    srand((unsigned int)time(NULL));

    unsigned int width = 64;
    unsigned int height = 64;

    neural_net betavae;
    neural_net_init(&betavae, 5, (unsigned int[]) { width * height * 3, 512, 256, 512, width * height * 3 });

    setup_console();

    unsigned char** imgs;
    int img_count;

    if (!load_all_images("images", &imgs, &img_count, width, height)) {
        printf("No images found.\n");
        return 1;
    }

    float** imgs_float = malloc(sizeof(float*) * img_count);

    if (!imgs_float)exit(1);
    for (int i = 0; i < img_count; i++) {
        imgs_float[i] = normalize_rgb(imgs[i], width * height * 3);
    }

    float losses[MAX_EPOCH];
    for (int epoch = 0; epoch < MAX_EPOCH; epoch++) {
        float error = 0;
        for (int i = 0; i < img_count; i++) {
            forward_prop(&betavae, imgs_float[i]);
            error += backward_prop(&betavae, imgs_float[i], 0.001f) / img_count;
        }
        losses[epoch] = error;
        int draw_image_index = 6 /*rand() % imag_count*/;

        forward_prop(&betavae, imgs_float[draw_image_index]);
        float* layer = get_layer(&betavae, betavae.size - 1);
        unsigned char* draw_img = unnormalize_rgb(layer, width * height * 3);

        if (epoch % 10 == 0) {
            unsigned char* bmp = (unsigned char*)malloc(width * height * 6 * sizeof(unsigned char));
            if (!bmp)exit(1);
            memcpy(bmp, imgs[draw_image_index], width * height * 3);
            memcpy(bmp + width * height * 3, draw_img, width* height * 3);
            char name[32];
            snprintf(name, sizeof(name), "image_output/%04d.bmp", epoch);
            makeBMP(name, width, 2 * height, bmp);
            free(bmp);
        }

        printf("\033[H");

        draw_image(imgs[draw_image_index], width, height, width, height);
        draw_image(draw_img, width, height, width, height);

        printf("\n");
        progress_bar(MAX_EPOCH, epoch, 20);
        printf(" epoch: %d loss: %f", epoch + 1, error);

        free(layer);
        free(draw_img);
    }

    for (int i = 0; i < img_count; i++) {
        free(imgs[i]);
        free(imgs_float[i]);
    }

    system("cls");
    for (int i = 0; i < MAX_EPOCH; i++) {
        printf("%f, ", losses[i]);
        Beep(losses[i] * 1000, 2);
    }

    free(imgs);
    free(imgs_float);
    free(betavae.structure);
    free(betavae.weights);
    free(betavae.bias);
    free(betavae.values);
    free(betavae.zvalues);

    return 0;
}