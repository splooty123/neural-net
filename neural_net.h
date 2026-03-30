#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

typedef struct {
    unsigned int size;
    unsigned int* structure;
    float* weights;
    float* bias;
    float* values;
    float* zvalues;
} neural_net;

static void neural_net_init(neural_net* a, unsigned int len, unsigned int structure[]) {
    a->size = len;
    a->structure = (unsigned int*)malloc(sizeof(unsigned int) * len);
    for (unsigned int i = 0; i < len; i++) {
        a->structure[i] = structure[i];
    }

    unsigned int weight_acc = 0;
    unsigned int bias_acc = 0;
    unsigned int value_acc = 0;

    for (unsigned int i = 1; i < len; i++) {
        weight_acc += structure[i - 1] * structure[i];
        bias_acc += structure[i];
    }

    value_acc = bias_acc + structure[0];
    a->weights = (float*)malloc(weight_acc * sizeof(float));
    a->bias = (float*)malloc(bias_acc * sizeof(float));
    a->values = (float*)malloc(value_acc * sizeof(float));
    a->zvalues = (float*)malloc(value_acc * sizeof(float));

    if (!(a->weights && a->bias && a->values && a->zvalues)) { exit(1); }

    float scale = sqrtf(2.0f / (structure[0] + structure[a->size - 1]));
    for (unsigned int i = 0; i < weight_acc; i++) {
        a->weights[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
    }
    for (unsigned int i = 0; i < bias_acc; i++) {
        a->bias[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
    }
}

static float* get_layer(neural_net* a, unsigned int n) {
    float* ret = (float*)malloc(a->structure[n] * sizeof(float));
    if (!ret) exit(1);
    unsigned int value_offset = 0;
    for (unsigned int i = 0; i < n; i++) {
        value_offset += a->structure[i];
    }
    for (unsigned int i = 0; i < a->structure[n]; i++) {
        ret[i] = a->values[i + value_offset];
    }
    return ret;
}

static inline float activation(float x) {
    return tanh(x);
}

// x here is the *activation*, not the pre-activation
static inline float activation_deriv(float a) {
    // derivative of tanh(x) is 1 - tanh(x)^2
    // but since a = tanh(z), we can use 1 - a^2
    return 1.0f - a * a;
}

static void forward_prop(neural_net* a, const float* input) {
    unsigned int L = a->size;

    unsigned int input_size = a->structure[0];
    for (unsigned int i = 0; i < input_size; i++) {
        a->values[i] = input[i];
        a->zvalues[i] = input[i];
    }

    unsigned int weight_off = 0;
    unsigned int bias_off = 0;
    unsigned int value_off = input_size;

    for (unsigned int layer = 1; layer < L; layer++) {

        unsigned int prev = a->structure[layer - 1];
        unsigned int curr = a->structure[layer];

        float* prev_vals = &a->values[value_off - prev];
        float* curr_vals = &a->values[value_off];
        float* curr_z = &a->zvalues[value_off];

        float* W = &a->weights[weight_off];
        float* B = &a->bias[bias_off];

#pragma omp parallel for
        for (int j = 0; j < (int)curr; j++) {
            float sum = B[j];

            for (unsigned int i = 0; i < prev; i++) {
                sum += W[j * prev + i] * prev_vals[i];
            }

            curr_z[j] = sum;

            if (layer == L - 1)
                curr_vals[j] = sum;          // linear output
            else
                curr_vals[j] = activation(sum);
        }

        weight_off += prev * curr;
        bias_off += curr;
        value_off += curr;
    }
}

static float backward_prop(neural_net* net, const float* target, float lr) {
    unsigned int L = net->size;

    unsigned int total = 0;
    for (unsigned int i = 0; i < L; i++)
        total += net->structure[i];

    float* delta = (float*)calloc(total, sizeof(float));
    if (!delta) exit(1);

    unsigned int* voff = (unsigned int*)malloc(L * sizeof(unsigned int));
    unsigned int* woff = (unsigned int*)malloc(L * sizeof(unsigned int));
    unsigned int* boff = (unsigned int*)malloc(L * sizeof(unsigned int));

    if (!voff || !woff || !boff) exit(1);

    unsigned int v_acc = 0, w_acc = 0, b_acc = 0;

    for (unsigned int i = 0; i < L; i++) {
        voff[i] = v_acc;
        v_acc += net->structure[i];

        if (i > 0) {
            woff[i] = w_acc;
            w_acc += net->structure[i - 1] * net->structure[i];

            boff[i] = b_acc;
            b_acc += net->structure[i];
        }
    }

    unsigned int out = L - 1;
    unsigned int out_size = net->structure[out];
    unsigned int out_vo = voff[out];

    // output delta = dL/da = (a - target) for MSE
#pragma omp parallel for
    for (int i = 0; i < (int)out_size; i++) {
        float a = net->values[out_vo + i];
        delta[out_vo + i] = (a - target[i]);
    }

    // backpropagate deltas
    for (int layer = (int)L - 1; layer > 0; layer--) {
        unsigned int curr = (unsigned int)layer;
        unsigned int prev = curr - 1;

        unsigned int curr_size = net->structure[curr];
        unsigned int prev_size = net->structure[prev];

        unsigned int curr_vo = voff[curr];
        unsigned int prev_vo = voff[prev];
        unsigned int w_off = woff[curr];

#pragma omp parallel for
        for (int i = 0; i < (int)prev_size; i++) {
            float sum = 0.0f;

            for (unsigned int j = 0; j < curr_size; j++) {
                float w = net->weights[w_off + j * prev_size + i];
                float d = delta[curr_vo + j];
                sum += w * d;
            }

            float a_prev = net->values[prev_vo + i];
            float deriv = activation_deriv(a_prev);
            delta[prev_vo + i] = sum * deriv;
        }
    }

    // gradient descent step
    for (unsigned int layer = 1; layer < L; layer++) {
        unsigned int curr_size = net->structure[layer];
        unsigned int prev_size = net->structure[layer - 1];

        unsigned int curr_vo = voff[layer];
        unsigned int prev_vo = voff[layer - 1];

        unsigned int w_off = woff[layer];
        unsigned int b_off = boff[layer];

#pragma omp parallel for
        for (int j = 0; j < (int)curr_size; j++) {
            float d = delta[curr_vo + j];

            for (unsigned int i = 0; i < prev_size; i++) {
                float a_prev = net->values[prev_vo + i];
                net->weights[w_off + j * prev_size + i] -= lr * d * a_prev;
            }

            net->bias[b_off + j] -= lr * d;
        }
    }

    // compute MSE for monitoring
    float error = 0.0f;
    float* layer_vals = get_layer(net, net->size - 1);
    unsigned int out_n = net->structure[net->size - 1];

    for (unsigned int i = 0; i < out_n; i++) {
        float diff = target[i] - layer_vals[i];
        error += (diff * diff) / (float)out_n;
    }

    free(delta);
    free(voff);
    free(woff);
    free(boff);
    free(layer_vals);

    return error;
}