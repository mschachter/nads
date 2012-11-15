#pragma OPENCL EXTENSION cl_amd_printf : enable

__kernel void unit_step(__global const int *state_index, __global const int *param_index,
                        __global const float *state, __global const float *params,
                        __global const int *weight_index, __global const int *conn_index,
                        __global const int *num_conns, __global const float *weights,
                        __global float *next_state,
                        const float step_size,
                        __global float4 *color)
{
    const uint gpu_index = get_global_id(0);
	const uint sindex = state_index[gpu_index];
	const uint pindex = param_index[gpu_index];

	const float R = params[pindex]; /* resistance */
	const float C = params[pindex+1]; /* capacitance */
	const float vthresh = params[pindex+2];  /* membrane potential threshold */
	const float vreset = params[pindex+3];   /* post-spike membrane potential reset */
	const float syntau = params[pindex+4];   /* synaptic time constant */

    const float output = state[sindex];
	const float v = state[sindex+1];  /* membrane potential */
	const float spike = state[sindex+2]; /* binary indicator for spike or no spike */

    const uint windex = weight_index[gpu_index];
    const uint nconn = num_conns[gpu_index];

    /*
    printf("gpu_index=%d, sindex=%d, pindex=%d, windex=%d, num_conns=%d, step_size=%f\nR=%f,C=%f,vthresh=%f,vreset=%f, v=%f\n",
            gpu_index, sindex, pindex, windex, nconn, step_size, R, C, vthresh, vreset, state[sindex]);
    */

    if (v > vthresh) {
        /* spike has occurred, reset membrane potential and set spike state to 1 */
        next_state[sindex+1] = vreset;
        next_state[sindex+2] = 1.0f;

        color[gpu_index].x = 1.0;
        color[gpu_index].y = 1.0;
        color[gpu_index].z = 1.0f;
        color[gpu_index].w = 1.0f;

    } else {
        /* integrate synaptic input */

        float input = 0.0f;
        float pre_state;
        int pre_index, pre_sindex;
        for (int k = 0; k < nconn; k++) {
            pre_index = conn_index[windex+k]; /* get unit number of presynaptic neuron */
            pre_sindex = state_index[pre_index]; /* get the index of that unit's state in the state vector */
            pre_state = state[pre_sindex];  /* second state value is spike or no spike */
/*
            printf("gpu_index=%d, windex=%d, nconn=%d, weights[%d]=%0.6f, pre_index=%d, pre_sindex=%d, input state=%0.6f\n",
                   gpu_index, windex, nconn, k, weights[windex+k], pre_index, pre_sindex, pre_state);
*/

            input += weights[windex+k]*pre_state;
        }

        /* update membrane potential and spike state */
        next_state[sindex+1] = v + step_size*( (-v / (R*C)) + (input / C));
        next_state[sindex+2] = 0.0f;

        /* update color */
        if (v > 0.0f) {
            float clr = v / (vthresh - vreset);
            color[gpu_index].x = clr;
            color[gpu_index].y = 0.0;
            color[gpu_index].z = 1.0f - clr;
            color[gpu_index].w = 1.0f;
        } else {
            color[gpu_index].x = 0.0;
            color[gpu_index].y = 0.0;
            color[gpu_index].z = 1.0f;
            color[gpu_index].w = 1.0f;
        }
    }

    /* update synaptic output */
    next_state[sindex] = output*(1 - syntau*step_size) + step_size*spike;
}
