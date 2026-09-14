/* Included AFTER an unchanged copy of native/x86/native_kernels.c. */
/* Raw history remains raw. Only its necessary pre-Snake is evaluated; no
 * depthwise or post-Snake history outputs are calculated and discarded.
 * Original choose_dw/snake_tile, FP32 tap order, SLEEF and 256 tile are reused.
 * This private entry point does not update history or allocate persistent state.
 */
NCC_API int32_t a2_raw_history_triple_f32(
    const float *x, const float *history, const float *weights, const float *bias,
    const float *alpha_pre, const float *reciprocal_pre,
    const float *alpha_post, const float *reciprocal_post, float *y,
    int64_t B, int64_t C, int64_t T, int32_t dilation,
    int32_t backend, int32_t threads) {
    if (dilation != 1 && dilation != 3 && dilation != 9) return NCC_INVALID_ARGUMENT;
    if (backend != NCC_AVX512 || threads != 1 || ncc_streaming_math_version(backend) != 1)
        return NCC_UNSUPPORTED_BACKEND;
    uint64_t rows, weight_count, history_count;
    size_t data_bytes, channel_bytes, weight_bytes, history_bytes;
    int status = validate_dimensions(B,C,T,backend,threads,&rows,&data_bytes,&channel_bytes);
    if (status || !rows) return status;
    if ((status=valid_pointer(y,data_bytes))) return status;
    if ((status=multiply((uint64_t)C,7,&weight_count))) return status;
    if ((status=checked_bytes(weight_count,&weight_bytes))) return status;
    if ((status=multiply(rows,6*(uint64_t)dilation,&history_count))) return status;
    if ((status=checked_bytes(history_count,&history_bytes))) return status;
    if ((status=check_read(x,data_bytes,y,data_bytes))) return status;
    if ((status=check_read(history,history_bytes,y,data_bytes))) return status;
    if ((status=check_read(weights,weight_bytes,y,data_bytes))) return status;
    const float *coefficients[5] = {bias,alpha_pre,reciprocal_pre,alpha_post,reciprocal_post};
    for (int i=0;i<5;++i)
        if ((status=check_read(coefficients[i],channel_bytes,y,data_bytes))) return status;
    dw_range_fn dw = choose_dw(backend);
    scale_fn scale; finish_fn finish;
    choose_snake(backend,&scale,&finish);
    const int halo=6*dilation;
    for (int64_t row=0;row<(int64_t)rows;++row) {
        const int64_t c=row%C;
        float transformed[54+NCC_TILE];
        float depthwise[NCC_TILE];
        snake_tile(history+row*halo,transformed,alpha_pre[c],reciprocal_pre[c],halo,
                   backend,scale,finish);
        for (int64_t t=0;t<T;t+=NCC_TILE) {
            const int n=(T-t<NCC_TILE)?(int)(T-t):NCC_TILE;
            snake_tile(x+row*T+t,transformed+halo,alpha_pre[c],reciprocal_pre[c],n,
                       backend,scale,finish);
            dw(transformed,NULL,weights+c*7,bias+c,depthwise,halo,n,dilation);
            snake_tile(depthwise,y+row*T+t,alpha_post[c],reciprocal_post[c],n,
                       backend,scale,finish);
            memmove(transformed,transformed+n,(size_t)halo*sizeof(float));
        }
    }
    return NCC_OK;
}
