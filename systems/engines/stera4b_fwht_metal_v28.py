"""128-long FP32 Walsh-Hadamard transform on Metal, independent vector blocks."""
import mlx.core as mx
KM_FWHT=mx.fast.metal_kernel(
 name='stera_fp32_fwht_vector_v28',
 input_names=['x'],
 output_names=['result'],
 source=r"""
 uint ri=thread_position_in_threadgroup.x;
 uint cb=threadgroup_position_in_grid.x;
 float a=x[cb*128u+ri];
 for(uint h=1u;h<=16u;h<<=1u){
    float b=simd_shuffle_xor(a,ushort(h));
    a=(ri & h)?(b-a):(a+b);
 }
 threadgroup float shv[128];
 shv[ri]=a;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 float b=shv[ri^32u];
 a=(ri&32u)?(b-a):(a+b);
 threadgroup_barrier(mem_flags::mem_threadgroup);
 shv[ri]=a;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 b=shv[ri^64u];
 a=(ri&64u)?(b-a):(a+b);
 result[cb*128u+ri]=a;
 """,compile_options={'math_mode':'fast'})
def fwht_metal(x):
    assert x.ndim==1 and x.size%128==0
    return KM_FWHT(inputs=[x],grid=(x.size,1,1),threadgroup=(128,1,1),
        output_shapes=[(x.size,)],output_dtypes=[mx.float32])[0]
