// Shared FMA body (see gpu_backend.h): VT/ST select float2 or half2; 32 chains, loop body 4x.
// b and c are runtime values, so the compiler cannot fold or specialise the arithmetic.
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer Buf { float data[]; };
layout(push_constant) uniform Params { uint n; uint a; uint b; uint c; } p;

#define R4(M, x) M(x##0) M(x##1) M(x##2) M(x##3)
#define R16(M) R4(M, a0) R4(M, a1) R4(M, a2) R4(M, a3)
#define R32(M) R16(M) R4(M, a4) R4(M, a5) R4(M, a6) R4(M, a7)
#define DECL(v) VT v = s; s += d;
#define FMA(v)  v = fma(v, b, c);
#define SUM(v)  r += v;

void main() {
    uint tid = gl_GlobalInvocationID.x;
    VT s = VT(ST(data[tid]));
    VT b = VT(ST(uintBitsToFloat(p.b)));
    VT c = VT(ST(uintBitsToFloat(p.c)));
    VT d = VT(ST(0.001));
    R32(DECL)
    for (uint i = 0u; i < p.n; i++) { R32(FMA) R32(FMA) R32(FMA) R32(FMA) }
    VT r = VT(ST(0.0));
    R32(SUM)
    data[tid] = float(r.x) + float(r.y);
}
