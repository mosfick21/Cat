"""Generate scalar Keccak kernels with fixed lane indexes and no system headers."""
from core import RC, RHO, interleave

PRELUDE = r'''
typedef unsigned long long u64;
typedef unsigned int u32;
static_assert(sizeof(u64)==8 && sizeof(u32)==4,"integer width mismatch");
#ifdef HOST_TEST
#define HD inline
#else
#define HD __device__ __forceinline__
#endif
HD u64 rol64(u64 x,int n){return n?((x<<n)|(x>>(64-n))):x;}
HD u32 rol32(u32 x,int n){return n?((x<<n)|(x>>(32-n))):x;}
HD u64 bswap(u64 x){
 x=((x&0x00ff00ff00ff00ffULL)<<8)|((x>>8)&0x00ff00ff00ff00ffULL);
 x=((x&0x0000ffff0000ffffULL)<<16)|((x>>16)&0x0000ffff0000ffffULL);
 return (x<<32)|(x>>32);
}
HD u32 compact(u32 x){
 x&=0x55555555U;x=(x|(x>>1))&0x33333333U;
 x=(x|(x>>2))&0x0f0f0f0fU;x=(x|(x>>4))&0x00ff00ffU;
 return (x|(x>>8))&0xffffU;
}
HD u32 spread(u32 x){
 x&=0xffffU;x=(x|(x<<8))&0x00ff00ffU;
 x=(x|(x<<4))&0x0f0f0f0fU;x=(x|(x<<2))&0x33333333U;
 return (x|(x<<1))&0x55555555U;
}
HD u64 join_lane(u32 e,u32 o){
 u32 lo=spread(e)|(spread(o)<<1);
 u32 hi=spread(e>>16)|(spread(o>>16)<<1);
 return bswap(((u64)hi<<32)|lo);
}
'''


def scalar64(unroll=1, lanes=(5, 6)):
    """Fully unrolled, one named register per lane - no arrays anywhere.

    `lanes` is where the nonce's low 64 bits sit, which depends on what the
    contract hashes: (5, 6) for a message that opens with a 20-byte address,
    (9, 10) for one that opens with a 32-byte seed. Everything else is the
    same Keccak.
    """
    hi, lo = lanes
    lines=['HD void scalar64(const u64* base,u64 nonce,u64* out){']
    lines += ['const u64 rc[24]={'+','.join(hex(x)+'ULL' for x in RC)+'};']
    lines += [f'u64 s{i}='+ (f'base[{i}];' if i<17 else '0;') for i in range(25)]
    lines += ['u64 n=bswap(nonce);',
              f's{hi}=(s{hi}&0xffffffffULL)|(n<<32);',
              f's{lo}=(s{lo}&0xffffffff00000000ULL)|(n>>32);',
              f'#pragma unroll {unroll}', 'for(int r=0;r<24;r++){']
    for x in range(5):
        lines += [f'u64 c{x}='+'^'.join(f's{x+5*y}' for y in range(5))+';']
    for x in range(5):lines += [f'u64 d{x}=c{(x+4)%5}^rol64(c{(x+1)%5},1);']
    for y in range(5):
        for x in range(5):
            lines += [f'u64 b{y+5*((2*x+3*y)%5)}=rol64(s{x+5*y}^d{x},{RHO[x+5*y]});']
    for y in range(5):
        for x in range(5):
            lines += [f's{x+5*y}=b{x+5*y}^((~b{(x+1)%5+5*y})&b{(x+2)%5+5*y});']
    lines += ['s0^=rc[r];','}']
    lines += [f'out[{i}]=bswap(s{i});' for i in range(4)]
    return '\n'.join(lines+['}'])


def interleaved32(unroll=1):
    erc,orc=zip(*(interleave(x) for x in RC))
    lines=['HD void interleaved32(const u32* base,u64 nonce,u64* out){']
    for name,rc in [('e',erc),('o',orc)]:
        lines += [f'const u32 rc{name}[24]={{'+','.join(hex(x)+'U' for x in rc)+'};']
    for i in range(25):
        lines += [f'u32 e{i}='+ (f'base[{2*i}];' if i<17 else '0;'),
                  f'u32 o{i}='+ (f'base[{2*i+1}];' if i<17 else '0;')]
    lines += ['u64 n=bswap(nonce);u32 lo=(u32)n,hi=(u32)(n>>32);',
              'e5=(e5&0xffffU)|(compact(lo)<<16);o5=(o5&0xffffU)|(compact(lo>>1)<<16);',
              'e6=(e6&0xffff0000U)|compact(hi);o6=(o6&0xffff0000U)|compact(hi>>1);',
              f'#pragma unroll {unroll}', 'for(int r=0;r<24;r++){']
    for part in ('e','o'):
        for x in range(5):lines += [f'u32 c{part}{x}='+'^'.join(f'{part}{x+5*y}' for y in range(5))+';']
    for x in range(5):
        lines += [f'u32 de{x}=ce{(x+4)%5}^rol32(co{(x+1)%5},1);',
                  f'u32 do{x}=co{(x+4)%5}^ce{(x+1)%5};']
    for y in range(5):
        for x in range(5):
            i=x+5*y;j=y+5*((2*x+3*y)%5);rot=RHO[i]
            if rot%2:
                lines += [f'u32 be{j}=rol32(o{i}^do{x},{rot//2+1});',
                          f'u32 bo{j}=rol32(e{i}^de{x},{rot//2});']
            else:
                lines += [f'u32 be{j}=rol32(e{i}^de{x},{rot//2});',
                          f'u32 bo{j}=rol32(o{i}^do{x},{rot//2});']
    for part in ('e','o'):
        for y in range(5):
            for x in range(5):
                lines += [f'{part}{x+5*y}=b{part}{x+5*y}^((~b{part}{(x+1)%5+5*y})&b{part}{(x+2)%5+5*y});']
    lines += ['e0^=rce[r];o0^=rco[r];','}']
    lines += [f'out[{i}]=join_lane(e{i},o{i});' for i in range(4)]
    return '\n'.join(lines+['}'])


def wrapper(name, word):
    return f'''
#ifdef HOST_TEST
extern "C" void host_{name}(const {word}* base,u64 nonce,u64* out){{{name}(base,nonce,out);}}
#else
extern "C" __global__ void probe_{name}(const {word}* base,u64 start,u32 count,u64* hashes){{
 u32 i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<count){name}(base,start+i,hashes+4*i);
}}
extern "C" __global__ void search_{name}(const {word}* base,const u64* target,u64 start,u32 count,u32* found,u64* result){{
 for(u64 i=(u64)blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=(u64)gridDim.x*blockDim.x){{
  u64 h[4];{name}(base,start+i,h);
  bool pass=false;
  #pragma unroll
  for(int k=0;k<4;k++){{if(h[k]<target[k]){{pass=true;break;}}if(h[k]>target[k])break;}}
  if(pass){{u32 slot=atomicAdd(found,1U);if(slot<64U)result[slot]=start+i;}}
 }}
}}
#endif
'''


def source(kind='scalar64', unroll=1, lanes=(5, 6)):
    if kind=='scalar64':return PRELUDE+scalar64(unroll, lanes)+wrapper(kind,'u64')
    if kind=='interleaved32':return PRELUDE+interleaved32(unroll)+wrapper(kind,'u32')
    raise ValueError(kind)
