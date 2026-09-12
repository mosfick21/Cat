"""Original SIMD Keccak generator for native CPU mining (GCC/Clang)."""
from core import RC, RHO


def source(width=1):
    if width not in (1, 4, 8):
        raise ValueError('Supported SIMD widths: 1, 4, 8')
    lines = ['#include <cstdint>', '#include <cstring>', '#include <atomic>',
             '#include <omp.h>', 'using u64=uint64_t;', f'constexpr int W={width};',
             'using V=u64;' if width == 1 else f'using V=u64 __attribute__((vector_size({width*8})));',
             'inline V splat(u64 x){return x;}' if width == 1 else
             'inline V splat(u64 x){V v;for(int i=0;i<W;i++)v[i]=x;return v;}',
             'inline V rol(V x,int n){return n?((x<<n)|(x>>(64-n))):x;}',
             'inline V swap(V x){',
             'x=((x&splat(0x00ff00ff00ff00ffULL))<<8)|((x>>8)&splat(0x00ff00ff00ff00ffULL));',
             'x=((x&splat(0x0000ffff0000ffffULL))<<16)|((x>>16)&splat(0x0000ffff0000ffffULL));',
             'return (x<<32)|(x>>32);}',
             'inline void hash_group(const u64* base,u64 start,u64 out[4][W]){',
             'const u64 rc[24]={'+','.join(hex(x)+'ULL' for x in RC)+'};']
    lines += [f'V s{i}='+ (f'splat(base[{i}]);' if i<17 else 'splat(0);') for i in range(25)]
    lines += ['V n=start;' if width==1 else 'V n;for(int i=0;i<W;i++)n[i]=start+i;',
              'n=swap(n);s5=(s5&splat(0xffffffffULL))|(n<<32);',
              's6=(s6&splat(0xffffffff00000000ULL))|(n>>32);',
              'for(int r=0;r<24;r++){']
    for x in range(5):
        lines += [f'V c{x}='+'^'.join(f's{x+5*y}' for y in range(5))+';']
    for x in range(5):lines += [f'V d{x}=c{(x+4)%5}^rol(c{(x+1)%5},1);']
    for y in range(5):
        for x in range(5):
            lines += [f'V b{y+5*((2*x+3*y)%5)}=rol(s{x+5*y}^d{x},{RHO[x+5*y]});']
    for y in range(5):
        for x in range(5):
            lines += [f's{x+5*y}=b{x+5*y}^((~b{(x+1)%5+5*y})&b{(x+2)%5+5*y});']
    lines += ['s0^=splat(rc[r]);','}']
    for i in range(4):
        lines += [f'V out{i}=swap(s{i});std::memcpy(out[{i}],&out{i},sizeof(V));']
    lines += ['}', r'''
extern "C" int cpu_width(){return W;}
extern "C" void cpu_probe(const u64* base,u64 start,u64* output){
 u64 h[4][W];hash_group(base,start,h);
 for(int lane=0;lane<W;lane++)for(int k=0;k<4;k++)output[4*lane+k]=h[k][lane];
}
extern "C" unsigned cpu_search(const u64* base,const u64* target,u64 start,
                               u64 count,int threads,u64* nonces,unsigned capacity){
 std::atomic<unsigned> found{0};
 const u64 groups=(count+W-1)/W;
 #pragma omp parallel for schedule(static) num_threads(threads)
 for(u64 group=0;group<groups;group++){
  u64 h[4][W];hash_group(base,start+group*W,h);
  for(int lane=0;lane<W && group*W+lane<count;lane++){
   bool pass=false;
   for(int k=0;k<4;k++){
    if(h[k][lane]<target[k]){pass=true;break;}
    if(h[k][lane]>target[k])break;
   }
   if(pass){unsigned slot=found.fetch_add(1,std::memory_order_relaxed);
    if(slot<capacity)nonces[slot]=start+group*W+lane;}
  }
 }
 return found.load(std::memory_order_relaxed);
}
''']
    return '\n'.join(lines)
