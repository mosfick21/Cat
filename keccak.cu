// Keccak-256, Ethereum padding. One 116-byte packed input per candidate.
//
// No system header: this is compiled by NVRTC, which has no include path of
// its own, so <stdint.h> fails outright on some CUDA installs. The one type
// needed is spelled out instead.
typedef unsigned long long uint64_t;
#ifdef HOST_TEST
#define DEVICE
#else
#define DEVICE __device__ __forceinline__
#endif
DEVICE uint64_t rot(uint64_t x, int n) { return n ? (x << n) | (x >> (64-n)) : x; }
DEVICE uint64_t swap64(uint64_t x) {
 x=((x&0x00ff00ff00ff00ffULL)<<8)|((x>>8)&0x00ff00ff00ff00ffULL);
 x=((x&0x0000ffff0000ffffULL)<<16)|((x>>16)&0x0000ffff0000ffffULL);
 return (x<<32)|(x>>32);
}
DEVICE void digest(const uint64_t* base, uint64_t nonce, uint64_t* out) {
 const uint64_t rc[24]={0x1ULL,0x8082ULL,0x800000000000808aULL,0x8000000080008000ULL,0x808bULL,0x80000001ULL,0x8000000080008081ULL,0x8000000000008009ULL,0x8aULL,0x88ULL,0x80008009ULL,0x8000000aULL,0x8000808bULL,0x800000000000008bULL,0x8000000000008089ULL,0x8000000000008003ULL,0x8000000000008002ULL,0x8000000000000080ULL,0x800aULL,0x800000008000000aULL,0x8000000080008081ULL,0x8000000000008080ULL,0x80000001ULL,0x8000000080008008ULL};
 const int rho[25]={0,1,62,28,27,36,44,6,55,20,3,10,43,25,39,41,45,15,21,8,18,2,61,56,14};
 uint64_t s[25]={0},c[5],d[5],b[25];
 #pragma unroll
 for(int i=0;i<17;i++)s[i]=base[i];
 uint64_t n=swap64(nonce);
 s[5]=(s[5]&0xffffffffULL)|(n<<32);
 s[6]=(s[6]&0xffffffff00000000ULL)|(n>>32);
 for(int r=0;r<24;r++) {
  #pragma unroll
  for(int x=0;x<5;x++)c[x]=s[x]^s[x+5]^s[x+10]^s[x+15]^s[x+20];
  #pragma unroll
  for(int x=0;x<5;x++)d[x]=c[(x+4)%5]^rot(c[(x+1)%5],1);
  #pragma unroll
  for(int y=0;y<5;y++) {
   #pragma unroll
   for(int x=0;x<5;x++)b[y+5*((2*x+3*y)%5)]=rot(s[x+5*y]^d[x],rho[x+5*y]);
  }
  #pragma unroll
  for(int y=0;y<5;y++) {
   #pragma unroll
   for(int x=0;x<5;x++)s[x+5*y]=b[x+5*y]^((~b[(x+1)%5+5*y])&b[(x+2)%5+5*y]);
  }
  s[0]^=rc[r];
 }
 #pragma unroll
 for(int i=0;i<4;i++)out[i]=swap64(s[i]);
}
#ifdef HOST_TEST
extern "C" void host_digest(const uint64_t* base,uint64_t nonce,uint64_t* out){digest(base,nonce,out);}
#else
extern "C" __global__ void check(const uint64_t* base,uint64_t nonce,uint64_t* out){digest(base,nonce,out);}
extern "C" __global__ void search(const uint64_t* base,const uint64_t* target,uint64_t start,unsigned int count,unsigned int* found,uint64_t* result){
 unsigned int id=blockDim.x*blockIdx.x+threadIdx.x;
 if(id>=count)return;
 uint64_t h[4];digest(base,start+id,h);
 bool pass=false;
 #pragma unroll
 for(int i=0;i<4;i++){if(h[i]<target[i]){pass=true;break;}if(h[i]>target[i])break;}
 if(pass && atomicCAS(found,0U,1U)==0U)*result=start+id;
}
#endif
