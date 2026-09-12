// Keccak-256, Ethereum padding. One 116-byte packed input per candidate.
typedef unsigned long long hc_u64;
static_assert(sizeof(hc_u64) == 8, "64-bit integer required");
#ifdef HOST_TEST
#define DEVICE
#else
#define DEVICE __device__ __forceinline__
#endif
DEVICE hc_u64 rot(hc_u64 x, int n) { return n ? (x << n) | (x >> (64-n)) : x; }
DEVICE hc_u64 swap64(hc_u64 x) {
 x=((x&0x00ff00ff00ff00ffULL)<<8)|((x>>8)&0x00ff00ff00ff00ffULL);
 x=((x&0x0000ffff0000ffffULL)<<16)|((x>>16)&0x0000ffff0000ffffULL);
 return (x<<32)|(x>>32);
}
DEVICE void digest(const hc_u64* base, hc_u64 nonce, hc_u64* out) {
 const hc_u64 rc[24]={0x1ULL,0x8082ULL,0x800000000000808aULL,0x8000000080008000ULL,0x808bULL,0x80000001ULL,0x8000000080008081ULL,0x8000000000008009ULL,0x8aULL,0x88ULL,0x80008009ULL,0x8000000aULL,0x8000808bULL,0x800000000000008bULL,0x8000000000008089ULL,0x8000000000008003ULL,0x8000000000008002ULL,0x8000000000000080ULL,0x800aULL,0x800000008000000aULL,0x8000000080008081ULL,0x8000000000008080ULL,0x80000001ULL,0x8000000080008008ULL};
 const int rho[25]={0,1,62,28,27,36,44,6,55,20,3,10,43,25,39,41,45,15,21,8,18,2,61,56,14};
 hc_u64 s[25]={0},c[5],d[5],b[25];
 #pragma unroll
 for(int i=0;i<17;i++)s[i]=base[i];
 hc_u64 n=swap64(nonce);
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
extern "C" void host_digest(const hc_u64* base,hc_u64 nonce,hc_u64* out){digest(base,nonce,out);}
#else
extern "C" __global__ void check(const hc_u64* base,hc_u64 nonce,hc_u64* out){digest(base,nonce,out);}
extern "C" __global__ void search(const hc_u64* base,const hc_u64* target,hc_u64 start,unsigned int count,unsigned int* found,hc_u64* result){
 unsigned int id=blockDim.x*blockIdx.x+threadIdx.x;
 if(id>=count)return;
 hc_u64 h[4];digest(base,start+id,h);
 bool pass=false;
 #pragma unroll
 for(int i=0;i<4;i++){if(h[i]<target[i]){pass=true;break;}if(h[i]>target[i])break;}
 if(pass && atomicCAS(found,0U,1U)==0U)*result=start+id;
}
#endif
